# VLA cloud inference on Azure

Serve a trained VLA policy (rung 2 — see [vla.md](vla.md)) from an always-on
Azure GPU VM instead of running it on the machine wired to Brachiomimus. This
covers the *deployment* half only — training the policy itself still goes
through [training-act.md](training-act.md); this doc picks up once a
checkpoint (e.g. a SmolVLA fine-tune) exists on the Hub to serve.

## Why cloud inference, and how it stays low-latency

LeRobot ships a built-in async inference mode for exactly this: a
**PolicyServer** (runs the model on a GPU box) and a **RobotClient** (runs on
the machine actually driving the arm and reading its cameras), talking gRPC.
The client requests a *chunk* of future actions at once and executes them
locally while the next chunk is being computed, so a network round-trip
doesn't stall every single tick the way naive per-step remote inference
would.

```
[Robot machine — local, wired to Brachiomimus]        [Azure VM — Standard_NC4as_T4_v3, 1x T4 16GB]
  lerobot.async_inference.robot_client                   lerobot.async_inference.policy_server
  - drives the arm via COM4                               - huggingface/lerobot-gpu container, --gpus all
  - reads front/wrist cameras                             - serves the SmolVLA checkpoint from the HF Hub
  - Tailscale node   <───── gRPC over the tailnet ─────>   Tailscale node
                            (never touches the public IP)
```

## ⚠️ Security: this must not be internet-reachable

LeRobot's PolicyServer has an **unpatched, unauthenticated remote-code-
execution vulnerability** (CVE-2026-25874): two of its gRPC handlers call
`pickle.loads()` on data that comes straight off the network, with no
authentication and no TLS. As of `lerobot` 0.4.3 there is no fix — the only
real mitigation is to keep the port off the public internet entirely.

That's the reason for every networking choice below:

- **[Tailscale](https://tailscale.com)** mesh VPN between the robot machine
  and the VM. Both sides get a private `100.x.y.z` address; the PolicyServer
  binds to *that* address, never `0.0.0.0` on the public NIC.
- **The Azure NSG denies all inbound traffic from the internet** — not even
  SSH. Tailscale doesn't need an inbound rule (it does its own NAT
  traversal), and admin access goes over `tailscale ssh` instead of public
  SSH.
- Revisit this doc once upstream ships a fix for CVE-2026-25874; until then,
  treat the PolicyServer as unsafe on any network you don't fully control.

## Quota and region strategy

The granted quota — 4 vCPUs in West US, 4 vCPUs in West US 2 — buys exactly
**one `Standard_NC4as_T4_v3` per region** (4 vCPU, a single 16 GB T4, 28 GiB
RAM): the smallest SKU in the NCasT4v3 family, and the ceiling without
requesting more quota.

Run one VM at a time, in whichever region is physically closer to wherever
Brachiomimus lives (round-trip latency still matters even with action
chunking). Leave the other region's quota unprovisioned as a **cold
standby** — redeploy the same script into it if the primary region turns out
not to have schedulable capacity (see gotcha below) or has an outage. This
also keeps cost to a single running VM.

> **Gotcha — vCPU quota ≠ GPU quota.** Azure tracks them as separate quota
> lines. A subscription can show 4/4 vCPUs for the NCasT4v3 family and still
> have 0 GPU quota, which fails VM creation with a confusing error unrelated
> to the vCPU number you already confirmed. `infra/azure/provision.sh` prints
> both lines from `az vm list-usage` and makes you confirm before it
> provisions anything.

> **Gotcha — a quota grant doesn't guarantee capacity.** Azure can approve a
> quota increase and still fail to schedule the VM if that region's GPU
> capacity is exhausted. This is the main reason to keep a second region's
> quota in reserve rather than requesting more in one region.

## One-time setup

### 1. Create a Tailscale auth key

In the [Tailscale admin console](https://login.tailscale.com/admin/settings/keys),
generate a reusable, tagged auth key (e.g. `tag:brachiomimus-inference`).
Treat it like the W&B key in this repo's `.env` — **never commit it**; it's
only ever passed as an environment variable at provision time (see
[training-act.md's authentication note](training-act.md#authentication-keys-live-in-env)
for the same pattern applied to `WANDB_API_KEY`).

### 2. Provision the VM

```bash
cd infra/azure
TAILSCALE_AUTHKEY=tskey-... ./provision.sh --location westus
```

```powershell
cd infra\azure
$env:TAILSCALE_AUTHKEY = "tskey-..."
./provision.sh --location westus   # run under Git Bash / WSL — this script is bash, not PowerShell
```

Add `--dry-run` first to see the exact `az` commands without creating
anything billed. The script:

1. Prints the quota lines for the region and asks you to confirm both are
   sufficient before continuing.
2. Creates a resource group and the VM (`Standard_NC4as_T4_v3`, Ubuntu
   22.04, **no inbound NSG rule at all** — see the security section above).
3. Runs `cloud-init-policy-server.yaml` on first boot: installs Docker, the
   NVIDIA container toolkit, and Tailscale, joins the tailnet, and
   pre-pulls the `huggingface/lerobot-gpu` image.
4. Attaches the Azure NVIDIA GPU driver extension (the container toolkit
   needs a real driver on the host underneath it).
5. Sets a 3am UTC auto-shutdown as a forgotten-VM safety net (see
   [Cost control](#cost-control) — this is not a substitute for stopping it
   yourself when you're done).

Confirm the VM joined the tailnet in the
[Tailscale admin console](https://login.tailscale.com/admin/machines) and
note its `100.x.y.z` address — you'll need it for both the server launch
command and the robot-side client below.

### 3. Start the PolicyServer

SSH in over Tailscale (`tailscale ssh azureuser@brachiomimus-policy-server`)
and launch the container, publishing the gRPC port **only on the Tailscale
interface**:

```bash
TAILSCALE_IP=$(tailscale ip -4)
docker run --gpus all -d \
  --name policy-server \
  --restart unless-stopped \
  -p "${TAILSCALE_IP}:8080:8080" \
  huggingface/lerobot-gpu:latest \
  python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080
```

`--host=0.0.0.0` is the bind address *inside the container*; the `-p`
publish flag is what actually controls reachability, and it's scoped to the
Tailscale IP only — the container's port is never mapped onto the VM's
public IP. `--restart unless-stopped` matters for the stop/start cycle
described in [Cost control](#cost-control): cloud-init already enables the
Docker daemon itself to start on boot, so with this flag a deallocated-then-
restarted VM brings the PolicyServer back up on its own — you don't have to
re-run this command every time you power the VM back on, only the first
time you ever start it on a given VM.

### 4. Join the robot machine to the tailnet

Install [Tailscale for Windows](https://tailscale.com/download/windows) on
the machine wired to Brachiomimus, sign in to the same tailnet, and confirm
`tailscale ping brachiomimus-policy-server` succeeds.

### 5. Run the RobotClient

Needs the async extra on top of the usual LeRobot install:

```bash
pip install "lerobot[async]"
```

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<vm-tailscale-ip>:8080 \
    --robot.type=so101_follower \
    --robot.port=COM4 \
    --robot.id=brachiomimus_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --policy_type=smolvla \
    --pretrained_name_or_path=<your-hf-repo-id> \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
```

- `--server_address` — the VM's Tailscale IP, **not** its public IP.
- `--pretrained_name_or_path` — the Hub repo of the trained SmolVLA
  checkpoint (produced the same way as the ACT policy in
  [training-act.md](training-act.md), with `--policy.type=smolvla`).
- `--actions_per_chunk` / `--chunk_size_threshold` — tune these if motion
  looks stuttery (chunk too small, requesting too often) or laggy/stale
  (chunk too large, executing outdated actions); pass
  `--debug_visualize_queue_size=True` to watch the action queue while
  tuning.

## Cost control

`Standard_NC4as_T4_v3` runs roughly **$0.53–0.59/hr** pay-as-you-go in US
regions (check the
[Azure pricing calculator](https://azure.microsoft.com/pricing/calculator/)
for the current rate before relying on this number). Stop or deallocate the
VM (`az vm deallocate --resource-group brachiomimus-inference --name
brachiomimus-policy-server`) when you're not actively running inference —
billing for compute stops on deallocation, not just VM shutdown from inside
the OS. The 3am auto-shutdown `provision.sh` sets is a safety net against a
forgotten running VM, not a replacement for stopping it yourself right after
a session.

**Deallocate/start is the normal day-to-day cycle, not `provision.sh`.** The
VM's disk (and everything on it — Docker, the pulled image, the Tailscale
registration, the `policy-server` container) persists across
`deallocate`/`start`; cloud-init only runs once, on the VM's very first
boot. So the routine loop is just:

```bash
az vm deallocate --resource-group brachiomimus-inference --name brachiomimus-policy-server   # end of session
az vm start      --resource-group brachiomimus-inference --name brachiomimus-policy-server   # next session
```

No Tailscale key, no NSG setup, no re-running `provision.sh` — the VM
rejoins the tailnet and the PolicyServer container comes back up on its own
(the `--restart unless-stopped` flag above). You only need `provision.sh`
again to create a genuinely *new* VM — the westus2 failover case, or if
you've deleted the VM outright rather than deallocated it.

Leave the standby region's quota unprovisioned until you actually need it —
there's no cost to unused quota, only to a running VM.

## Failing over to the standby region

If the primary region has a capacity or outage problem:

```bash
TAILSCALE_AUTHKEY=tskey-... ./provision.sh --location westus2
```

Same script, same cloud-init, new region. Once the new VM joins the tailnet,
start the PolicyServer on it (step 3 above) and re-point
`robot_client --server_address` at its Tailscale IP.

## Security checklist recap

- [ ] NSG has no inbound rule for port 8080, and no public SSH rule (`az
      network nsg rule list --resource-group brachiomimus-inference --nsg-name <name>`
      should show nothing but Azure's default deny-all).
- [ ] PolicyServer's Docker `-p` publish is bound to the Tailscale IP, not
      `0.0.0.0`, on the host side.
- [ ] Tailscale auth key was passed as an env var only, never committed —
      same rule as `.env` / `WANDB_API_KEY` in this repo.
- [ ] Admin access goes through `tailscale ssh`, not a public-IP SSH session.
