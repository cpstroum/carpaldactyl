# brachiomimus

Experiments with a pair of SO-101 robotic arms using LeRobot:

- **Brachius Rex** — the leader arm (teleoperator, human-driven)
- **Brachiomimus** — the follower arm (does the actual work)

## The spectrum of experiments

This repo isn't one project — it's a ladder of approaches to making the arm
act, each rung trading **hand-authoring** for **learned generalization**. The
code and docs are organized around that ladder:

| Rung | Approach | Perception | Control | Generalizes to | Where |
|------|----------|------------|---------|----------------|-------|
| **0 — open-loop scripted** | hand-authored motion, no feedback | none | you author every pose | nothing — it does exactly what you wrote | `demos/wave.py`, `demos/dance.py` + [music](docs/music.md) |
| **0.5 — classical closed-loop** | perception in the loop, hand-authored control law | OpenCV (faces, color blobs) | visual servoing | new object *positions* in view | `demos/track.py`, `demos/reach.py` |
| **1 — imitation (ACT)** | learn one task from your demos | learned | learned | new positions *within* the trained task | [teleoperation](docs/teleoperation.md) → [training](docs/training-act.md) |
| **2 — VLA** *(next)* | language-conditioned, pretrained | learned + language | learned | new *instructions* and tasks | [roadmap](docs/vla.md) |

The `wave` → `dance` → `track` → `reach` → ACT → VLA progression is the story:
you start by scripting every move, then close a perception loop, then hand the
whole sensorimotor mapping to a learned policy, then to one that understands
language. See [docs/vla.md](docs/vla.md) for where the learned work is headed.

## Repository layout

```
brachiomimus/     shared core (importable package)
  hardware.py       motor defs, canonical poses, calibration loading
  motion.py         pose interpolation / slew-limit helpers
  vision.py         OpenCV perception (face + colored-blob detection)
  audio.py          real-time audio sources for the music demo
  analysis.py       audio DSP (loudness, beat detection, BPM)
  config.py         env/.env-sourced user settings
demos/            runnable behaviors — rungs 0 and 0.5
  wave.py  dance.py  track.py  reach.py
tools/            tuning / calibration helpers (not behaviors)
  diagnostics.py  probe_color.py
infra/            deploy/provisioning tooling (not behaviors either)
  azure/            cloud-init + az CLI script for VLA cloud inference (see docs/vla-inference-azure.md)
docs/             the how-to and the roadmap
```

Behaviors and tools import from the `brachiomimus` package and are launched as
modules from the repo root, e.g. `python -m demos.wave`. (Running them lets
Python find the package; `python demos/wave.py` would not.)

Ports/IDs in the commands below are examples — substitute your own (`COM3`
on Windows, `/dev/ttyACM0` on Linux, etc).

**Settings & secrets:** copy `.env.example` to `.env` (gitignored) and fill in
your arm's calibration; `.env` is also where the W&B key for cloud training
lives. See [docs/training-act.md](docs/training-act.md#authentication-keys-live-in-env).

## Finding your serial port

**Windows (PowerShell):**
```powershell
[System.IO.Ports.SerialPort]::GetPortNames()
```
Plug in the arm, run it again, and the new entry is your port (e.g. `COM3`).

**Linux:**
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```
Feetech-based arms typically appear as `/dev/ttyACM0`. Run `dmesg | tail -20` if unsure.

## Calibration

After running `lerobot-calibrate`, files are saved here:

| Arm | Name | Port | Calibration file |
|-----|------|------|------------------|
| Follower | Brachiomimus | `COM4` | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/brachiomimus_follower.json` |
| Leader | Brachius Rex | `COM7` | `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/brachio_rex_leader.json` |

(LeRobot writes under `so_follower` / `so_leader`, not `so101_*`. Ports are the
Windows enumeration on this setup — `COM4` follower, `COM7` leader; on Linux
they'd be `/dev/ttyACM*`.)

Calibrate the follower:
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=COM4 --robot.id=brachiomimus_follower
```

Calibrate the leader:
```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM7 --teleop.id=brachio_rex_leader
```

**Tip:** When calibrating, position the arm at the physical midpoint of every
joint *before* launching the script — not just when prompted. If a joint is
too far off-center, LeRobot will crash with a `Magnitude exceeds 2047` error.

## Rung 0 — Wave: direct scripted control

Drives Brachiomimus straight through a hand-rolled `FeetechMotorsBus`,
bypassing LeRobot's robot/teleoperator classes entirely. No leader arm or
camera — just the follower doing a scripted wave. Good first check that
calibration took and the arm responds.

```bash
python -m demos.wave --port /dev/ttyACM0 --reps 3   # Linux
python -m demos.wave --port COM4 --reps 3           # Windows
```

Music-reactive dancing (also rung 0) lives in [docs/music.md](docs/music.md).

## Rung 0.5 — Face tracking: Brachiomimus watches the room

Points a plain webcam (not eye-in-hand — anywhere in the room works) at the
space and turns the arm to face whoever it sees, using OpenCV's built-in
Haar cascade face detector. No ML training or camera calibration involved.

> **OpenCV 5 caveat:** this uses the classic `cv2.CascadeClassifier` Haar
> API, which OpenCV 5 removed (along with the bundled cascade files). Run
> it on OpenCV **4.x** — `pip install "opencv-python>=4.8,<5"`. `reach`
> below has no such constraint.

```bash
python -m demos.track --port /dev/ttyACM0
python -m demos.track --port COM4 --show      # debug window with the face boxed
python -m demos.track --dry-run --show        # try it with no arm connected
```

If the arm pans or tilts the wrong way for your camera's orientation, add
`--invert-pan` / `--invert-tilt`.

## Rung 0.5 — Reach: grasp a colored object with the wrist camera

Uses a **wrist-mounted** camera (not the room-facing one from the tracking
demo above) to visually servo onto a colored blob — defaults to a
lavender-ish purple — and grasp it. No inverse kinematics or depth
estimation: it centers the blob with `shoulder_pan`/`wrist_flex` and
extends `elbow_flex` a bit each tick, using the blob getting bigger as the
"getting closer" signal, then closes the gripper once it fills enough of
the frame.

Install the vision dependencies first (on top of your working LeRobot
environment): `pip install -r requirements.txt`.

```bash
python -m demos.reach --port /dev/ttyACM0 --camera 1 --show
python -m demos.reach --dry-run --show      # tune detection with no arm connected
```

**This needs on-arm tuning before it'll do anything useful** — see the
docstring in `demos/reach.py` for what to jog in by hand (the hover pose) and
what to dial in with `--show` (the HSV color range, and
`--invert-pan`/`--invert-tilt` if centering moves the wrong way). There's
no force sensing, so it doesn't verify the grasp actually took — watch the
lift and judge for yourself, and don't leave it unattended.

**Pick a good target.** Detection is color-based, so it wants a distinct,
*saturated* color. Muted natural objects (dried lavender) under a color
cast detect poorly. The reliable trick: tie a scrap of brightly colored
yarn at the spot you want grabbed, in a hue that's absent from the rest of
the scene, and detect that — high saturation also keeps the gripper and any
pale background out of the mask. Use `python -m tools.probe_color` to read the
marker's real HSV (click it in the feed), then set `--hue-min/--hue-max/
--sat-min/--val-min` to bracket it. For a thin marker, lower `--min-area` so
the grasp still triggers. (There's also a `--white` mode for a white-string
target, but it needs a dark backdrop and can latch onto a shiny gripper — a
colored marker is safer.)

## Rungs 1 & 2 — learned policies

- **[docs/teleoperation.md](docs/teleoperation.md)** — drive Brachiomimus with
  Brachius Rex and record a demonstration dataset with a webcam
- **[docs/training-act.md](docs/training-act.md)** — train an ACT policy on that
  dataset and run it on the robot (rung 1)
- **[docs/vla.md](docs/vla.md)** — the roadmap toward language-conditioned,
  generalizing policies (rung 2)

## Lessons learned

Hard-won, cross-cutting notes (LeRobot version gotchas, etc.) live in
**[docs/learnings.md](docs/learnings.md)**, tagged by rung.
