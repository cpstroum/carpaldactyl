# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Experiments controlling a pair of SO-101 robotic arms (Feetech STS3215 servos)
via [LeRobot](https://github.com/huggingface/lerobot): **Brachius Rex** (leader,
human-driven teleoperator) and **Brachiomimus** (follower, does the work).

The repo is organized as a ladder of approaches, each rung trading
hand-authoring for learned generalization — see the table in `README.md` (the
"spectrum of experiments"). Rungs 0/0.5 are custom scripts in this repo; rungs
1/2 are LeRobot's own CLI (`lerobot-teleoperate`, `lerobot-record`,
`lerobot-train`, etc.) driving datasets/policies on the Hub — there's no
training code to read in this repo, only the docs describing the CLI
invocations (`docs/teleoperation.md`, `docs/training-act.md`, `docs/vla.md`).

## Running things

Everything under `demos/` and `tools/` must be run as a module from the repo
root so Python can resolve the `brachiomimus` package import — running by path
(`python demos/wave.py`) puts only that file's directory on `sys.path` and the
import fails:

```bash
python -m demos.wave --port COM4 --reps 3
python -m demos.dance --port COM4 --audio-source loopback
python -m demos.track --port COM4 --show
python -m demos.reach --port COM4 --camera 1 --show
python -m tools.probe_color
python -m tools.diagnostics
```

Every demo supports `--dry-run` (prints computed poses instead of sending
them to the arm) and most support `--show` (OpenCV debug windows) — both work
with no arm/hardware connected, which is the fastest way to sanity-check a
change to motion or vision code before touching real hardware.

There is no build step, lint config, or test suite in this repo — LeRobot
itself is the only "framework" dependency, installed separately (not via
`requirements.txt`, which only lists the extra vision packages for
`reach.py`/`track.py`; see the note at the top of that file for the OpenCV
4.x-vs-5.x split between the two demos).

No CI is configured.

## Architecture

**`brachiomimus/`** is the shared core package that every demo and tool
imports from — it holds nothing that moves the arm by itself, only shared
building blocks:

- `hardware.py` — the six `Motor` definitions (`MOTORS`), `load_calibration()`
  (reads the JSON LeRobot writes at
  `~/.cache/huggingface/lerobot/calibration/robots/so_follower/...json`), and
  two canonical poses: `REST_POSE` (flat — arm sags once torque re-enables,
  not a safe shutdown state) and `READY_POSE` (raised/tucked — the shared
  safe pose several demos ramp to on start/stop).
- `motion.py` — pure dict math over pose dicts (`{joint_name: degrees}`):
  `blend()` (lerp between two poses) and `clamp_step()` (per-tick slew limit,
  the thing that keeps motion smooth/safe — was previously copy-pasted into
  every demo before being extracted here).
- `vision.py` — OpenCV perception primitives: Haar-cascade face detection
  (used by `track.py`'s `--target face`) and HSV colored-blob detection
  (used by `reach.py` and `track.py`'s `--target color`).
- `audio.py` — real-time audio sources for `dance.py` (`mic` via
  `sounddevice`/PortAudio, `loopback` via `soundcard`/WASAPI, `file` playback).
- `analysis.py` — audio DSP: loudness envelope + bass-band beat/onset
  detection + BPM estimate, feeding `dance.py`'s motion.
- `config.py` — env/`.env`-sourced settings (gripper calibration, port,
  sensitivity, intensity). Precedence: CLI flag > real env var > `.env` >
  built-in default. `.env` is gitignored — see "Secrets" below.

**`demos/`** — the runnable behaviors (rungs 0 and 0.5), each a standalone
`argparse` script that imports from `brachiomimus/`:

- `wave.py` — rung 0, open-loop scripted wave. Talks to `FeetechMotorsBus`
  directly, bypassing LeRobot's robot/teleoperator classes — the minimal
  "does calibration work at all" check.
- `dance.py` — rung 0, music-reactive motion driven live by `audio.py` +
  `analysis.py` (see `docs/music.md` for the full behavior description).
- `track.py` — rung 0.5, room-facing webcam, turns to face a target. Two
  `--target` modes: `face` (default, Haar-cascade detection, requires
  OpenCV **4.x** — `cv2.CascadeClassifier` was removed in OpenCV 5) or
  `color` (HSV blob detection via `vision.find_blob`, the same primitive
  `reach.py` uses for its wrist-camera grasp — no OpenCV-version constraint).
- `reach.py` — rung 0.5, wrist-camera colored-blob visual servoing (center →
  advance → grasp), no IK/depth — see the module docstring for the on-arm
  tuning workflow (hover pose, HSV range, pan/tilt inversion). No force
  sensing/grasp confirmation.

**`tools/`** — tuning/calibration helpers, not behaviors: `diagnostics.py`
(the `--monitor`/`--read-gripper` helpers for `dance.py`), `probe_color.py`
(click a live camera feed to read a target's real HSV for `reach.py`).

**`infra/azure/`** — deploy/provisioning tooling, not a behavior or a tuning
helper either: `cloud-init-policy-server.yaml` + `provision.sh` stand up the
Azure GPU VM that serves a trained VLA policy for rung 2 — see
`docs/vla-inference-azure.md`.

**`docs/`** — how-to guides plus the roadmap, one file per rung/topic:
`teleoperation.md` (record datasets), `training-act.md` (train/eval ACT on HF
Jobs), `vla.md` (rung-2 roadmap) + `vla-inference-azure.md` (rung-2 cloud
inference deployment), `music.md` (dance.py deep dive), `learnings.md`
(cross-cutting gotchas — LeRobot API-shape changes, OpenCV version split, ACT
training tuning notes — read this before debugging anything that "used to
work").

## Conventions specific to this repo

- **Poses are `dict[str, float]`** keyed by the six joint names in `MOTORS`
  (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
  `gripper`), values in degrees. Nearly all motion code is dict math over this
  shape (`motion.blend`, `motion.clamp_step`) — new motion logic should follow
  the same shape rather than inventing a new pose representation.
- **LeRobot v0.4.x API shape**: `Motor`/`MotorCalibration`/`MotorNormMode`
  dataclasses from `lerobot.motors`, `FeetechMotorsBus` from
  `lerobot.motors.feetech` — not the older tuple-based / `lerobot.common.*`
  paths. If LeRobot imports break, check `docs/learnings.md` first.
- **Remote/cloud policy inference** (rung 2) goes through LeRobot's built-in
  `lerobot.async_inference.policy_server` / `.robot_client` pair (gRPC,
  action-chunked) rather than any custom server code — see
  `docs/vla-inference-azure.md`. As of `lerobot` 0.4.3 the PolicyServer has
  an unpatched, unauthenticated RCE (CVE-2026-25874, unsafe pickle
  deserialization in its gRPC handlers) — it must never be bound to a
  public/internet-reachable address; the doc's whole networking design
  (Tailscale mesh + deny-all NSG) exists because of this.
- **Calibration files** land under `~/.cache/huggingface/lerobot/calibration/`
  under `so_follower`/`so_leader` (not `so101_*`) — `robots/so_follower/` for
  Brachiomimus, `teleoperators/so_leader/` for Brachius Rex.
- **Secrets and per-arm calibration** live only in the gitignored root `.env`
  (copied from the committed, secret-free `.env.example`): non-secret
  `BRACHIOMIMUS_*` keys read by `brachiomimus/config.py`, plus `WANDB_API_KEY`
  read by cloud training via `hf jobs run --secrets-file .env`. Never put
  secrets in code, commands, or commit them — see
  `docs/training-act.md#authentication-keys-live-in-env`.
- **Windows-first**: ports are `COM*` (vs. `/dev/ttyACM*`/`/dev/ttyUSB*` on
  Linux), PowerShell line continuation is `` ` `` not `\`. Both are used
  throughout the docs — match whichever the surrounding examples use.
