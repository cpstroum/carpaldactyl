# Lessons learned

Cross-cutting things that broke, surprised us, or took a while to figure out —
kept out of the how-to docs so they don't clutter the happy path. Tagged by the
rung of the [spectrum](../README.md#the-spectrum-of-experiments) they came from.

## Rung 0 / 0.5 — low-level control (LeRobot v0.4.x compatibility)

These broke silently when upgrading from older LeRobot versions — relevant if
you're writing or reading the low-level scripts under `demos/` (`wave.py` and
friends) or the shared `brachiomimus/hardware.py`.

**Import path** — `lerobot.common.robot_devices.motors.feetech` no longer
exists. Use:
```python
from lerobot.motors.feetech import FeetechMotorsBus
```

**Motor definitions** — motors are no longer plain `(id, model)` tuples. Use
the `Motor` dataclass:
```python
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

motors = {
    "shoulder_pan": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.DEGREES),
    # ...
}
```

**Calibration** — pass calibration explicitly as a `dict[str, MotorCalibration]`
loaded from the JSON file LeRobot saves at:
```
~/.cache/huggingface/lerobot/calibration/robots/so_follower/brachiomimus_follower.json
```
```python
import json
from lerobot.motors import MotorCalibration

with open(calibration_path) as f:
    data = json.load(f)
calibration = {name: MotorCalibration(**fields) for name, fields in data.items()}
bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)
```
(In this repo, `brachiomimus/hardware.py` already wraps the last two for you —
`MOTORS` and `load_calibration()`.)

**Calibrating:** position the arm at the physical midpoint of every joint
*before* launching `lerobot-calibrate`, not just when prompted. If a joint is
too far off-center, LeRobot crashes with `Magnitude exceeds 2047`.

## Rung 0.5 — OpenCV vision

**OpenCV 5 removed the Haar cascade API.** `track.py`'s face detection uses the
classic `cv2.CascadeClassifier` + `cv2.data.haarcascades`, which OpenCV 5
dropped along with the bundled cascade XML files. Pin OpenCV 4.x
(`opencv-python>=4.8,<5`) to run the face demo. The color-blob path (`reach.py`)
works on 4.x or 5.x.

**Color detection lives or dies on real HSV values.** A target's hue under
*your* camera and lighting is often nowhere near the textbook value, especially
for muted objects under a color cast. Sample actual pixels with
`python -m tools.probe_color` rather than guessing, and prefer a bright,
*saturated* marker in a hue absent from the scene — high saturation also keeps
the low-saturation gripper and pale background out of the mask.

**No true force sensing, but `reach.py` has a load-based heuristic.** Visual
servoing alone has no grasp confirmation. `reach.py`'s GRASP state can end
early by reading the gripper servo's `Present_Load` and comparing it to
`--load-threshold` (disabled by default at 0): closing on a rigid object
(a can, a block) stalls the servo against it well before the commanded
`--gripper-closed` angle, so load climbs past whatever's normal in open
air. This is still a heuristic, not confirmation — a soft/thin target may
never build enough load (falls back to the `GRASP_HOLD_S` timer, same as
before this existed), and friction could false-trigger it early. Tune the
threshold empirically with `--show` (watch the live `load=` overlay while
squeezing the real target) rather than guessing a number. Watch the lift;
don't leave it running unattended regardless.

## Rung 1 — training ACT on HF Jobs

Notes from actually training the block-movement policy on
[`cpstroum/so101-brachiomimus-50ep`](https://huggingface.co/datasets/cpstroum/so101-brachiomimus-50ep)
(50 episodes, "Pick up the yellow block and put it in the blue box").

**5k steps is not enough.** The first run (`--steps=5000`) produced a policy that
was "super jittery" and failed eval. Bumping to `--steps=30000` (6×) gave a
usable one. Batch size stayed at 16 throughout.

**Watch the wall clock, not just the step count.** A `t4-small` Job **timed out**
before finishing 30k steps — loading the dataset and setting up the environment
eats real minutes before training even starts. Fixes that worked: a bigger
flavor (`a10g-small`) and a longer `--timeout` (6h), plus `--save_freq=5000` so a
timeout still leaves recoverable checkpoints.

**Always enable W&B.** `--wandb.enable=true --wandb.project=so101-brachiomimus` on
every run — the loss curve is the only way to judge a slow cloud run without
burning an eval, and comparing curves across runs is how the step-count decision
got made.

**Publish the policy or you can't eval it.** `--policy.push_to_hub=true` pushes
the trained model to `--policy.repo_id` on the Hub; eval then pulls it with
`--policy.path=<that repo id>`. The early runs that skipped this left the policy
stuck in the Job.

**Auth without keys in the command.** HF was already authenticated
(`huggingface-cli login`), so no token appears in any command — HF Jobs pulls it
from the `HF_TOKEN` secret. The W&B key lives in the gitignored `.env` and is
passed with `--secrets-file .env` (encrypted server-side), so it never reaches
shell history, notes, or a commit. Early runs pasted the key inline
(`--secrets WANDB_API_KEY=wandb_...`) — the `.env` + `--secrets-file` approach
replaced that.

**Eval via record, not rollout.** Rollouts are captured with `lerobot-record
--policy.path=...` into an `eval_*` dataset, optionally with the leader still
attached for resets between episodes. Keep eval-dataset and policy versions in
lockstep (`-v2` ↔ `-v2`).

## Repository structure

The behaviors and tools are launched as modules from the repo root
(`python -m demos.wave`, `python -m tools.probe_color`) so Python can resolve
the `brachiomimus` package they import from. Running a script by path
(`python demos/wave.py`) puts only the script's own directory on `sys.path`, so
the package import fails — use `-m`.
