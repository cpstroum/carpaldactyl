"""
track.py — make Brachiomimus turn and "look" toward whoever/whatever is in
view.

Points a webcam at the room (it doesn't need to be mounted on the arm — any
camera pointed at the space works) and tracks a target each frame.
Brachiomimus turns to face it: shoulder_pan tracks left/right, wrist_flex
tracks up/down. No ML training or eye-in-hand calibration needed.

Two target modes, selected with --target:
  - face  (default) — OpenCV's built-in Haar cascade face detector.
  - color            — an HSV color blob, same detection reach.py uses for
    its wrist-camera grasp (brachiomimus.vision.find_blob). The
    --hue-min/--hue-max/--sat-min/--val-min defaults bracket a red/orange
    can (probed with tools.probe_color: mean H=6 S=96 V=231, spread
    H[5-9] S[90-102] V[224-235]) - re-probe and override them for a
    different target the same way reach.py's docstring describes: use
    `python -m tools.probe_color` to read the real HSV, and --show to see
    the mask while dialing it in.

When the target first appears, the gripper gives a quick friendly pulse.
When nothing is in view, the arm eases back to a centered "watching" pose
instead of snapping.

Usage:
    python -m demos.track --port /dev/ttyACM0
    python -m demos.track --port COM4 --show                    # debug window with target box
    python -m demos.track --dry-run --show                       # no arm, just watch detection
    python -m demos.track --dry-run --show --target color        # track a colored marker instead of a face

Requires opencv-python (`pip install opencv-python`), not otherwise a
dependency of this repo. --target face needs OpenCV 4.x (see the caveat
below); --target color has no such constraint.
"""

import argparse
import time

import cv2

from lerobot.motors.feetech import FeetechMotorsBus

from brachiomimus import config
from brachiomimus.hardware import CALIBRATION_PATH, MOTORS, READY_POSE, load_calibration
from brachiomimus.motion import clamp_step
from brachiomimus.vision import face_detector, find_blob, largest_face

# Centered version of the raised "ready" pose - arm up and alert, facing
# forward, rather than angled out for a wave.
TRACK_READY_POSE = {**READY_POSE, "shoulder_pan": 0.0}

PAN_JOINT = "shoulder_pan"
TILT_JOINT = "wrist_flex"
GRIPPER_JOINT = "gripper"

MAX_STEP_DEG = 6.0  # per-tick slew limit, matches dance.py's feel
LOST_TIMEOUT_S = 1.0  # stop chasing a face this long after losing it

GREET_PULSE_S = 0.4  # how long the gripper stays open on first sighting
GREET_OPEN_DEG = 20.0


def run(
    port: str,
    camera: int,
    dry_run: bool,
    show: bool,
    pan_range: float,
    tilt_range: float,
    invert_pan: bool,
    invert_tilt: bool,
    target: str,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    val_min: int,
) -> None:
    # Only touch cv2.CascadeClassifier in face mode - it's the removed-in-
    # OpenCV-5 API, so color mode never constructs it and stays OpenCV-5-safe.
    detector = face_detector() if target == "face" else None
    lower, upper = (hue_min, sat_min, val_min), (hue_max, 255, 255)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera}")

    bus = None
    if not dry_run:
        calibration = load_calibration(CALIBRATION_PATH)
        bus = FeetechMotorsBus(port=port, motors=MOTORS, calibration=calibration)
        bus.connect()
        bus.sync_write("Torque_Enable", 1)

    current_pose = dict(TRACK_READY_POSE)
    last_seen = 0.0
    greet_until = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed, stopping.")
                break

            mask = None
            if target == "face":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                box = largest_face(gray, detector)
                fh, fw = gray.shape
            else:
                blob, mask = find_blob(frame, lower, upper)
                box = blob[:4] if blob is not None else None
                fh, fw = frame.shape[:2]

            goal_pose = dict(TRACK_READY_POSE)
            now = time.monotonic()

            if box is not None:
                if now - last_seen > LOST_TIMEOUT_S:
                    greet_until = now + GREET_PULSE_S
                last_seen = now

                x, y, w, h = box
                cx, cy = x + w / 2, y + h / 2
                dx = (cx - fw / 2) / (fw / 2)  # -1 (left) .. 1 (right)
                dy = (cy - fh / 2) / (fh / 2)  # -1 (up) .. 1 (down)
                if invert_pan:
                    dx = -dx
                if invert_tilt:
                    dy = -dy

                goal_pose[PAN_JOINT] += pan_range * dx
                goal_pose[TILT_JOINT] += tilt_range * dy

                if show:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            goal_pose[GRIPPER_JOINT] = GREET_OPEN_DEG if now < greet_until else 0.0

            current_pose = clamp_step(current_pose, goal_pose, MAX_STEP_DEG)

            if dry_run:
                print({k: round(v, 1) for k, v in current_pose.items()})
            else:
                bus.sync_write("Goal_Position", current_pose)

            if show:
                cv2.imshow("track.py", frame)
                if mask is not None:
                    cv2.imshow("track.py mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Stopping, returning to rest…")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        if bus is not None:
            pose = current_pose
            for _ in range(20):
                pose = clamp_step(pose, TRACK_READY_POSE, MAX_STEP_DEG)
                bus.sync_write("Goal_Position", pose)
                time.sleep(0.05)
            bus.sync_write("Torque_Enable", 0)
            bus.disconnect()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Brachiomimus track a face or colored object with a webcam")
    parser.add_argument(
        "--port", default=config.PORT,
        help=f"Serial port the arm is connected to (default: {config.PORT})"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index to read from (default: 0)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print computed poses instead of sending them to the arm"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open a debug window showing the camera feed with the detected target boxed (plus the color mask in --target color mode)"
    )
    parser.add_argument(
        "--pan-range", type=float, default=45.0,
        help="Max shoulder_pan degrees off-center when the target is at the frame edge (default: 45)"
    )
    parser.add_argument(
        "--tilt-range", type=float, default=20.0,
        help="Max wrist_flex degrees off-center when the target is at the frame edge (default: 20)"
    )
    parser.add_argument(
        "--invert-pan", action="store_true",
        help="Flip left/right tracking direction (camera orientation dependent)"
    )
    parser.add_argument(
        "--invert-tilt", action="store_true",
        help="Flip up/down tracking direction (camera orientation dependent)"
    )
    parser.add_argument(
        "--target", choices=["face", "color"], default="face",
        help="What to track: a face (Haar cascade, needs OpenCV 4.x) or an HSV color blob "
             "(needs no particular OpenCV version - see --hue-min etc.) (default: face)"
    )
    parser.add_argument(
        "--hue-min", type=int, default=3,
        help="--target color: HSV hue lower bound, 0-179 (default: 3, bracketing a red/orange can - "
             "probed with tools.probe_color at mean H=6 S=96 V=231, spread H[5-9] S[90-102] V[224-235])"
    )
    parser.add_argument(
        "--hue-max", type=int, default=11,
        help="--target color: HSV hue upper bound, 0-179 (default: 11)"
    )
    parser.add_argument(
        "--sat-min", type=int, default=80,
        help="--target color: HSV saturation lower bound, 0-255 (default: 80)"
    )
    parser.add_argument(
        "--val-min", type=int, default=210,
        help="--target color: HSV value/brightness lower bound, 0-255 (default: 210)"
    )
    args = parser.parse_args()
    run(
        args.port, args.camera, args.dry_run, args.show,
        args.pan_range, args.tilt_range, args.invert_pan, args.invert_tilt,
        args.target, args.hue_min, args.hue_max, args.sat_min, args.val_min,
    )
