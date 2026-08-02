"""
reach.py — visually servo Brachiomimus's wrist camera onto a colored object,
then reach in and grasp it.

Built for the wrist-mounted (eye-in-hand) camera, not the room-facing one
track.py uses. Detects a colored blob (defaults to a lavender-ish purple; use
--hue-min/--hue-max to retarget) and closes the gripper on it. No depth
estimation or inverse kinematics involved - visual servoing instead works
entirely off what the wrist camera sees getting bigger and more centered as
the arm approaches:

  1. SEARCH — no target in view: slowly sweep shoulder_pan until the blob
     appears.
  2. TRACK  — target in view but not close enough yet: center it with
     shoulder_pan (left/right) and wrist_flex (up/down), and extend
     elbow_flex a little each tick while centered but not yet close (blob
     area is the proxy for distance — bigger blob = closer).
  3. GRASP  — blob fills enough of the frame and is centered: close the
     gripper and hold, until the servo's own reported load says it's met
     resistance (a rigid object like a can) or GRASP_HOLD_S elapses,
     whichever comes first — see --load-threshold below.
  4. LIFT   — raise back to the ready pose so you can see whether the grasp
     actually took.

IMPORTANT — this needs on-arm tuning before it'll do anything sensible:
  - REACH_READY_POSE below is a guess at a pose that points the wrist
    camera down/forward at a workspace. Jog the arm by hand to a good hover
    position over wherever the lavender sits and replace these numbers.
  - The dx/dy -> joint mapping (which joint moves the image left/right vs
    up/down) depends on how the wrist camera is physically mounted and can
    come out inverted or swapped; use --invert-pan/--invert-tilt, same idea
    as track.py.
  - Detection is color-based, so it needs a target that's a distinct,
    SATURATED color against the background. A muted natural object (dried
    lavender) under a color cast is a poor target - the reliable trick is to
    tie a brightly colored marker (a scrap of saturated yarn) at the exact
    spot you want grabbed, in a hue absent from the scene, and detect that.
    High saturation also keeps the low-saturation gripper and any pale
    background out of the mask. Find the marker's real HSV by clicking it in
    probe_color.py, then set --hue-min/--hue-max/--sat-min/--val-min to
    bracket it. Run reach.py with --show to see the mask and confirm only the
    marker lights up. (--white instead targets a bright, near-colorless
    object like white string, but only against a dark backdrop and at the
    risk of latching onto a shiny gripper - a colored marker is safer.)
  - For a thin marker, lower --min-area so the grasp still triggers before
    the target fills 18% of the frame.
  - GRASP now reads the gripper servo's Present_Load (how hard it's pushing
    against its commanded position) instead of always waiting a fixed
    GRASP_HOLD_S. Closing on a rigid object (a can, a block) makes the
    gripper stall against it well before reaching the fully-closed angle,
    so load climbs - crossing --load-threshold ends the hold early. This
    is still not real grasp confirmation (a soft or thin target may never
    build enough load, or friction could false-trigger it early), so watch
    the lift and judge for yourself. Don't leave it unattended. Tune
    --load-threshold by running with --show, squeezing the gripper closed
    on your actual target, and reading the live "load=" number in the
    overlay - pick a threshold clearly above the open-air baseline but
    below wherever the servo's own overcurrent/stall limit kicks in. If
    your installed LeRobot's Feetech bus doesn't expose "Present_Load" the
    same way, this prints one warning and falls back to the old
    GRASP_HOLD_S-only behavior automatically.

Usage:
    python -m demos.reach --port /dev/ttyACM0 --camera 1 --show
    python -m demos.reach --dry-run --show          # tune detection, no arm

Requires opencv-python and numpy, same extra dependency as track.py.
"""

import argparse
import math
import time

import cv2

from lerobot.motors.feetech import FeetechMotorsBus

from brachiomimus import config
from brachiomimus.hardware import CALIBRATION_PATH, MOTORS, load_calibration
from brachiomimus.motion import clamp_step
from brachiomimus.vision import find_blob

# Arm pose that points the wrist camera down at the target with the gripper
# open and clear, captured with --read-pose (marker centered on the crosshair)
# on the arm this repo is tuned for. Re-capture with --read-pose if your
# workspace or mounting differs.
REACH_READY_POSE = {
    "shoulder_pan": 32.0,
    "shoulder_lift": -56.4,
    "elbow_flex": 45.3,
    "wrist_flex": 33.6,
    "wrist_roll": -55.5,
    "gripper": -14.3,  # overwritten each tick with the real open/closed angle
}

# Pose the arm ramps to on shutdown before torque is released. It should be
# LOW and self-supporting - arm folded down and/or resting on the table - so
# that when torque cuts off the arm has nowhere to fall and doesn't flop. A
# raised pose (like REACH_READY_POSE) drops as soon as it goes limp. Captured
# with --read-pose by letting the arm settle into a stable resting position by
# hand. Re-capture if your mounting or surface differs.
PARK_POSE = {
    "shoulder_pan": 30.5,
    "shoulder_lift": -108.9,
    "elbow_flex": 99.4,
    "wrist_flex": 35.3,
    "wrist_roll": -55.4,
    "gripper": -38.0,
}

PAN_JOINT = "shoulder_pan"
TILT_JOINT = "wrist_flex"
ADVANCE_JOINT = "elbow_flex"
GRIPPER_JOINT = "gripper"

MAX_STEP_DEG = 4.0       # slower than track.py's 6 - this is reaching, not just looking
ADVANCE_STEP_DEG = 2.5   # how far elbow_flex extends per tick while approaching
CENTER_TOLERANCE = 0.15  # blob center must be within this fraction of frame center...
CLOSE_AREA_FRACTION = 0.18  # ...and blob must cover this fraction of the frame...
                             # ...before GRASP triggers

LOST_TIMEOUT_S = 1.5
SEARCH_PERIOD_S = 8.0
SEARCH_SWEEP_LIMIT_DEG = 40.0

LOAD_REGISTER = "Present_Load"  # Feetech STS3215 control-table field; see --load-threshold

GRASP_HOLD_S = 0.6


def read_pose(port: str, camera: int | None = None) -> None:
    """Release ALL joints and print their live angles, so you can hand-jog the
    arm to a good hover pose over the target (wrist camera aimed at it, gripper
    open and clear) and read off the six numbers for REACH_READY_POSE.

    If a camera index is given, a live preview of the wrist camera opens with a
    crosshair at frame center - jog the arm until the marker sits on the
    crosshair, then press q in the window (or Ctrl+C) to capture. Without a
    camera it just prints angles; capture with Ctrl+C.

    SAFETY: this turns torque OFF, so the arm goes limp and will sag under
    gravity. Support it by hand before you start, and lower it gently after.
    """
    calibration = load_calibration(CALIBRATION_PATH)
    bus = FeetechMotorsBus(port=port, motors=MOTORS, calibration=calibration)
    bus.connect()
    cap = None
    if camera is not None:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            print(f"Warning: could not open camera {camera}; continuing with no preview.")
            cap = None
    pos = {}
    try:
        bus.sync_write("Torque_Enable", 0)
        print("Torque OFF - hold the arm, it's limp now. Move it to a good hover")
        if cap is not None:
            print("pose with the marker centered on the crosshair, then press q in the")
            print("preview window (or Ctrl+C) to capture. Live angles:")
        else:
            print("pose over the target, then Ctrl+C to capture it. Live angles:")
        while True:
            pos = bus.sync_read("Present_Position")
            print("  " + "  ".join(f"{k}={pos[k]:6.1f}" for k in MOTORS), end="\r", flush=True)
            if cap is not None:
                ok, frame = cap.read()
                if ok:
                    h, w = frame.shape[:2]
                    cv2.drawMarker(frame, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                    cv2.imshow("read_pose (q to capture)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        if pos:
            print("\n\nPaste this into REACH_READY_POSE in reach.py:")
            print("REACH_READY_POSE = {")
            for k in MOTORS:
                print(f'    "{k}": {pos[k]:.1f},')
            print("}")
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()
        bus.disconnect()
        print("\nDone.")


def run(
    port: str,
    camera: int,
    dry_run: bool,
    show: bool,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    val_min: int,
    min_area_fraction: float,
    invert_pan: bool,
    invert_tilt: bool,
    gripper_closed: float,
    gripper_open_deg: float,
    repeat: bool,
    white: bool = False,
    sat_max: int = 60,
    track_only: bool = False,
    max_step: float = MAX_STEP_DEG,
    load_threshold: float = 0.0,
) -> None:
    # Build the HSV mask bounds once. The default path brackets a saturated
    # hue (a colored marker or blob); --white instead looks for bright,
    # near-colorless pixels (a white string) - low saturation, high value,
    # any hue. White mode only separates cleanly against a dark backdrop and
    # can be fooled by a light/shiny gripper, so a saturated colored marker
    # is usually the more reliable target.
    if white:
        lower = (0, 0, val_min)
        upper = (179, sat_max, 255)
    else:
        lower = (hue_min, sat_min, val_min)
        upper = (hue_max, 255, 255)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera}")

    bus = None
    if not dry_run:
        calibration = load_calibration(CALIBRATION_PATH)
        bus = FeetechMotorsBus(port=port, motors=MOTORS, calibration=calibration)
        bus.connect()
        bus.sync_write("Torque_Enable", 1)

    current_pose = {**REACH_READY_POSE, GRIPPER_JOINT: gripper_open_deg}
    state = "search"
    last_seen = time.monotonic()
    search_start = time.monotonic()
    grasp_start = None
    load = None
    # Read Present_Load during GRASP whenever hardware is attached, even with
    # --load-threshold left at its disabled default (0) - so the live value
    # shows up in --show for tuning before you pick a real threshold.
    load_capable = bus is not None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed, stopping.")
                break

            fh, fw = frame.shape[:2]
            blob, mask = find_blob(frame, lower, upper)
            now = time.monotonic()

            if state == "grasp" and load_capable:
                try:
                    load = bus.sync_read(LOAD_REGISTER)[GRIPPER_JOINT]
                except Exception as e:
                    print(f"\nWarning: couldn't read {LOAD_REGISTER} ({e}); "
                          f"falling back to GRASP_HOLD_S-only grasp timing.")
                    load_capable = False
                    load = None

            area_fraction = 0.0
            if blob is not None:
                last_seen = now
                x, y, w, h, area = blob
                area_fraction = area / (fw * fh)
                if show:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if state == "search":
                    state = "track"
            elif state == "track" and now - last_seen > LOST_TIMEOUT_S:
                print("Lost the target, searching again…")
                state = "search"
                search_start = now

            target = dict(current_pose)

            if state == "search":
                offset = SEARCH_SWEEP_LIMIT_DEG * math.sin(2 * math.pi * (now - search_start) / SEARCH_PERIOD_S)
                target[PAN_JOINT] = REACH_READY_POSE[PAN_JOINT] + offset
                target[TILT_JOINT] = REACH_READY_POSE[TILT_JOINT]
                target[ADVANCE_JOINT] = REACH_READY_POSE[ADVANCE_JOINT]
                target[GRIPPER_JOINT] = gripper_open_deg

            elif state == "track":
                x, y, w, h, area = blob
                cx, cy = x + w / 2, y + h / 2
                dx = (cx - fw / 2) / (fw / 2)
                dy = (cy - fh / 2) / (fh / 2)
                if invert_pan:
                    dx = -dx
                if invert_tilt:
                    dy = -dy

                target[PAN_JOINT] = current_pose[PAN_JOINT] - dx * MAX_STEP_DEG * 2
                target[TILT_JOINT] = current_pose[TILT_JOINT] - dy * MAX_STEP_DEG * 2
                target[GRIPPER_JOINT] = gripper_open_deg

                # --track-only stops here: the arm re-centers the marker with
                # pan/tilt but never advances or grasps, so you can safely
                # confirm the centering directions before it reaches in.
                centered = abs(dx) < CENTER_TOLERANCE and abs(dy) < CENTER_TOLERANCE
                if not track_only:
                    if area_fraction >= min_area_fraction and centered:
                        state = "grasp"
                        grasp_start = now
                    elif centered:
                        target[ADVANCE_JOINT] = current_pose[ADVANCE_JOINT] + ADVANCE_STEP_DEG

            elif state == "grasp":
                target[GRIPPER_JOINT] = gripper_closed
                loaded = load_threshold > 0 and load is not None and abs(load) >= load_threshold
                timed_out = grasp_start and now - grasp_start > GRASP_HOLD_S
                if loaded or timed_out:
                    state = "lift"

            elif state == "lift":
                target = {**REACH_READY_POSE, GRIPPER_JOINT: gripper_closed}
                if all(abs(current_pose[k] - target[k]) < 1.0 for k in target):
                    if repeat:
                        print("Lifted. Searching for next target…")
                        state = "search"
                        search_start = now
                    else:
                        print("Lifted. Done.")
                        break

            current_pose = clamp_step(current_pose, target, max_step)

            if dry_run:
                print(state, {k: round(v, 1) for k, v in current_pose.items()})
            else:
                bus.sync_write("Goal_Position", current_pose)

            if show:
                load_label = f"  load={load}" if state == "grasp" and load is not None else ""
                label = (f"{state}{' (track-only)' if track_only else ''}  "
                         f"area={area_fraction * 100:4.1f}%  grasp>={min_area_fraction * 100:.0f}%{load_label}")
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("reach.py", frame)
                cv2.imshow("reach.py mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.03)
    except KeyboardInterrupt:
        print("Stopping…")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        if bus is not None:
            # Ramp down to the low, self-supporting park pose before releasing
            # torque, so the arm settles instead of flopping when it goes limp.
            park_target = {**PARK_POSE, GRIPPER_JOINT: gripper_open_deg}
            pose = current_pose
            for _ in range(40):
                pose = clamp_step(pose, park_target, MAX_STEP_DEG)
                bus.sync_write("Goal_Position", pose)
                time.sleep(0.05)
            bus.sync_write("Torque_Enable", 0)
            bus.disconnect()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visually servo Brachiomimus onto a colored object and grasp it")
    parser.add_argument("--port", default=config.PORT, help=f"Serial port the arm is connected to (default: {config.PORT})")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV index of the WRIST camera (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Print computed poses instead of sending them to the arm")
    parser.add_argument("--show", action="store_true", help="Open debug windows: camera view with the detected blob boxed, and the color mask")
    parser.add_argument("--hue-min", type=int, default=125, help="HSV hue lower bound, 0-179 (default: 125, roughly lavender)")
    parser.add_argument("--hue-max", type=int, default=155, help="HSV hue upper bound, 0-179 (default: 155)")
    parser.add_argument("--sat-min", type=int, default=40, help="HSV saturation lower bound, 0-255 (default: 40)")
    parser.add_argument("--val-min", type=int, default=60, help="HSV value/brightness lower bound, 0-255 (default: 60)")
    parser.add_argument("--min-area", type=float, default=CLOSE_AREA_FRACTION, help=f"Fraction of frame the blob must fill before grasping (default: {CLOSE_AREA_FRACTION})")
    parser.add_argument("--invert-pan", action="store_true", help="Flip left/right centering direction (wrist camera mount dependent)")
    parser.add_argument("--invert-tilt", action="store_true", help="Flip up/down centering direction (wrist camera mount dependent)")
    parser.add_argument("--gripper-closed", type=float, default=config.GRIPPER_CLOSED_DEG, help="Gripper angle that is FULLY CLOSED on your arm - see --read-gripper in dance.py")
    parser.add_argument("--gripper-open", type=float, default=config.GRIPPER_OPEN_DEG, help="Gripper angle that is fully open")
    parser.add_argument("--repeat", action="store_true", help="After lifting, go back to SEARCH instead of exiting")
    parser.add_argument("--white", action="store_true", help="Detect a bright, near-colorless target (e.g. white string) instead of a hue: ignores --hue-*, matches saturation up to --sat-max and value at/above --val-min. Needs a dark backdrop and can be fooled by a shiny gripper - a saturated colored marker is usually more reliable.")
    parser.add_argument("--sat-max", type=int, default=60, help="In --white mode, the maximum saturation that still counts as 'white/pale' (default: 60)")
    parser.add_argument("--read-pose", action="store_true", help="Release torque and print live joint angles so you can hand-jog the arm to a good hover pose and capture it for REACH_READY_POSE, then exit. Pass --camera too to see a live wrist-camera preview with a center crosshair while you pose. Needs --port.")
    parser.add_argument("--track-only", action="store_true", help="Center the marker with pan/tilt but never advance or grasp. Use for the first live run to confirm the centering directions (add --invert-pan/--invert-tilt if it corrects the wrong way) before letting the arm reach in.")
    parser.add_argument("--max-step", type=float, default=MAX_STEP_DEG, help=f"Per-tick slew limit in degrees (default: {MAX_STEP_DEG}). Lower it (e.g. 2) for a slower, more cautious first live run.")
    parser.add_argument("--load-threshold", type=float, default=0.0, help=f"Gripper {LOAD_REGISTER} magnitude that ends the grasp hold early, for a rigid target (a can, a block) that stalls the servo before it reaches --gripper-closed. Disabled by default (0) - GRASP_HOLD_S always applies as a fallback/ceiling regardless. Tune by running with --show and a real target: watch the 'load=' number in the overlay while it grasps, then pick a threshold clearly above the open-air baseline.")
    args = parser.parse_args()
    if args.read_pose:
        read_pose(args.port, args.camera)
    else:
        run(
            args.port, args.camera, args.dry_run, args.show,
            args.hue_min, args.hue_max, args.sat_min, args.val_min,
            args.min_area, args.invert_pan, args.invert_tilt,
            args.gripper_closed, args.gripper_open, args.repeat,
            args.white, args.sat_max, args.track_only, args.max_step,
            args.load_threshold,
        )
