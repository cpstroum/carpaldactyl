# TODO

## Coarse-to-fine track → reach pipeline

Right now `track.py` and `reach.py` are two independent, separately-launched
demos:

- `track.py` — room-facing camera, pan/tilt only (no advance/grasp).
- `reach.py` — wrist-mounted (eye-in-hand) camera, full
  search/track/advance/grasp state machine. Its SEARCH state currently does
  a *blind* sinusoidal `shoulder_pan` sweep with the wrist camera alone,
  hoping the target drifts into view — it doesn't look around the room.

Idea: use the room-facing camera to coarsely point `shoulder_pan`/
`wrist_flex` at the target (replacing the blind sweep), then hand off to
`reach.py`'s existing TRACK → GRASP logic once the target is roughly framed,
for fine centering/approach off the wrist camera.

Open questions before building:
- Needs both cameras readable at once — two `--camera` indices, not one.
- Need a handoff rule: how "roughly pointed at it" (room-cam detection
  centered within some tolerance?) triggers switching control to the wrist
  camera's TRACK state.
- New script, or a `--coarse-camera` flag bolted onto `reach.py`?

Not started — revisit after plain `reach.py` (wrist-camera-only) grasping is
working end-to-end on the can.
