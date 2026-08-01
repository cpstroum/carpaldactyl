# Rung 2 — VLA: language-conditioned, generalizing policies

> Status: **roadmap / not yet implemented.** This is the direction the project
> is heading after ACT (rung 1), and the reason the repo is organized as a
> [ladder of experiments](../README.md#the-spectrum-of-experiments).

## The idea: generalization before sim2real

The goal is to go from "train on moving a block" to *asking* the arm to "stack
the blocks" or "put the block in the bowl" — i.e. following **language
instructions** and generalizing across tasks, not just across positions within
one task.

That capability comes from a different class of model than ACT. It does **not**
require simulation or sim2real transfer: we already have a working real-world
pipeline (teleoperate → record → train), and a pretrained
**VLA (vision-language-action)** model reuses all of it. Sim2real earns its
keep later — for scaling data or for skills that are unsafe to teleoperate —
not as the next step.

## Where the pieces fit

- **ACT** (rung 1) is a from-scratch policy with **no** language backbone. It
  learns the demonstrated sensorimotor mapping and generalizes to new object
  *positions*, not to new instructions or tasks. It cannot do "stack the blocks"
  unless stacking was demonstrated.
- **VLA** is a *category* of policy: a pretrained vision-language model bolted
  onto an action head, so the policy understands language + images before it
  ever sees our robot. Fine-tuning one on our LeRobot-format demos yields
  instruction-following across tasks plus better object/scene generalization.
- Specific VLAs are members of that category, not competitors to the category:

| Model | Who | Fit for SO-101 / single GPU | Note |
|-------|-----|------------------------------|------|
| **SmolVLA** | HuggingFace / LeRobot | Best first step | ~450M, built for cheap arms + community SO-10x datasets; drops into the same `lerobot-train` CLI as ACT (`--policy.type=smolvla`) |
| **π0 / openpi** | Physical Intelligence | Heavier | Flow-matching VLA on a PaliGemma backbone; the frontier-dexterity lineage |
| **GR00T N1.5** | NVIDIA | Heavier | Dual-system (VLM reasoning + fast diffusion action head); humanoid-oriented, works on arms |
| **MolmoAct** | Ai2 | Heavier | "Action reasoning" VLA — emits explicit spatial reasoning before acting; open weights + data |

## What a VLA does and does not buy us

**Does:** language grounding ("block", "bowl", "stack"), one policy for many
instructions, better generalization to novel positions / distractors / some
novel objects.

**Does not (at hobby-data scale):** reliable *zero-shot* execution of a skill
never demonstrated in any form. We still need demonstrations of stacking and of
bowl-placing — the VLA generalizes *across* them and to new arrangements, where
ACT would need a separate policy per task and still wouldn't follow the prompt.

## Recommended next step

Start with **SmolVLA** — lowest friction from the current ACT workflow — and
prove the generalization thesis before investing in π0 / GR00T / MolmoAct.

The real work before any training run is **dataset design**: pick 2–3
language-labeled tasks (e.g. move-the-block, stack, put-in-bowl), decide the
object/position variation, and make sure every episode is cleanly labeled with
its instruction. Then measure instruction-following and spatial generalization
separately.

Once a checkpoint exists, [vla-inference-azure.md](vla-inference-azure.md)
covers serving it from an always-on Azure GPU VM via LeRobot's async
inference (PolicyServer + RobotClient) instead of running it locally.
