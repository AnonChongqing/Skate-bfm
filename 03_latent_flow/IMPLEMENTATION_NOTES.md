# Stage 03 Implementation Audit

## 1. Existing evidence before Stage 03

Stage 01 is a frozen-policy inference baseline. Every joint action comes from
BFM0; HUSKY provides only simulation, contact, commands, rewards, and phase
state. The four curated videos are intentionally a compact evidence set:

| Video | Method | Why retained | Result |
| --- | --- | --- | --- |
| `01_best_transition.mp4` | Phase controller blends push/steer reward latents and uses BFM tracking latent only during push-to-steer. | Confirms the transition path executes while upright and exposes intermittent contact and separation. | 180 low-level steps, min root 0.706 m, mean board speed +0.193 m/s, contact 24.4%, final distance 0.680 m. |
| `02_safe_reactive.mp4` | Contact-aware `push_hold`/`push_drive` latent FSM with speed, contact, distance, and height gates. | Establishes the conservative closed-loop baseline and why physical gates are needed. | 190 steps, min root 0.665 m, mean board speed -0.129 m/s, contact 38.4%, final distance 0.927 m. |
| `03_kickaway_failure.mp4` | Aggressive reverse locomotion prompt. | Negative evidence: board speed alone is reward hacking when the board is kicked away. | 190 steps, min root 0.111 m, mean board speed +0.760 m/s, contact 2.6%. |
| `04_direct_steer.mp4` | CEM-adapted static BFM latent, signed rotate latent, and foot-marker latent feedback. | Best direct-steer evidence and the clearest demonstration of the frozen latent-search ceiling. | First complete contact loss moves from step 35 to 47; final distance improves 0.914 m to 0.574 m, but contact is not sustained after step 58. |

These results motivate Stage 03: a static prompt or hand-tuned latent blend can
react only along a few preselected directions. A learned 10 Hz flow policy can
choose a mode-conditioned local update from state, while keeping BFM0 as the
only low-level action generator. Twin skateboard critics rank and improve those
updates without training or bypassing BFM0.

## 2. Audited runtime interfaces

- HUSKY source: `husky_sim/src/mjlab_husky`.
- BFM model: `/63data1/hwh_data/Skate-bfm/models/bfm0/checkpoint/model`.
- BFM observations: `state[64]`, `history_actor[372]`, `last_action[29]`,
  and `privileged_state[463]`.
- BFM latent/action dimensions are 256 and 29. HUSKY has 23 shared actions;
  its model omits six BFM wrist joints.
- MuJoCo/Warp physics is 200 Hz (`dt=0.005`); HUSKY/BFM control is 50 Hz
  (`decimation=4`, `step_dt=0.02`); five low-level steps produce 10 Hz flow.
- The existing wrapper supports one environment. Stage 03 v0 branches by exact
  snapshot/restore rather than approximate resets.
- `_get_feet_contact_b()` and `_get_feet_contact_g()` mutate contact-history
  buffers, so Stage 03 reads each once per low-level step.
- Force vectors exist for both feet against board and ground.
- HUSKY returns terminated separately from timeouts, but internally resets a
  terminal environment during `step()`; terminal flags are retained explicitly.

## 3. Differences from the plan assumptions

1. There is no independent Stage 02 implementation. The skateboard-aware
   heuristic is `SteerControl` inside Stage 01. Stage 03 uses it only as a
   baseline/candidate source and does not claim a separate Stage 02 policy.
2. HUSKY has velocity and heading commands, not a persistent XY goal. The goal
   vector derives a local direction and heading error from the real command.
3. Real privileged dimensions are robot 79, board 19, contact 26, and
   goal/mode 16. The proposed 28-dimensional goal/mode vector cannot be filled
   without invented fields, so schema `skate-flow-v1` keeps 16 named fields.
4. Stage 01's action adapter depends on a live environment. Stage 03 extracts
   indices, scales, and references into a frozen tensor batch module.
5. BFM inference source is vendored under Stage 03 and is not imported through
   Stage 01. HUSKY remains the repository's dedicated `husky_sim` component.

## 4. Ownership

Frozen: BFM actor, maps, BFM critics, discriminator, normalizer, latent basis,
and action adapter tensors. Trainable: Flow Policy, independent Q1/Q2, and
optional entropy temperature. Target Q networks change only by soft update.
No HUSKY policy checkpoint or hardcoded joint target is used for control.

## 5. Data placement

Code, tests, documentation, and compact summaries remain in Git. Basis files,
datasets, checkpoints, logs, and evaluation videos are written only below
`/63data1/hwh_data/Skate-bfm`.
