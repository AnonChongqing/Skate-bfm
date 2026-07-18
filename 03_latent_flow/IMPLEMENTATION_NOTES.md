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
- The wrapper preserves batch dimensions through HUSKY observations, frozen
  BFM inference, 29D-to-23D action mapping, feature construction, reward, and
  snapshots. The formal config uses 64 environments.
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
`/63data1/hwh_data/Skate-bfm`. The Git-ignored
`03_latent_flow/results/runs` symlink exposes the run directory in the project.

## 6. HUSKY motion prior

The official `human_push_1.npy` and `human_push_2.npy` files are 50 Hz state
references, not policy actions. Their validated 36D layout is root position 3,
root quaternion `wxyz` 4, and 29 joint positions in MuJoCo order. Following the
original AMP loader, source joint indices `[0:19,22:26]` select the HUSKY 23DoF
joint vector while omitting six wrist DoFs. Stage 03 maps those joints into a
BFM tracking observation and passes them through BFM0's frozen backward map.
The resulting 256D samples improve the PUSH PCA basis. The prior builder also
interpolates HUSKY's default and steer endpoint joint poses in both directions
and encodes 64 frames per direction with frozen BFM tracking inference. These
synthetic MOUNT/DISMOUNT tracking bases are not motion demonstrations. No HUSKY
policy checkpoint is loaded, and no HUSKY action supervises or controls the
Flow Policy. STEER and RECOVER still depend on Stage 01 prompts, phase rewards,
branch rollouts, and online SAC experience.

## 7. Parallel robust training

The formal v2 branch collector treats each HUSKY environment as a distinct
anchor and evaluates one candidate index for all anchors in parallel. Exact
snapshot restore still prevents state differences between candidates belonging
to the same anchor. Online SAC stores all environment transitions, resets only
the terminated environment IDs, and applies update ratio per collected
transition.

Formal branch collection assigns equal anchor quotas to PUSH, MOUNT
(push-to-steer), STEER, and DISMOUNT (steer-to-push). Stable phases use their
physical HUSKY reset state. Transition control begins 0.3 seconds before the
matching contact boundary and uses frozen-BFM tracking-flow rollout to sample
intermediate states.
The candidate set adds a latent direction toward the next phase prototype; it
never substitutes HUSKY reference joints for BFM-generated actions. Merge
metadata records the resulting per-phase anchor counts.

Branch semantics are `single_macro_then_phase_baseline_v2`: the tested flow is
applied for one 0.1-second macro step. Stable phases then hold the resulting
latent, while transition phases continue with state-dependent frozen-BFM
tracking flow. This evaluates a local transition decision without assuming one
static latent can represent an entire time-varying transition.

The large config also adopts HUSKY's command range, COM/friction domain
randomization, interval pushes, noisy actor observations, and mixed push/steer
initialization. A command curriculum expands from a narrow range to the full
HUSKY range. Clean privileged Q features are retained. Retention-gated progress
and safety terms now contribute to the optimized reward rather than existing
only as diagnostics.

Random interval pushes are enabled for online SAC but disabled while comparing
same-anchor branches. Static domain randomization remains active during branch
collection; disabling interval events prevents candidate labels from being
confounded by different random disturbances.

The multimode basis contains 416 PUSH samples, 94 MOUNT samples, 30 STEER
samples, 94 DISMOUNT samples, and 30 RECOVER samples, providing 16 nonzero PCA
directions per mode. Only PUSH includes official HUSKY motion. MOUNT/DISMOUNT
add synthetic BFM tracking interpolation; the remaining samples are curated
BFM latents from Stage 01 searches and must not be described as expert
demonstrations.

## 8. Scaling, logging, and evaluation

Independent branch shards can run on separate GPUs. Shard seeds and anchor ID
ranges are disjoint, and merge-time checks reject incompatible or duplicate
samples. This is valid data parallelism. Online SAC remains one policy, one
replay, and one optimizer process with vectorized HUSKY environments; launching
independent SAC jobs is not equivalent to distributed training.

All trainers use one logger for terminal progress, JSONL, CSV, and TensorBoard.
Online SAC accumulates rollout, reward-component, mode, command, optimization,
replay, and curriculum metrics between reports. Policy evaluation provides a
four-command deterministic suite, a same-seed zero-flow comparison, 50 Hz MP4
capture, and Viser.

The optimized scalar reward is the phase-selected HUSKY push/steer/transition
reward plus HUSKY regularization and explicit retention-gated board/heading
progress, retention, upright, fall, and illegal-contact terms. Latent magnitude
and smoothness are logged as reward diagnostics but regularize the Actor loss,
so they are not counted twice in the environment return. The offline critic
learns finite-horizon branch return; online Twin-Q learns the semi-MDP SAC
target. Per-anchor ranking now audits whether both return and Q agree with
physical progress, retention, contact loss, illegal contact, and falls. Those
measurements are necessary evidence, not a mathematical guarantee that reward
weights cannot be exploited.

Offline-Q training samples complete 16-candidate anchor groups instead of
independent rows. Mode/outcome balancing prevents common stable phases from
dominating transition and failure examples. Its objective adds same-anchor
pairwise return ranking and a safe-over-failure margin to Huber return
regression. Formal Q and BC reject datasets that lack any of the four required
behavior phases.

## 9. HUSKY reference boundary and dated evidence

HUSKY defines the skateboard articulation, wheel/deck contacts, body-frame board
velocity, absolute target heading, push/steer/transition phases, transition
Bezier/Slerp targets, and baseline reward terms. Stage 03 references these
quantities but does not delegate policy decisions to HUSKY. Retention, safety,
latent regularization, branch return, Twin-Q, and zero-flow comparisons remain
independent checks. The HUSKY PPO actor remains unused.

This correction changes feature and label meaning without changing dimensions,
so the schema is `skate-flow-v2`. V1 branch data and checkpoints are rejected to
prevent a relative-heading/world-X model from being silently mixed with the
absolute-heading/body-frame objective.

Online logs expose every HUSKY reward term alongside Stage 03 reward, physical,
optimization, and replay metrics. Configured evaluation intervals save the
current policy and launch the existing evaluator in a subprocess, isolating its
random state from training. Push, push-to-steer, and steer clips use explicit
phase starts and never inject hardcoded actions. Evaluation failure is reported
without terminating a long training run.

All trained models are visible under
`03_latent_flow/checkpoint/YYYY-MM-DD/<experiment>`. This path links to the
checkpoint area on `/63data1`, retaining the project layout without placing
large binary files in Git. `source activate.sh` creates the link when needed.

Formal-scale post-processing uses sorted `(anchor_id, candidate_id)` groups.
BC target construction is vectorized over groups, while Q ranking batches neural
inference and moves only compact predictions and quality labels to CPU. This
removes the previous quadratic anchor-by-dataset scan that was acceptable only
for smoke data.
