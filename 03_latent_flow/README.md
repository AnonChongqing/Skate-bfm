# Stage 03: Skateboard Latent Flow

Stage 03 learns state-conditioned local updates in BFM0's 256D latent space.
BFM0 remains the only low-level action generator; HUSKY supplies MuJoCo/Warp
physics, contact, task state, and reward.

## Architecture

```text
robot + board + contact + goal + mode + current z + previous flow
                              |
                frame-stacked Flow Policy (10 Hz)
                              |
                     flow u in R^16
                              |
        mode basis U_m -> tangent residual -> ||z|| = 16
                              |
                   frozen BFM0 (50 Hz)
                              |
             29D action -> frozen 23D adapter
                              |
                 HUSKY physics (200 Hz)
                              |
       macro reward + independent Twin Skateboard Q
```

One flow is held for five BFM/HUSKY policy steps. The default macro discount is
0.99 and low-level reward discount is `0.99 ** (1/5)`.

## Ownership

Frozen: BFM actor, maps, BFM critics, discriminator, observation normalizer,
mode basis, and action adapter. Trainable: tanh-Gaussian Flow Policy, independent
Q1/Q2, and SAC entropy temperature. HUSKY checkpoint actions and hardcoded joint
targets are not used.

## Features

The deployable actor frame has 393 values and is stacked over five frames:

- robot: 79;
- board estimate: 18;
- deployable contact: 14;
- velocity/heading goal: 5;
- mode one-hot: 5;
- current BFM latent: 256;
- previous flow: 16.

The privileged Q state uses real fields only: robot 79, board 19, contact 26,
goal/mode 16. See `IMPLEMENTATION_NOTES.md` for the difference from the proposed
28D goal/mode profile.

## Latent Mapper

Each mode has an orthonormal basis `[256, flow_dim]`. The default update removes
the radial component from `U_m u`, applies a configurable step, then projects
back to radius `sqrt(256)=16`. Basis files are fixed and checksum validated.

## Twin Q

Q1 and Q2 have separate parameters and evaluate the same flow candidate. Q
network structure, input construction, aggregation, target strategy, and loss
are separate modules.

`q.input_profile`:

- `minimal`: state + current z + flow;
- `candidate`: minimal + candidate z + latent statistics;
- `preview`: minimal + frozen BFM action preview;
- `full_preview`: candidate + previous flow + 23D preview.

Configurable Q calculation:

```text
q.target.type: finite_horizon_return | semi_mdp_td | sac_td | td3_td
q.target.aggregation: min | mean | mean_minus_std | min_minus_disagreement
q.loss.type: huber | mse | mae
q.preview.type: none | action_29d | action_23d | lower_body_12d
q.state_profile: privileged | deployable
latent.flow_dim: 8 | 16 | 32
control.macro_steps: 2 | 5 | 10
```

Repeated `--set key=value` overrides are validated by Pydantic.

## Setup

```bash
cd /home/hu_wenhui/workspace/Skate-bfm
source activate.sh

CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/build_latent_basis.py \
  --config 03_latent_flow/configs/base.yaml \
  --output /63data1/hwh_data/Skate-bfm/latent_basis/skate_mode_basis_v0.pt
```

## Data And Training

### Formal HUSKY-prior run

The official HUSKY push references contain 401 frames in two 50 Hz motion
files. Each 36D frame is parsed as root position 3, root quaternion `wxyz` 4,
and MuJoCo-order joint position 29. They are reference poses, not policy actions.
Following HUSKY's AMP loader, source joint indices `[0:19,22:26]` select the
23DoF robot joints while omitting six wrist DoFs. `build_husky_prior.py` maps
those 23 joint positions into BFM0's 29-joint tracking observation and encodes
them with the frozen BFM backward map. The result augments the PUSH latent
basis. The other mode bases use the retained Stage 01 push/steer search results;
they do not claim unavailable HUSKY steer motion demonstrations.

The formal workflow uses two foreground commands: branch collection with an
automatic checked merge, followed by Offline-Q, Flow-BC, and SAC training. Each
stage prints progress, metrics, ETA, evaluation output, failures, and artifact
paths directly in the launching terminal. No second terminal or log-tail
command is required. The HUSKY prior and latent basis are existing prerequisites
and only need rebuilding when their source data changes.
The heading/progress semantics use schema `skate-flow-v2`; branch datasets and
checkpoints made with v1 are intentionally rejected and must not be reused.

```bash
cd /home/hu_wenhui/workspace/Skate-bfm
source activate.sh
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export SKATE_BFM_RUN_DATE="$(date +%F)"
CHECKPOINT_DIR="03_latent_flow/checkpoint/$SKATE_BFM_RUN_DATE/latent_flow_husky_parallel_v2"

python 03_latent_flow/scripts/collect_branches.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --gpus 3,4,5

python 03_latent_flow/scripts/train.py \
  --q-config 03_latent_flow/configs/train/q_large.yaml \
  --bc-config 03_latent_flow/configs/train/bc_large.yaml \
  --sac-config 03_latent_flow/configs/train/large.yaml \
  --sac-set train.steps=1000000 \
  --sac-set 'logging.eval_cuda_visible_devices="4"'
```

Wait for collection to finish before starting training. Closing the terminal
terminates its foreground pipeline; `train.py --start-stage sac --sac-resume`
can continue an interrupted SAC checkpoint without rerunning Q and BC.

The large config runs 64 HUSKY environments in parallel. Branch collection
uses 20,000 anchors, 16 same-state candidates per anchor, and a horizon sampled
uniformly from 25 to 50 low-level steps (`0.5–1.0s`). Each
candidate flow is applied for one 0.1-second macro step, followed by zero-flow
continuation at the resulting latent. Each environment represents an independent
anchor; candidate index `k` is evaluated for all 64 anchors concurrently after
exact snapshot restore, and candidates from one anchor share the same horizon.
The 250,000-transition replay leaves headroom for MuJoCo-Warp, frozen BFM, and
Twin-Q on a 48 GB RTX 4090.

Branch collection can be sharded across at most three independent GPUs because
every shard owns disjoint anchors and restores candidates only within its own
anchor. The foreground launcher waits for all workers, validates and merges the
shards, and removes temporary parts. The complete command is maintained in
[`run_train.md`](run_train.md):

```bash
python 03_latent_flow/scripts/collect_branches.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --gpus 3,4,5
```

The merge rejects mismatched fields, basis paths, candidate counts, horizons,
and duplicate `(anchor_id, candidate_id)` pairs. Online SAC uses 64 parallel
environments on one GPU. True multi-GPU SAC is intentionally not exposed yet:
independent SAC processes would train different policies unless replay,
gradients, entropy temperature, and target networks were synchronized.

Robustness settings follow HUSKY training practice:

- velocity/heading commands ramp from `[0.4,0.8]`, `[-0.2,0.2]` to
  `[0,1.5]`, `[-pi/4,pi/4]` over 250,000 environment transitions;
- startup randomizes robot/board COM and robot/board/foot/wheel friction;
- interval velocity pushes remain active during online SAC; branch collection
  disables them so candidates from one snapshot face identical dynamics;
- actor joint, velocity, angular-velocity, and gravity features receive
  HUSKY-scale observation noise while Q keeps clean privileged features;
- resets mix 65% push and 35% steer starts, and replay samples modes evenly;
- branch/SAC reward combines HUSKY reward with retention-gated board/heading
  progress, retention, upright, fall, and illegal-contact terms.

HUSKY is a reference, not the sole decision rule. Its articulated skateboard,
body-frame velocity, target heading, contact phases, and individual reward terms
provide physically grounded inputs and diagnostics. Stage 03 still optimizes a
combined objective with retention, safety, latent regularization, finite return,
and independent Twin-Q. No HUSKY policy action is used.

The large run saves a checkpoint every 10,000 transitions and performs an
independent policy evaluation every 100,000 transitions. Each evaluation writes
`push.mp4`, `push2steer.mp4`, `steer.mp4`, and `metrics.json` below:

```text
03_latent_flow/results/runs/<experiment>/online_sac/policy_eval/step_<N>/
```

The three clips use the same learned policy but controlled HUSKY phase starts:
push phase 0.0, push-to-steer phase 0.4 with Bezier/Slerp transition targets, and
steer phase 0.5 from HUSKY's board-relative steer pose. They are evaluation only
and do not inject actions or gradients into training.

By default evaluation reuses the training GPU. The formal `train.py` command
above reserves GPU 4 for evaluation through
`--sac-set 'logging.eval_cuda_visible_devices="4"'` without changing the GPU 3
training device.

Evaluate the final checkpoint with both metrics and video:

```bash
python 03_latent_flow/scripts/evaluate_flow.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --checkpoint "$CHECKPOINT_DIR/sac_final.pt" \
  --episodes 20 \
  --suite standard \
  --compare-zero \
  --video-dir /home/hu_wenhui/workspace/Skate-bfm/03_latent_flow/results/runs/latent_flow_husky_parallel_v2/eval_videos
```

The standard suite evaluates slow straight, straight, left, and right commands
and writes one MP4 per scenario. `--compare-zero` reruns identical seeds with a
zero latent flow, so the report separates gains from the frozen BFM baseline.

This is a real training plan, not a performance guarantee. The HUSKY data adds
a push-motion prior but contains no steer labels, and Stage 03 still needs
rollout data to learn board retention, transition, steering, and recovery.

### Small pipeline commands

Collect exact same-state branches:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/collect_branches.py \
  --config 03_latent_flow/configs/train/collect_branches.yaml \
  --set branch.num_anchors=5000 \
  --set branch.candidates_per_anchor=16 \
  --output /63data1/hwh_data/Skate-bfm/datasets/latent_flow/branches_v0.pt
```

Offline Q, BC, and online SAC:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/train.py \
  --q-config 03_latent_flow/configs/train/offline_q.yaml \
  --bc-config 03_latent_flow/configs/train/bc_flow.yaml \
  --sac-config 03_latent_flow/configs/train/online_sac.yaml
```

Replay is preallocated tensor storage. Branch train/validation splitting is by
`anchor_id`, so candidates from one physical state cannot leak across splits.
BC groups all candidates by anchor and selects targets vectorially. Q ranking
runs candidate inference in 4096-row batches and computes per-anchor metrics on
CPU, avoiding an all-dataset scan for every one of the 20,000 anchors.

## Evaluation

Q candidate ranking:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/evaluate_q.py \
  --config 03_latent_flow/configs/eval/q_ranking.yaml \
  --checkpoint 03_latent_flow/checkpoint/YYYY-MM-DD/latent_flow_v0/offline_q.pt
```

Deterministic rollout and MP4:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/evaluate_flow.py \
  --config 03_latent_flow/configs/eval/rollout.yaml \
  --checkpoint 03_latent_flow/checkpoint/YYYY-MM-DD/latent_flow_v0/flow_bc.pt \
  --episodes 10 \
  --suite standard \
  --compare-zero \
  --video-dir /home/hu_wenhui/workspace/Skate-bfm/03_latent_flow/results/runs/eval_videos
```

Viser uses the same macro policy at 10 Hz and can be viewed through the existing
SSH port-forward workflow:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/evaluate_flow.py \
  --config 03_latent_flow/configs/eval/rollout.yaml \
  --checkpoint <flow-checkpoint.pt> \
  --speed 0.8 --heading 0.4 \
  --viewer viser --port 8080
```

Q metrics include Spearman, Kendall, Top-1 regret, Top-3 hit rate, NDCG,
failure-last rate, and Q disagreement. The Q report also measures both finite
return and learned-Q alignment against board progress, heading progress,
retention, contact loss, illegal contact, and falls. Positive progress,
retention, and non-failure gaps are expected; failure fields are sign-normalized
so a higher correlation still means safer ranking. Constant, unobserved outcomes
are omitted instead of being reported as artificial perfect correlations. These
diagnostics test reward/Q alignment but do not guarantee it. Rollout metrics
keep board progress, contact, retention, robot-board distance, fall, timeout,
and success separate.

## Checkpoints And Logs

Formal runs write `metrics.jsonl`, `summary.csv`, TensorBoard data, and videos
under `03_latent_flow/results/runs`. That path is a Git-ignored symlink to
`/63data1/hwh_data/Skate-bfm/runs/latent_flow`, so artifacts are visible in the
project while physically remaining on the data disk. `03_latent_flow/checkpoint`
is likewise a Git-ignored link to the data disk. Models are grouped as
`checkpoint/YYYY-MM-DD/<experiment>/`; set `SKATE_BFM_RUN_DATE` once before the
Offline-Q, BC, and SAC commands so all three stages share one directory.

Branch collection, Offline Q, BC, and online SAC print progress, elapsed time,
throughput, ETA, and grouped metrics directly in the terminal running that
stage. Online reports include reward diagnostics and Actor regularizers,
push/mount/steer/dismount/recover occupancy, commands, termination rates,
Q/Actor/alpha losses, replay fill, parallel environment count, curriculum
progress, every HUSKY reward term, body-frame board speed, speed/heading error,
board tilt, root height, board distance, and phase occupancy. TensorBoard can be
opened with:

```bash
tensorboard --logdir /63data1/hwh_data/Skate-bfm/runs/latent_flow --port 6006
```

## Tests

```bash
CUDA_VISIBLE_DEVICES=6 pytest -q 03_latent_flow/tests
```

The two test modules cover config and motion-schema validation, mapper geometry,
action mapping, all Q profiles, target semantics, replay, offline overfit, live
HUSKY features, macro stepping, snapshot roundtrip, and BFM freezing.

## Current Limitations

- The 64-env path is validated on one RTX 4090; `env.num_envs` can be reduced
  when another process occupies substantial GPU memory, or raised to 128 only
  after checking memory and throughput on the selected GPU.
- Multi-GPU branch collection is supported; synchronized multi-GPU online SAC
  is not implemented.
- HUSKY has heading/velocity commands but no persistent XY navigation goal.
- Fixed-FSM mode labels are used in v0; autonomous mode switching is reserved.
- HUSKY provides push motion only. Other mode bases use search-derived BFM
  latents and simulation returns, not expert motion imitation.
- Short engineering runs validate the training loop but do not establish stable
  skateboard success. Long branch collection and SAC training are still needed.
- MP4 and Viser paths are both available, but Viser advances at the 10 Hz macro
  policy rate rather than displaying every 50 Hz BFM substep.
