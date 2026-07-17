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
root linear/angular velocity 6, and MuJoCo-order joint position 23. They are not
actions. `build_husky_prior.py` maps the 23 joint positions into BFM0's 29-joint
tracking observation and encodes them with the frozen BFM backward map. The
result augments the PUSH latent basis. The other mode bases use the retained
Stage 01 push/steer search results; they do not claim unavailable HUSKY steer
motion demonstrations.

Run the formal pipeline in this order. The first two commands are one-time data
preparation; the remaining training commands can be resumed from checkpoints.

```bash
cd /home/hu_wenhui/workspace/Skate-bfm
source activate.sh
export CUDA_VISIBLE_DEVICES=6

python 03_latent_flow/scripts/build_husky_prior.py \
  --config 03_latent_flow/configs/train/large.yaml

python 03_latent_flow/scripts/build_latent_basis.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --output /63data1/hwh_data/Skate-bfm/latent_basis/skate_mode_basis_husky_parallel_v2.pt

python 03_latent_flow/scripts/collect_branches.py \
  --config 03_latent_flow/configs/train/large.yaml

python 03_latent_flow/scripts/train_offline_q.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --set q.target.type=finite_horizon_return \
  --set train.steps=200000

python 03_latent_flow/scripts/train_flow_bc.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --set train.steps=100000

python 03_latent_flow/scripts/train_online_sac.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --set train.steps=1000000 \
  --policy-checkpoint /63data1/hwh_data/Skate-bfm/checkpoints/latent_flow/latent_flow_husky_parallel_v2/flow_bc.pt \
  --q-checkpoint /63data1/hwh_data/Skate-bfm/checkpoints/latent_flow/latent_flow_husky_parallel_v2/offline_q.pt
```

The large config runs 64 HUSKY environments in parallel. Branch collection
uses 20,000 anchors, 16 same-state candidates per anchor, and a 25-low-step
horizon. Each environment represents an independent anchor; candidate index
`k` is evaluated for all 64 anchors concurrently after exact snapshot restore.
The 250,000-transition replay leaves headroom for MuJoCo-Warp, frozen BFM, and
Twin-Q on a 48 GB RTX 4090.

Branch collection can be sharded across independent GPUs because every shard
owns disjoint anchors and restores candidates only within its own anchor. For
example, use four currently idle GPUs and then merge the checked shards:

```bash
GPUS=(3 4 5 6)
for shard in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" \
    python 03_latent_flow/scripts/collect_branches.py \
      --config 03_latent_flow/configs/train/large.yaml \
      --num-shards "${#GPUS[@]}" \
      --shard-index "$shard" &
done
wait

python 03_latent_flow/scripts/collect_branches.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --merge-glob '/63data1/hwh_data/Skate-bfm/datasets/latent_flow/husky_parallel_v2.part-*.pt'
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

Evaluate the final checkpoint with both metrics and video:

```bash
python 03_latent_flow/scripts/evaluate_flow.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --checkpoint /63data1/hwh_data/Skate-bfm/checkpoints/latent_flow/latent_flow_husky_parallel_v2/sac_final.pt \
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

Offline Q and BC warm start:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/train_offline_q.py \
  --config 03_latent_flow/configs/train/offline_q.yaml

CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/train_flow_bc.py \
  --config 03_latent_flow/configs/train/bc_flow.yaml
```

Online semi-MDP SAC:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/train_online_sac.py \
  --config 03_latent_flow/configs/train/online_sac.yaml
```

Replay is preallocated tensor storage. Branch train/validation splitting is by
`anchor_id`, so candidates from one physical state cannot leak across splits.

## Evaluation

Q candidate ranking:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/evaluate_q.py \
  --config 03_latent_flow/configs/eval/q_ranking.yaml \
  --checkpoint /63data1/hwh_data/Skate-bfm/checkpoints/latent_flow/latent_flow_v0/offline_q.pt
```

Deterministic rollout and MP4:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/evaluate_flow.py \
  --config 03_latent_flow/configs/eval/rollout.yaml \
  --checkpoint /63data1/hwh_data/Skate-bfm/checkpoints/latent_flow/latent_flow_v0/flow_bc.pt \
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
diagnostics test reward/Q alignment but do not guarantee it. Rollout metrics keep board progress,
contact, retention, robot-board distance, fall, timeout, and success separate.

## Checkpoints And Logs

Formal runs write `metrics.jsonl`, `summary.csv`, TensorBoard data, and videos
under `03_latent_flow/results/runs`. That path is a Git-ignored symlink to
`/63data1/hwh_data/Skate-bfm/runs/latent_flow`, so artifacts are visible in the
project while physically remaining on the data disk. Datasets and checkpoints
also remain under `/63data1/hwh_data/Skate-bfm`.

Offline Q, BC, and online SAC print progress, elapsed time, throughput, ETA, and
grouped metrics. Online reports include reward diagnostics and Actor regularizers,
push/mount/steer/dismount/recover occupancy, commands, termination rates,
Q/Actor/alpha losses, replay fill, parallel environment count, and curriculum
progress. TensorBoard can be opened with:

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
