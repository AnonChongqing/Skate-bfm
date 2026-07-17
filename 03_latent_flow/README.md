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

## Setup And Inspection

```bash
cd /home/hu_wenhui/workspace/Skate-bfm
source activate.sh

CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/build_latent_basis.py \
  --config 03_latent_flow/configs/base.yaml \
  --output /63data1/hwh_data/Skate-bfm/latent_basis/skate_mode_basis_v0.pt

CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/inspect_stage03.py \
  --config 03_latent_flow/configs/base.yaml
```

## Data And Training

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
  --video-dir /63data1/hwh_data/Skate-bfm/runs/latent_flow/eval_videos
```

Viser uses the same macro policy at 10 Hz and can be viewed through the existing
SSH port-forward workflow:

```bash
CUDA_VISIBLE_DEVICES=6 python 03_latent_flow/scripts/evaluate_flow.py \
  --config 03_latent_flow/configs/eval/rollout.yaml \
  --checkpoint <flow-checkpoint.pt> \
  --viewer viser --port 8080
```

Q metrics include Spearman, Kendall, Top-1 regret, Top-3 hit rate, NDCG,
failure-last rate, and Q disagreement. Rollout metrics keep board progress,
contact, retention, robot-board distance, fall, timeout, and success separate.

## Checkpoints And Logs

Formal runs write `metrics.jsonl`, `summary.csv`, and checkpoints beneath the
configured data root. Checkpoints include model/config/schema metadata. Large
datasets, weights, logs, and videos must remain under
`/63data1/hwh_data/Skate-bfm`; only compact summaries belong in Git.

## Tests

```bash
CUDA_VISIBLE_DEVICES=6 pytest -q 03_latent_flow/tests
```

The suite covers config validation, mapper geometry, action mapping, all Q input
profiles, target semantics, replay, offline overfit, live feature shapes, macro
step, exact snapshot roundtrip, and BFM freezing.

## Current Limitations

- The environment wrapper is single-instance; branch collection uses exact
  snapshot/restore and is slower than a validated vector clone.
- HUSKY has heading/velocity commands but no persistent XY navigation goal.
- Fixed-FSM mode labels are used in v0; autonomous mode switching is reserved.
- Short engineering runs validate the training loop but do not establish stable
  skateboard success. Long branch collection and SAC training are still needed.
- MP4 and Viser paths are both available, but Viser advances at the 10 Hz macro
  policy rate rather than displaying every 50 Hz BFM substep.
