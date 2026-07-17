# Stage 03 Training Runbook

All stages run in the foreground and print progress in the terminal that starts
them. Run one stage at a time. Training artifacts remain under `/63data1` or the
Git-ignored `03_latent_flow/checkpoint` and `03_latent_flow/results/runs` links.

## 1. Shell Setup

```bash
cd /home/hu_wenhui/workspace/Skate-bfm
source activate.sh
export PYTHONUNBUFFERED=1
export SKATE_BFM_RUN_DATE="${SKATE_BFM_RUN_DATE:-$(date +%F)}"
CHECKPOINT_DIR="03_latent_flow/checkpoint/$SKATE_BFM_RUN_DATE/latent_flow_husky_parallel_v2"
```

Keep the printed `SKATE_BFM_RUN_DATE` for later resume commands; set that same
date explicitly when resuming from a new shell on another day.

The corrected HUSKY prior and latent basis already exist. Rebuild them only
after changing expert motion data or latent source prompts:

```bash
CUDA_VISIBLE_DEVICES=3 python 03_latent_flow/scripts/build_husky_prior.py \
  --config 03_latent_flow/configs/train/large.yaml

CUDA_VISIBLE_DEVICES=3 python 03_latent_flow/scripts/build_latent_basis.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --output /63data1/hwh_data/Skate-bfm/latent_basis/skate_mode_basis_husky_parallel_v2.pt
```

## 2. Branch Collection

Use at most three GPUs. One foreground command launches the workers, keeps their
progress in this terminal, waits for every worker, validates and merges their
shards, then deletes the temporary shard files. The final dataset is written
only when all workers succeed. Each progress line includes its shard number,
progress bar, anchor/candidate counters,
sampled horizon, throughput, ETA, return, phase reward totals, retention, and
contact loss. Each anchor batch samples a horizon from `0.5` to `1.0` seconds in
`0.1`-second increments. Every candidate flow is applied for the first `0.1`
seconds, then zero flow holds the resulting latent for the remaining horizon.
All 16 candidates belonging to the same anchor use the same sampled horizon.

```bash
python 03_latent_flow/scripts/collect_branches.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --gpus 3,4,5
```

Expected output:

```text
/63data1/hwh_data/Skate-bfm/datasets/latent_flow/husky_parallel_v2.pt
20,000 anchors x 16 candidates = 320,000 branch samples
```

### Branch Code

- `scripts/collect_branches.py`: CLI, GPU worker launch, shard range/seed,
  environment creation, automatic checked merge, and temporary-file cleanup.
- `skate_bfm_flow/algorithms/collector.py::_candidates`: zero, local Gaussian,
  prototype-directed, and uniform latent-flow candidates.
- `skate_bfm_flow/algorithms/collector.py::collect`: snapshot each anchor,
  sample the shared `0.5–1.0s` horizon, restore the same state for every
  candidate, execute one candidate update plus a zero-flow continuation,
  aggregate rewards, print progress, and assemble tensors.
- `skate_bfm_flow/env/snapshot.py`: exact HUSKY state capture and restore.
- `skate_bfm_flow/env/macro_env.py::step`: flow-to-latent mapping, frozen BFM0
  actions, HUSKY stepping, macro reward, and next state.
- `skate_bfm_flow/env/reward_adapter.py`: combined HUSKY/task reward terms.
- `skate_bfm_flow/data/branch_dataset.py`: save/load, basis checksum validation,
  and checked shard merge.

## 3. SAC Pretraining

One foreground command sequentially trains the Offline Twin-Q critic warm start
and the Flow-BC actor warm start. They retain separate configs because their
networks, losses, and step counts differ. If Q training fails, BC is not started.

```bash
CUDA_VISIBLE_DEVICES=3 python 03_latent_flow/scripts/pretrain.py \
  --q-config 03_latent_flow/configs/train/q_large.yaml \
  --bc-config 03_latent_flow/configs/train/bc_large.yaml
```

This produces `$CHECKPOINT_DIR/offline_q.pt` and
`$CHECKPOINT_DIR/flow_bc.pt`, which are both consumed by SAC.

## 4. Online SAC

GPU 3 trains while periodic policy evaluation uses GPU 4. This remains within
the three-GPU limit.

```bash
CUDA_VISIBLE_DEVICES=3 python 03_latent_flow/scripts/train_online_sac.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --set train.steps=1000000 \
  --set logging.eval_cuda_visible_devices=4 \
  --policy-checkpoint "$CHECKPOINT_DIR/flow_bc.pt" \
  --q-checkpoint "$CHECKPOINT_DIR/offline_q.pt"
```

## 5. Final Evaluation

```bash
CUDA_VISIBLE_DEVICES=3 python 03_latent_flow/scripts/evaluate_flow.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --checkpoint "$CHECKPOINT_DIR/sac_final.pt" \
  --episodes 20 \
  --suite standard \
  --compare-zero \
  --video-dir 03_latent_flow/results/runs/latent_flow_husky_parallel_v2/final_eval
```

Current Stage 03 uses the corrected HUSKY push latent prior but does not yet
train an AMP discriminator. Offline-Q and BC are warm starts; Twin-Q and the
Flow Policy continue updating together during online SAC rollout.
