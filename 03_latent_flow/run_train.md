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

Use at most three GPUs. The three processes share this terminal; each progress
line includes its shard number, progress bar, anchor/candidate counters,
throughput, ETA, return, phase reward totals, retention, and contact loss.

```bash
GPUS=(3 4 5)
PIDS=()

for shard in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" \
  python 03_latent_flow/scripts/collect_branches.py \
    --config 03_latent_flow/configs/train/large.yaml \
    --num-shards 3 \
    --shard-index "$shard" &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done
test "$FAILED" -eq 0
```

Do not merge if the final `test` fails. After all three shards finish:

```bash
CUDA_VISIBLE_DEVICES=3 python 03_latent_flow/scripts/collect_branches.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --merge-glob '/63data1/hwh_data/Skate-bfm/datasets/latent_flow/husky_parallel_v2.part-*-of-003.pt'
```

Expected output:

```text
/63data1/hwh_data/Skate-bfm/datasets/latent_flow/husky_parallel_v2.pt
20,000 anchors x 16 candidates = 320,000 branch samples
```

### Branch Code

- `scripts/collect_branches.py`: CLI, shard range/seed, environment creation,
  output naming, merge entry point.
- `skate_bfm_flow/algorithms/collector.py::_candidates`: zero, local Gaussian,
  prototype-directed, and uniform latent-flow candidates.
- `skate_bfm_flow/algorithms/collector.py::collect`: snapshot each anchor,
  restore the same state for every candidate, execute the finite horizon,
  aggregate rewards, print progress, and assemble tensors.
- `skate_bfm_flow/env/snapshot.py`: exact HUSKY state capture and restore.
- `skate_bfm_flow/env/macro_env.py::step`: flow-to-latent mapping, frozen BFM0
  actions, HUSKY stepping, macro reward, and next state.
- `skate_bfm_flow/env/reward_adapter.py`: combined HUSKY/task reward terms.
- `skate_bfm_flow/data/branch_dataset.py`: save/load, basis checksum validation,
  and checked shard merge.

## 3. Offline Twin-Q

```bash
CUDA_VISIBLE_DEVICES=3 python 03_latent_flow/scripts/train_offline_q.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --set q.target.type=finite_horizon_return \
  --set train.steps=200000
```

## 4. Flow Behavior Cloning

```bash
CUDA_VISIBLE_DEVICES=3 python 03_latent_flow/scripts/train_flow_bc.py \
  --config 03_latent_flow/configs/train/large.yaml \
  --set train.steps=100000
```

## 5. Online SAC

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

## 6. Final Evaluation

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
