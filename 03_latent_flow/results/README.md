# Stage 03 Results

Large run artifacts live under `/63data1/hwh_data/Skate-bfm`. This directory
keeps only compact, reviewed evidence and summaries suitable for Git.

## Engineering Validation

- Real feature inspection: actor `[1,1965]`, critic branches `[79,19,26,16]`.
- BFM observation: `[64,372,29,463]`; frozen BFM action 29D; HUSKY action 23D.
- Macro control: five 50 Hz low-level steps per 10 Hz flow update.
- Snapshot roundtrip maximum errors: robot joint state `7.06e-6`, board pose
  `2.98e-7`, macro reward `4.59e-6`.
- Same-state branch collection, offline Q update/checkpoint, BC update, and one
  online SAC update have executed successfully on GPU.

These checks establish a functioning learning pipeline. They are not evidence
that stable skateboarding has been learned. A curated evaluation video is added
only after a named formal checkpoint is evaluated with retention, contact,
distance, progress, and fall metrics together.

## `v0_engineering` Short Run

This run verifies artifact production and warm-start wiring; its scale is too
small for a performance claim.

```text
branch data: 10 anchors x 4 candidates, 10 low-level-step horizon
offline Q:   50 updates
BC:          50 updates
online SAC:  20 macro environment steps, warm-started from BC and offline Q
evaluation:  3 deterministic seeds
```

Offline Q training loss decreased substantially, but held-out ranking remained
weak because the dataset contains only ten anchors:

| Metric | Value |
| --- | ---: |
| Spearman | -0.067 |
| Kendall | -0.333 |
| NDCG | 0.737 |
| Top-1 regret | 0.000073 |
| Top-3 hit | 0.667 |
| Q disagreement | 0.042 |

Deterministic SAC rollout:

| Metric | Value |
| --- | ---: |
| Success rate | 0.000 |
| Contact-loss rate | 1.000 |
| Fall rate | 0.000 |
| Board contact | 0.249 |
| Retention | 0.569 |
| Board progress | 0.079 m |
| Final robot-board distance | 0.176 m |
| Episode return | 0.205 |

The policy loses legal board contact after roughly 0.2 s. This is expected for
the deliberately short run and confirms that the current bottleneck is basic
contact/balance, not heading refinement. The video records the complete early
failure rather than truncating to a favorable frame.

Artifacts:

```text
/63data1/hwh_data/Skate-bfm/datasets/latent_flow/v0_engineering_branches.pt
/63data1/hwh_data/Skate-bfm/checkpoints/latent_flow/v0_engineering/
/63data1/hwh_data/Skate-bfm/runs/latent_flow/v0_engineering/flow_eval.json
/63data1/hwh_data/Skate-bfm/runs/latent_flow/v0_engineering/offline_q/ranking.json
/63data1/hwh_data/Skate-bfm/runs/latent_flow/v0_engineering/eval_videos/episode_000.mp4
```
