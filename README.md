# Skate-BFM

Self-contained experiment workspace for running BFM0-generated humanoid motion
in the HUSKY skateboard simulator.

## Layout

- `husky_sim/`: local copy of the HUSKY simulation code, assets, checkpoints, and task configs used by this project.
- `01_bfm0_motion_husky/`: BFM0 produces 29D motion, the adapter runs it in HUSKY-23DoF, HUSKY rewards score pushing and steering, and key-body goal tracking scores transitions.
- `02_feedback/`: skateboard-aware feedback around BFM0 motion, including contact, board roll/yaw, heading, and foot-placement corrections.
- `03_latent_flow/`: frozen-BFM latent-flow learning with exact same-state branches, Twin Skate Q, BC warm start, and semi-MDP SAC.
- `activate.sh`, `run.sh`, `setup.sh`: project-local environment and launch entry points.

Each numbered stage owns its own code package, scripts, configs, notes, and
results. There is intentionally no shared top-level `common/` package. If two
stages need similar code, copy it into the stage first and then decide whether a
proper project package is warranted.

## Data Layout

Large files are outside Git but under a project-owned data root:

```bash
/63data1/hwh_data/Skate-bfm/
  envs/skate-bfm/
  models/bfm0/
  cache/
```

The source tree does not import code from the original BFM-Zero or HUSKY clone.
Both vendored packages record their upstream revisions in `UPSTREAM.md`.

## Setup

The current server already has the independent environment and model. On a new
machine, place the BFM0 model under the data layout above and run:

```bash
cd /home/hu_wenhui/workspace/Skate-bfm
USE_PROXYCHAINS=1 bash setup.sh
source activate.sh
```

## Run

Viser interactive view on port 8080:

```bash
CUDA_VISIBLE_DEVICES=6 ./run.sh \
  --device cuda --husky-device cuda:0 --mean \
  --control phase \
  --viewer viser --port 8080 --steps 0
```

Record a six-second video without starting Viser:

```bash
CUDA_VISIBLE_DEVICES=6 ./run.sh \
  --steps 300 --device cuda --husky-device cuda:0 --mean \
  --control phase \
  --output /63data1/hwh_data/Skate-bfm/runs/run.json \
  --video /63data1/hwh_data/Skate-bfm/runs/run.mp4
```

`--viewer viser` and `--video` are intentionally separate modes. For remote
Viser access, forward local port 8080 to server port 8080.

Stage 01 is currently a partial baseline. It keeps the G1 upright and runs a
live BFM prompt state machine, but no tested zero-shot locomotion latent yet
combines positive board velocity with sustained foot-board contact. See
`01_bfm0_motion_husky/notes/results.md` before interpreting board-speed-only
results.

Stage 03 commands, feature schemas, Q switches, training stages, and the current
short engineering result are documented in `03_latent_flow/README.md` and
`03_latent_flow/results/README.md`.
