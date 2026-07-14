# Stage 01: BFM0 Motion in HUSKY

This stage runs a frozen BFM0 actor in the HUSKY MuJoCo/Warp skateboard
simulation. It is self-contained under `Skate-bfm`: no source is imported from
the original BFM-Zero or humanoid_skateboarding clones.

## Boundary

- BFM0 produces every 29D robot action.
- The adapter maps the 23 shared joints into HUSKY; six BFM wrist actions are
  dropped because HUSKY uses the G1-23DoF model.
- HUSKY provides dynamics, contact state, command state, and task rewards.
- No HUSKY checkpoint policy or hardcoded joint target controls the robot.
- Push/steer use BFM reward prompts; tracking inference is active only during
  transition.

## Run

From the project root:

```bash
cd /home/hu_wenhui/workspace/Skate-bfm
source activate.sh

CUDA_VISIBLE_DEVICES=6 ./run.sh \
  --device cuda --husky-device cuda:0 --mean \
  --control phase --steps 190 \
  --output /63data1/hwh_data/Skate-bfm/runs/default_final.json \
  --video /63data1/hwh_data/Skate-bfm/runs/default_final.mp4
```

The safe default uses `/63data1/hwh_data/Skate-bfm/prompts/push_back.npy` as a
drive source. It is BFM0 `move-ego-180-0.3[2]`; the live
`push_hold`/`push_drive` state machine projects only 30% of it into the
zero-speed hold prompt, based on board contact, ground contact, board distance,
and board speed.

Viser mode is separate from MP4 recording:

```bash
CUDA_VISIBLE_DEVICES=6 ./run.sh \
  --device cuda --husky-device cuda:0 --mean \
  --control phase --viewer viser --port 8080 --steps 0
```

Direct on-board steering starts from HUSKY's official steer pose and runs only
BFM0 reward latents after reset:

```bash
CUDA_VISIBLE_DEVICES=6 ./run.sh \
  --device cuda --husky-device cuda:0 --mean \
  --control steer \
  --steer-z /63data1/hwh_data/Skate-bfm/prompts/steer_adapt.npy \
  --turn-index 9 --turn-mix 0.1 \
  --follow-mix 0.05 --follow-start 0.02 \
  --follow-pos-gain 0 --follow-foot-gain 1 \
  --steps 120 \
  --output /63data1/hwh_data/Skate-bfm/runs/direct_steer.json \
  --video /63data1/hwh_data/Skate-bfm/runs/direct_steer.mp4
```

The foot-marker feedback delays the first complete contact loss from step 35
to step 47 and reduces the 120-step final robot-board distance from `0.914 m`
to `0.574 m`. It still loses sustained board contact after step 58, so this is
an improved short-steer baseline, not solved dynamic balance.

`adapt.py` performs frozen-BFM few-shot latent adaptation with isolated HUSKY
rollouts. It never updates the actor or writes joint actions directly:

```bash
CUDA_VISIBLE_DEVICES=6 python 01_bfm0_motion_husky/scripts/adapt.py \
  --base-key move-ego-0-0 --base-index 0 \
  --iterations 2 --population 6 --elite 2 --steps 60 --sigma 0.08 \
  --output /63data1/hwh_data/Skate-bfm/prompts/steer_adapt.npy \
  --report /63data1/hwh_data/Skate-bfm/runs/steer_adapt.json
```

## Search

`search.py` runs every explorer in a fresh process, labels the collected BFM
observations with HUSKY contact-aware push reward, and calls BFM0's native
`reward_wr_inference`. It also saves the best explorer separately from the
inferred latent.

```bash
CUDA_VISIBLE_DEVICES=6 python 01_bfm0_motion_husky/scripts/search.py \
  --device cuda --husky-device cuda:0 --steps 60 \
  --z-key move-ego-180-0.3 --indices 2 \
  --anchor-key move-ego-0-0 --anchor-index 0 \
  --mixes 0.12,0.14,0.16,0.18 \
  --output /63data1/hwh_data/Skate-bfm/prompts/push_narrow_inferred.npy \
  --best-output /63data1/hwh_data/Skate-bfm/prompts/push_narrow_best.npy \
  --report /63data1/hwh_data/Skate-bfm/runs/search_narrow.json
```

## Current Result

This is an honest partial baseline, not solved skateboarding. The safe 190-step
run stays upright (`min root height 0.665 m`) and exercises live hold/drive.
During the first 100 steps, robot-board distance remains below 0.178 m; over
the full run mean board velocity is `-0.129 m/s` and final separation reaches
0.927 m, so transition is correctly blocked. The retained evidence videos and
their interpretation are documented in `results/README.md`.

A diagnostic run with `--trigger-speed -0.5 --trigger-distance 0.35` reaches
`push2steer` and `steer` while remaining upright, proving those BFM-only paths
execute, but board distance reaches 2.151 m. Positive propulsion with retained
board contact and reliable heading control remain unresolved.
