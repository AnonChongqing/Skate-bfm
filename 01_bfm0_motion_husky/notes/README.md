# Stage 01 Notes

The goal is not to train a new HUSKY policy in this stage. The goal is to run
BFM0-generated motion inside the HUSKY skateboard simulator and score that
motion with HUSKY-style task terms.

Data flow:

```text
BFM0 obs + z -> BFM0 29D action -> 23D HUSKY action -> HUSKY simulation
                                     -> push/steer reward + transition goal score
```

Action mapping:

```text
bfm_target = q_reference + bfm_action * bfm_action_scale * bfm_action_rescale * gain
husky_action = (bfm_target - husky_default) / husky_action_scale
```

The selected baseline uses `q_reference=bfm_default`, gain `1.25`, and no
extra action clipping. This keeps BFM observation and action coordinates
consistent while HUSKY's actuator model enforces the physical dynamics.

Known mismatch:

- BFM0 has six wrist actions; HUSKY-23DoF has fixed wrists.
- BFM0 actor input does not explicitly observe skateboard state.
- Push and steer use the phase-specific terms and weights from HUSKY.
- Transition uses HUSKY's online Bezier/Slerp key-body goal and is zero outside
  the two transition phases.
- Push uses a selected BFM0 reward latent. Steering updates its reward mixture
  online from live skateboard heading error.
