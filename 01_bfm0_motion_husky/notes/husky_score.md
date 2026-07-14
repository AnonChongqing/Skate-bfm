# HUSKY Scores

These terms score BFM0 rollouts in the HUSKY skateboard environment. The code
calls the official HUSKY reward functions instead of maintaining approximate
copies.

Push terms and weights:

```text
3.0 velocity + 1.0 yaw + 3.0 foot air time + 0.5 ankle parallel
```

The HUSKY AMP discriminator style term is excluded because this baseline runs a
BFM0 policy and does not train or load HUSKY's pushing discriminator.

Steer terms and weights:

```text
3.0 contact + 1.5 pose + 1.0 foot marker + 5.0 heading + 4.0 tilt
```

Transition goal:

```text
transition_goal = 0.5 * key-body-position + 0.5 * key-body-orientation
```

The transition target is generated online from the terminal pose of the current
phase to the canonical pose of the next phase. Position follows a Bezier curve;
orientation follows Slerp. Goal rewards are active only during push-to-steer or
steer-to-push transitions. The transition contact penalty remains separate from
goal tracking.
