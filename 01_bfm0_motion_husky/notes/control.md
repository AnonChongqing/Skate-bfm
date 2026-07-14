# Phase Control

`--control fixed` keeps one BFM0 latent for the whole episode. It is retained as
the original open-loop task-prompt baseline. The BFM0 actor itself is still
proprioceptive and computes a new action from the current robot observation.

`--control phase` reads HUSKY state every control step:

```text
push_hold     -> BFM0 zero-speed reward latent
push_drive    -> short BFM0 locomotion-reward pulse
push2steer    -> reward interpolation + BFM0 tracking latent
steer         -> zero-speed reward + signed rotate reward from live heading error
steer2push    -> reverse tracking blend toward the push reward
```

Push itself is feedback-driven. The controller enters `push_drive` only when
the right foot is on the board, the left foot is on the ground, the robot is
within 0.35 m of the board, and board speed is below the command. Drive is
limited to eight steps, then returns to `push_hold`. The two prompts are
projected BFM latent mixtures; every joint action is decoded by the frozen
BFM0 actor.

Push-to-steer is event-driven rather than a fixed one-shot schedule. The
default trigger requires phase >= 0.12, board speed >= 0.2 m/s, board distance
<= 0.25 m, at least one board contact, and root height >= 0.5 m. The phase 0.35
fallback uses the same physical safety conditions; elapsed time cannot bypass
the speed or contact gates.

At transition entry, the controller captures the current 23-DoF pose and builds
a smooth joint trajectory toward HUSKY's steering pose. The adapter maps this
trajectory to batched 29-DoF BFM observations and BFM0's native
`tracking_inference` produces one latent per frame. A tracking component is
blended into an 18-step push-to-steer reward interpolation; goal tracking is
never active outside transition.

During steering, live board heading error selects BFM0's positive or negative
`rotate-z` reward and scales its mixture with a zero-speed reward. No HUSKY
policy, fixed joint target, or direct stabilization action is used.

This is closed-loop prompt switching. It approximates HUSKY's key-body
Bezier/Slerp transition with a joint-space trajectory because an IK conversion
to BFM0's full goal observation is not available yet.

The current rotate prompt does not produce reliable heading convergence. A
temporary recovery state that switched back to the push prompt after board
separation was tested and rejected because it increased board distance and
reduced stability.

The safe default does not enter transition when board speed is negative. Use
`--trigger-speed -0.5 --trigger-distance 0.35` only as a diagnostic to verify
the tracking and steer code paths; it is not a successful forward-skating
configuration.
