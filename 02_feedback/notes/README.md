# Stage 02: Feedback

Goal: use BFM0 output as a high-level body motion proposal and add
skateboard-aware feedback for lower-body control.

Candidate feedback signals:

- foot-board contact state
- skateboard roll and yaw error
- heading error
- feet marker distance
- pelvis/skateboard relative pose

This stage starts after Stage 01 has usable BFM0 rollouts and transition goals.
