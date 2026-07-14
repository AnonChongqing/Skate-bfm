# Transition Goal

Goal tracking belongs to Stage 01 because it is the transition objective between
the push and steer phases, not a separate experiment stage.

The HUSKY simulator captures the current key-body poses at transition entry and
plans toward the next phase's canonical pose. It uses a Bezier position path and
Slerp orientation path. `score/goal.py` evaluates both targets only while either
transition phase is active.

This is HUSKY trajectory goal tracking. Turning the same target into a native
BFM0 goal latent would additionally require converting target key-body poses to
the complete BFM0 goal observation, including its privileged state.
