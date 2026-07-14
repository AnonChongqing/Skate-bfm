# Baseline Run

The safe default alternates a BFM0 zero-speed hold prompt with short projected
locomotion prompt pulses. Every 29D action comes from the frozen BFM0 actor.

The representative safe run is `../results/02_safe_reactive.mp4`:

- minimum root height: 0.665 m
- mean board x velocity: -0.129 m/s
- maximum robot-board distance in the first 100 steps: 0.178 m
- final robot-board distance: 0.927 m
- board contact rate: 38.4%
- controller states: `push_hold`, `push_drive`

The positive-speed transition gate is not reached. This run demonstrates
stable reactive execution, not successful forward skateboarding. See
`../results/README.md` for the transition and kick-away comparisons.
