from __future__ import annotations

import os

from mjlab.viewer import ViserPlayViewer


class ViewerEnv:
    """Expose the BFM0 wrapper through mjlab's viewer protocol."""

    def __init__(self, env) -> None:
        self.env = env
        self.num_envs = env.husky_env.num_envs
        self.device = env.device

    @property
    def cfg(self):
        return self.env.husky_env.cfg

    @property
    def unwrapped(self):
        return self.env.husky_env

    def get_observations(self):
        return self.env.observation

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()


def run_viser(env, policy, port: int = 8080, steps: int | None = None) -> None:
    os.environ["_VISER_PORT_OVERRIDE"] = str(port)
    viewer_env = ViewerEnv(env)
    viewer = ViserPlayViewer(
        viewer_env,
        policy,
        frame_rate=1.0 / env.husky_env.step_dt,
    )
    viewer.run(num_steps=steps)
