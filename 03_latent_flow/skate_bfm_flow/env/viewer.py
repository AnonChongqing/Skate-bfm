from __future__ import annotations

import os

from mjlab.viewer import ViserPlayViewer


class MacroViewerEnv:
    def __init__(self, env) -> None:
        self.env = env
        self.num_envs = env.low_env.husky_env.num_envs
        self.device = env.low_env.device

    @property
    def cfg(self):
        return self.env.low_env.husky_env.cfg

    @property
    def unwrapped(self):
        return self.env.low_env.husky_env

    def get_observations(self):
        return self.env._stacked_actor_obs()

    def reset(self):
        return self.env.reset()

    def step(self, flow):
        return self.env.step(flow)

    def close(self):
        pass


def run_viser(env, policy, port: int = 8080, steps: int | None = None) -> None:
    os.environ["_VISER_PORT_OVERRIDE"] = str(port)

    def policy_fn(obs):
        return policy.sample(obs, deterministic=True).mean_action

    ViserPlayViewer(MacroViewerEnv(env), policy_fn, frame_rate=env.cfg.control.flow_hz).run(num_steps=steps)
