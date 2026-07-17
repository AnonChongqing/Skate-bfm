from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch


SIM_FIELDS = (
    "qpos", "qvel", "act", "ctrl", "qacc_warmstart", "time", "mocap_pos",
    "mocap_quat", "eq_active", "plugin_state", "qfrc_applied", "xfrc_applied",
)
ENV_TENSORS = (
    "episode_length_buf", "phase_length_buf", "last_contacts", "last_wheel_contacts",
    "last_contacts_b", "last_contacts_g", "contact_phase", "last_contact_phase", "still",
    "reset_buf", "reset_terminated", "reset_time_outs", "just_entered_push2steer",
    "just_entered_steer2push", "just_exited_push2steer", "just_exited_steer2push",
)


def _direct_tensors(obj: Any) -> dict[str, torch.Tensor]:
    if obj is None or not hasattr(obj, "__dict__"):
        return {}
    return {name: value.clone() for name, value in vars(obj).items() if isinstance(value, torch.Tensor)}


def _restore_direct(obj: Any, values: dict[str, torch.Tensor]) -> None:
    for name, value in values.items():
        current = getattr(obj, name, None)
        if isinstance(current, torch.Tensor):
            current.copy_(value)


@dataclass
class HuskyEnvSnapshot:
    sim: dict[str, torch.Tensor]
    env: dict[str, torch.Tensor]
    body_buffers: dict[str, torch.Tensor]
    action_manager: dict[str, torch.Tensor]
    action_terms: dict[str, dict[str, torch.Tensor]]
    command_terms: dict[str, dict[str, torch.Tensor]]
    sensors: dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]
    reward_terms: dict[str, list[dict[str, torch.Tensor]]]
    wrapper: dict[str, torch.Tensor]
    counters: dict[str, int]
    extra: dict[str, Any]

    @classmethod
    def capture(cls, wrapper, extra: dict[str, Any] | None = None) -> "HuskyEnvSnapshot":
        env = wrapper.husky_env
        sim_state: dict[str, torch.Tensor] = {}
        for name in SIM_FIELDS:
            try:
                value = getattr(env.sim.data, name)
                if hasattr(value, "clone"):
                    sim_state[name] = value.clone()
            except (AttributeError, RuntimeError):
                continue
        env_state = {name: getattr(env, name).clone() for name in ENV_TENSORS if isinstance(getattr(env, name, None), torch.Tensor)}
        action_terms = {name: _direct_tensors(term) for name, term in env.action_manager._terms.items()}
        command_terms = {name: _direct_tensors(term) for name, term in env.command_manager._terms.items()}
        sensors = {
            name: (_direct_tensors(sensor), _direct_tensors(getattr(sensor, "data", None)))
            for name, sensor in env.scene.sensors.items()
        }
        reward_terms = {}
        for manager_name in ("push_reward_manager", "steer_reward_manager", "transition_reward_manager", "reg_reward_manager"):
            manager = getattr(env, manager_name)
            reward_terms[manager_name] = [_direct_tensors(cfg.func) for cfg in manager._term_cfgs]
        wrapper_state = {
            name: getattr(wrapper, name).clone()
            for name in (
                "_last_bfm0_action", "_history_action", "_history_ang_vel",
                "_history_dof_pos", "_history_dof_vel", "_history_projected_gravity",
            )
        }
        wrapper_state.update({f"obs/{name}": value.clone() for name, value in wrapper._obs.items()})
        return cls(
            sim=sim_state,
            env=env_state,
            body_buffers={name: value.clone() for name, value in env.body_bezier_buffers.items()},
            action_manager=_direct_tensors(env.action_manager),
            action_terms=action_terms,
            command_terms=command_terms,
            sensors=sensors,
            reward_terms=reward_terms,
            wrapper=wrapper_state,
            counters={"common_step_counter": env.common_step_counter, "_sim_step_counter": env._sim_step_counter},
            extra=copy.deepcopy(extra or {}),
        )

    def restore(self, wrapper) -> dict[str, Any]:
        env = wrapper.husky_env
        for name, value in self.sim.items():
            getattr(env.sim.data, name).copy_(value)
        for name, value in self.env.items():
            getattr(env, name).copy_(value)
        for name, value in self.body_buffers.items():
            env.body_bezier_buffers[name].copy_(value)
        _restore_direct(env.action_manager, self.action_manager)
        for name, values in self.action_terms.items():
            _restore_direct(env.action_manager._terms[name], values)
        for name, values in self.command_terms.items():
            _restore_direct(env.command_manager._terms[name], values)
        for name, (sensor_values, data_values) in self.sensors.items():
            _restore_direct(env.scene.sensors[name], sensor_values)
            _restore_direct(env.scene.sensors[name].data, data_values)
        for manager_name, states in self.reward_terms.items():
            manager = getattr(env, manager_name)
            for cfg, values in zip(manager._term_cfgs, states, strict=True):
                _restore_direct(cfg.func, values)
        for name, value in self.wrapper.items():
            if name.startswith("obs/"):
                continue
            getattr(wrapper, name).copy_(value)
        wrapper._obs = {name.removeprefix("obs/"): value.clone() for name, value in self.wrapper.items() if name.startswith("obs/")}
        for name, value in self.counters.items():
            setattr(env, name, value)
        env.sim.forward()
        env.scene.update(dt=0.0)
        return copy.deepcopy(self.extra)
