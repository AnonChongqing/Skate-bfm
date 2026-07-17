from __future__ import annotations

import torch

from ..schemas import FeatureBatch, QInputBatch


PROFILE_BRANCHES = {
    "minimal": ("robot", "board", "contact", "goal_mode", "z_current", "flow"),
    "candidate": ("robot", "board", "contact", "goal_mode", "z_current", "z_candidate", "flow"),
    "preview": ("robot", "board", "contact", "goal_mode", "z_current", "flow", "preview"),
    "full_preview": ("robot", "board", "contact", "goal_mode", "z_current", "z_candidate", "flow", "preview"),
}


class QInputBuilder:
    def __init__(self, profile: str, state_profile: str = "privileged") -> None:
        if profile not in PROFILE_BRANCHES:
            raise ValueError(f"Unknown Q input profile: {profile}")
        self.profile = profile
        if state_profile not in {"privileged", "deployable"}:
            raise ValueError(f"Unknown Q state profile: {state_profile}")
        self.state_profile = state_profile

    def build(
        self,
        features: FeatureBatch,
        z_current: torch.Tensor,
        flow: torch.Tensor,
        previous_flow: torch.Tensor,
        z_candidate: torch.Tensor | None = None,
        latent_stats: torch.Tensor | None = None,
        preview: torch.Tensor | None = None,
    ) -> QInputBatch:
        board = features.critic_board if self.state_profile == "privileged" else features.critic_board[:, :18]
        contact = features.critic_contact if self.state_profile == "privileged" else features.critic_contact[:, :14]
        goal_mode = features.critic_goal_mode if self.state_profile == "privileged" else features.critic_goal_mode[:, :10]
        source = {
            "robot": features.critic_robot,
            "board": board,
            "contact": contact,
            "goal_mode": goal_mode,
            "z_current": z_current,
            "z_candidate": z_candidate,
            "preview": preview,
        }
        flow_branch = [flow]
        if self.profile == "full_preview":
            flow_branch.append(previous_flow)
            flow_branch.append(latent_stats if latent_stats is not None else self._missing("latent_stats"))
        source["flow"] = torch.cat(flow_branch, dim=-1)
        branches: dict[str, torch.Tensor] = {}
        for name in PROFILE_BRANCHES[self.profile]:
            value = source[name]
            if value is None:
                self._missing(name)
            branches[name] = value
        batch_size = next(iter(branches.values())).shape[0]
        if any(value.shape[0] != batch_size for value in branches.values()):
            raise ValueError("Q branch batch dimensions differ")
        return QInputBatch(branch_tensors=branches, batch_size=batch_size)

    @staticmethod
    def _missing(name: str):
        raise ValueError(f"Q profile requires {name}")

    @staticmethod
    def branch_dims(batch: QInputBatch) -> dict[str, int]:
        return {name: value.shape[-1] for name, value in batch.branch_tensors.items()}
