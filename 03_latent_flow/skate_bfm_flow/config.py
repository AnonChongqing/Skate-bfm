from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictModel):
    name: str = "latent_flow_v0"
    seed: int = 42
    device: str = "cuda:0"
    deterministic: bool = False


class PathsConfig(StrictModel):
    project_root: str = "/home/hu_wenhui/workspace/Skate-bfm"
    data_root: str = "/63data1/hwh_data/Skate-bfm"
    bfm_model_dir: str = "/63data1/hwh_data/Skate-bfm/models/bfm0"
    basis_path: str = "/63data1/hwh_data/Skate-bfm/latent_basis/skate_mode_basis_v0.pt"
    run_dir: str = "/home/hu_wenhui/workspace/Skate-bfm/03_latent_flow/results/runs"
    dataset_dir: str = "/63data1/hwh_data/Skate-bfm/datasets/latent_flow"
    checkpoint_dir: str = "/home/hu_wenhui/workspace/Skate-bfm/03_latent_flow/checkpoint"


class ControlConfig(StrictModel):
    physics_hz: int = 200
    bfm_hz: int = 50
    flow_hz: int = 10
    macro_steps: int = 5
    mean_bfm_action: bool = True
    gamma_macro: float = 0.99
    allow_rate_mismatch: bool = False

    @model_validator(mode="after")
    def validate_rates(self) -> "ControlConfig":
        if self.flow_hz > self.bfm_hz:
            raise ValueError("flow_hz cannot exceed bfm_hz")
        if not self.allow_rate_mismatch:
            if self.bfm_hz % self.flow_hz or self.macro_steps != self.bfm_hz // self.flow_hz:
                raise ValueError("macro_steps must equal bfm_hz / flow_hz")
        return self


class EnvConfig(StrictModel):
    task_id: str = "Mjlab-Skater-Flat-Unitree-G1"
    num_envs: int = 1
    command_speed: float | None = 0.7
    command_heading: float | None = 0.4
    command_speed_range: tuple[float, float] | None = None
    command_heading_range: tuple[float, float] | None = None
    domain_randomization: bool = False
    interval_push: bool = True
    reset_noise: float = 0.0
    observation_noise: bool = False
    initial_mode: Literal["push", "steer", "mixed"] = "push"
    steer_reset_fraction: float = Field(default=0.35, ge=0.0, le=1.0)
    steer_initial_speed: float = 0.7
    action_mapping: Literal["reference", "nominal_aligned", "target_position", "raw_shared"] = "reference"
    reference_blend: float = 0.0
    action_gain: float = 1.25
    action_clip: float | None = None
    render_mode: Literal["rgb_array"] | None = None
    quiet: bool = False


class ModeConfig(StrictModel):
    names: list[str] = Field(default_factory=lambda: ["push", "mount", "steer", "dismount", "recover"])
    source: Literal["fixed_fsm", "policy_head"] = "fixed_fsm"
    use_one_hot: bool = True
    recover_root_height: float = 0.55
    recover_board_distance: float = 0.8


class LatentConfig(StrictModel):
    z_dim: int = 256
    flow_dim: int = Field(default=16, gt=0)
    step_size: float = Field(default=0.25, gt=0.0)
    update_type: Literal["tangent_residual", "euclidean_residual", "prototype_residual", "direct_z"] = "tangent_residual"
    basis_type: str = "fixed_mode_basis"
    basis_trainable: bool = False
    project_radius: Literal["sqrt_dim"] | float = "sqrt_dim"
    prototype_paths: dict[str, str] = Field(default_factory=lambda: {
        "push": "/63data1/hwh_data/Skate-bfm/prompts/push_back.npy",
        "mount": "/63data1/hwh_data/Skate-bfm/prompts/steer.npy",
        "steer": "/63data1/hwh_data/Skate-bfm/prompts/steer_adapt.npy",
        "dismount": "/63data1/hwh_data/Skate-bfm/prompts/push.npy",
        "recover": "/63data1/hwh_data/Skate-bfm/prompts/steer_adapt.npy",
    })
    basis_source_paths: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def frozen_basis(self) -> "LatentConfig":
        if self.z_dim != 256:
            raise ValueError("The frozen BFM0 checkpoint requires z_dim=256")
        if self.basis_trainable:
            raise ValueError("Stage 03 v0 requires a frozen latent basis")
        return self


class HuskyPriorConfig(StrictModel):
    motion_dir: str = "/home/hu_wenhui/workspace/Skate-bfm/husky_sim/dataset/skate_push"
    output_path: str = "/63data1/hwh_data/Skate-bfm/priors/husky_push_latents.npy"
    frame_stride: int = Field(default=1, gt=0)
    batch_size: int = Field(default=512, gt=0)


class PreviewConfig(StrictModel):
    enabled: bool = True
    type: Literal["none", "action_29d", "action_23d", "lower_body_12d", "first_action_23d"] = "action_23d"
    detach_bfm: bool = True


class TargetConfig(StrictModel):
    type: Literal["finite_horizon_return", "semi_mdp_td", "sac_td", "td3_td"] = "sac_td"
    aggregation: Literal["min", "mean", "mean_minus_std", "min_minus_disagreement"] = "min"
    gamma_macro: float = Field(default=0.99, gt=0.0, le=1.0)
    bootstrap_on_timeout: bool = True
    entropy_term: bool = True
    uncertainty_beta: float = Field(default=0.5, ge=0.0)


class LossConfig(StrictModel):
    type: Literal["huber", "mse", "mae"] = "huber"
    huber_delta: float = Field(default=1.0, gt=0.0)


class OptimizerConfig(StrictModel):
    type: Literal["adam", "adamw"] = "adamw"
    lr: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    grad_clip: float = Field(default=10.0, gt=0.0)


class QConfig(StrictModel):
    input_profile: Literal["minimal", "candidate", "preview", "full_preview"] = "full_preview"
    state_profile: Literal["privileged", "deployable"] = "privileged"
    twin_independent: bool = True
    architecture: str = "multi_branch"
    activation: Literal["elu", "relu", "silu"] = "elu"
    final_hidden_dims: list[int] = Field(default_factory=lambda: [512, 256, 128])
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    target_tau: float = Field(default=0.005, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def independent_twins(self) -> "QConfig":
        if not self.twin_independent:
            raise ValueError("Q1 and Q2 must be independent")
        if self.preview.type == "first_action_23d":
            self.preview.type = "action_23d"
        if not self.preview.enabled:
            self.preview.type = "none"
        return self


class PolicyConfig(StrictModel):
    architecture: str = "frame_stack_mlp"
    frame_stack: int = Field(default=5, gt=0)
    hidden_dims: list[int] = Field(default_factory=lambda: [512, 256, 128])
    activation: Literal["elu", "relu", "silu"] = "elu"
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    optimizer_lr: float = Field(default=3e-4, gt=0.0)


class SacConfig(StrictModel):
    batch_size: int = Field(default=512, gt=0)
    replay_capacity: int = Field(default=2_000_000, gt=0)
    random_steps: int = Field(default=20_000, ge=0)
    update_after: int = Field(default=5_000, ge=0)
    updates_per_macro_step: float = Field(default=1.0, gt=0.0)
    learn_alpha: bool = True
    initial_alpha: float = Field(default=0.1, gt=0.0)
    target_entropy: float | Literal["auto"] = "auto"
    flow_magnitude: float = Field(default=0.001, ge=0.0)
    flow_smoothness: float = Field(default=0.01, ge=0.0)


class RewardConfig(StrictModel):
    source: str = "husky_existing"
    macro_aggregation: Literal["discounted_sum", "sum", "mean"] = "discounted_sum"
    normalize: bool = False
    clip: float | None = None
    gate_progress_by_retention: bool = True
    contact_loss_distance: float = 0.7
    fall_height: float = 0.45
    husky_weight: float = 1.0
    board_progress_weight: float = 2.0
    heading_progress_weight: float = 0.5
    retention_weight: float = 0.5
    upright_weight: float = 0.1
    fall_penalty_weight: float = 5.0
    illegal_contact_weight: float = 2.0


class ReplayConfig(StrictModel):
    cache_z_candidate: bool = False
    cache_action_preview: bool = False
    sampling: Literal["uniform", "mode_balanced", "outcome_balanced"] = "uniform"


class BranchConfig(StrictModel):
    num_anchors: int = Field(default=5000, gt=0)
    candidates_per_anchor: int = Field(default=16, gt=0)
    warmup_low_steps: int = Field(default=10, ge=0)
    horizon_low_steps: int = Field(default=25, gt=0)
    anchor_stride_macro: int = Field(default=2, gt=0)
    local_std: float = Field(default=0.35, gt=0.0)
    disable_interval_push: bool = True


class TrainConfig(StrictModel):
    dataset_path: str = "/63data1/hwh_data/Skate-bfm/datasets/latent_flow/branches_v0.pt"
    checkpoint_path: str | None = None
    steps: int = Field(default=10000, gt=0)
    batch_size: int = Field(default=512, gt=0)
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    log_interval: int = Field(default=100, gt=0)
    checkpoint_interval: int = Field(default=1000, gt=0)


class BcConfig(StrictModel):
    target_type: Literal["hard_best", "soft_weighted"] = "hard_best"
    temperature: float = Field(default=0.25, gt=0.0)


class CurriculumConfig(StrictModel):
    enabled: bool = False
    ramp_steps: int = Field(default=250000, gt=0)
    speed_start: tuple[float, float] = (0.4, 0.8)
    speed_end: tuple[float, float] = (0.0, 1.5)
    heading_start: tuple[float, float] = (-0.2, 0.2)
    heading_end: tuple[float, float] = (-0.785398, 0.785398)


class EvalConfig(StrictModel):
    episodes: int = Field(default=10, gt=0)
    macro_steps: int = Field(default=60, gt=0)
    deterministic: bool = True
    video: bool = False
    video_dir: str = "/home/hu_wenhui/workspace/Skate-bfm/03_latent_flow/results/runs/eval_videos"


class LoggingConfig(StrictModel):
    tensorboard: bool = True
    jsonl: bool = True
    csv: bool = True
    checkpoint_interval: int = 10000
    eval_interval: int = Field(default=10000, gt=0)
    eval_video: bool = False
    eval_episodes: int = Field(default=1, gt=0)
    eval_suite: Literal["standard", "phases"] = "phases"
    eval_cuda_visible_devices: str | None = None


class DebugConfig(StrictModel):
    anomaly_detection: bool = False
    assert_every_n_steps: int = 1000
    dump_bad_batch: bool = True
    max_abs_q_warning: float = 10000.0


class Stage03Config(StrictModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    mode: ModeConfig = Field(default_factory=ModeConfig)
    latent: LatentConfig = Field(default_factory=LatentConfig)
    husky_prior: HuskyPriorConfig = Field(default_factory=HuskyPriorConfig)
    q: QConfig = Field(default_factory=QConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    sac: SacConfig = Field(default_factory=SacConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    branch: BranchConfig = Field(default_factory=BranchConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    bc: BcConfig = Field(default_factory=BcConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = set() if seen is None else seen
    path = path.expanduser().resolve()
    if path in seen:
        raise ValueError(f"Recursive config base: {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text()) or {}
    base_ref = raw.pop("_base", None)
    if base_ref is None:
        return raw
    base_path = (path.parent / base_ref).resolve()
    return _merge(_read_yaml(base_path, seen), raw)


def _parse_override(value: str) -> Any:
    return yaml.safe_load(value)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got {override!r}")
        dotted, value = override.split("=", 1)
        keys = dotted.split(".")
        cursor = result
        for key in keys[:-1]:
            child = cursor.get(key)
            if child is None:
                child = {}
                cursor[key] = child
            if not isinstance(child, dict):
                raise ValueError(f"Cannot descend into non-mapping override path {dotted!r}")
            cursor = child
        cursor[keys[-1]] = _parse_override(value)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> Stage03Config:
    return Stage03Config.model_validate(apply_overrides(_read_yaml(Path(path)), overrides or []))


def save_resolved_config(cfg: Stage03Config, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
