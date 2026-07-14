"""Unitree G1 skateboarding environment configurations."""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab_husky.asset_zoo.robots.skateboard.g1_skater_constants import (
  G1_23Dof_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
  get_skateboard_cfg
)
from mjlab_husky.envs import G1SkaterManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import HfPyramidSlopedTerrainCfg, TerrainGeneratorCfg
from mjlab_husky.tasks.skater import mdp
from mjlab_husky.tasks.skater.mdp import SkateUniformVelocityCommandCfg
from mjlab_husky.tasks.skater.skater_env_cfg import make_g1_skater_env_cfg


@dataclass(kw_only=True)
class MaterialSlopeTerrainCfg(HfPyramidSlopedTerrainCfg):
  """Pyramid-slope terrain with explicit friction and visual color."""

  friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
  rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ):
    output = super().function(difficulty, spec, rng)
    for geometry in output.geometries:
      if geometry.geom is None:
        continue
      geometry.geom.friction = self.friction
      geometry.geom.rgba = self.rgba
      geometry.color = self.rgba
    return output


def unitree_g1_skater_env_cfg(play: bool = False) -> G1SkaterManagerBasedRlEnvCfg:
  cfg = make_g1_skater_env_cfg()
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 55

  cfg.scene.entities = {"robot": get_g1_23dof_robot_cfg(), "skateboard": get_skateboard_cfg()}
  
  #########################################################
  ##### terrain #####
  #########################################################
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  #########################################################
  ##### contact sensors #####
  #########################################################
  left_feet_ground_cfg = ContactSensorCfg(
    name="left_feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  right_feet_ground_cfg = ContactSensorCfg(
    name="right_feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  left_feet_board_cfg = ContactSensorCfg(
    name="left_feet_board_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="skateboard_marker_collision", entity="skateboard"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  right_feet_board_cfg = ContactSensorCfg(
    name="right_feet_board_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="skateboard_deck_collision", entity="skateboard"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  robot_collision_cfg = ContactSensorCfg(
    name="robot_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  skateboard_collision_cfg = ContactSensorCfg(
    name="skateboard_collision",
    primary=ContactMatch(mode="geom", pattern=r".*_wheel_collision$", entity="skateboard"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found","force"),
    reduce="none",
    num_slots=1,
  )

  illegal_contact_cfg = ContactSensorCfg(
    name="illegal_contact",
    primary=ContactMatch(mode="geom", pattern=r".*_shin_collision|.*_linkage_brace_collision|.*_shoulder_yaw_collision|.*_elbow_yaw_collision|.*_wrist_collision|.*_hand_collision|pelvis_collision$", entity="robot"),
    # secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg.scene.sensors = (robot_collision_cfg,skateboard_collision_cfg,left_feet_ground_cfg, right_feet_ground_cfg,left_feet_board_cfg, right_feet_board_cfg,illegal_contact_cfg)
  

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_23Dof_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  skate_cmd = cfg.commands["skate"]
  assert isinstance(skate_cmd, SkateUniformVelocityCommandCfg)
  skate_cmd.viz.z_offset = 1.15


  cfg.beizer_names = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    ]

  cfg.slerp_names = cfg.beizer_names
  cfg.phase_ratios = [0.0, 0.4, 0.5, 0.95, 1.0]
  cfg.steer_init_pos = [
    -0.15, 0.1, 0.05, 0.6, -0.42, 0.0, 
    -0.15, -0.1, 0.05, 0.6, -0.42, 0.0,
    0, 0, 0.1, 
    0, 0.55, -0.25, 0.55,
    0, -0.55, -0.25, 0.55
    ]

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(60.0)
    cfg.eval_mode = True
    cfg.observations["policy"].enable_corruption = False
    cfg.terminations = {
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    cfg.events.pop("push_robot", None)
    # cfg.commands["skate"].ranges.lin_vel_x = (1.0, 1.0)  # pyright: ignore[reportAttributeAccessIssue]
    # cfg.commands["skate"].ranges.heading = (0.7, 0.7)  # pyright: ignore[reportAttributeAccessIssue]
  return cfg


def unitree_g1_skater_slope_materials_env_cfg(
  play: bool = False,
) -> G1SkaterManagerBasedRlEnvCfg:
  cfg = unitree_g1_skater_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.env_spacing = None
  cfg.scene.terrain.max_init_terrain_level = None
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=1.0,
    border_height=0.25,
    num_rows=2,
    num_cols=4,
    color_scheme="height",
    difficulty_range=(0.25, 0.75),
    add_lights=True,
    sub_terrains={
      "rubber_high_grip": MaterialSlopeTerrainCfg(
        proportion=1.0,
        slope_range=(0.05, 0.18),
        platform_width=2.0,
        border_width=0.3,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        friction=(1.8, 0.02, 0.002),
        rgba=(0.18, 0.42, 0.22, 1.0),
      ),
      "concrete": MaterialSlopeTerrainCfg(
        proportion=1.0,
        slope_range=(0.05, 0.18),
        platform_width=2.0,
        border_width=0.3,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        friction=(1.0, 0.01, 0.001),
        rgba=(0.45, 0.47, 0.50, 1.0),
      ),
      "wood": MaterialSlopeTerrainCfg(
        proportion=1.0,
        slope_range=(0.05, 0.18),
        platform_width=2.0,
        border_width=0.3,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        friction=(0.65, 0.006, 0.0006),
        rgba=(0.58, 0.39, 0.20, 1.0),
      ),
      "wet_low_grip": MaterialSlopeTerrainCfg(
        proportion=1.0,
        slope_range=(0.05, 0.18),
        platform_width=2.0,
        border_width=0.3,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        friction=(0.28, 0.003, 0.0003),
        rgba=(0.16, 0.30, 0.55, 1.0),
      ),
    },
  )

  cfg.scene.extent = 16.0
  cfg.sim.nconmax = 192
  cfg.sim.njmax = 800
  cfg.sim.contact_sensor_maxmatch = 128

  # Keep evaluation on one environment unless explicitly overridden by CLI.
  if play:
    cfg.scene.num_envs = 1

  return cfg
