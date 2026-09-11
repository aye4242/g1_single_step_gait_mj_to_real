"""Unitree G1 single-step-down environment configuration.

Task: robot spawns on flat ground directly in front of a single step DOWN,
walks straight forward and steps down onto lower ground.

- Single-step terrain: one full-width step DOWN on the far half of the tile;
  the near half is flat approach ground. Spawn origin sits on the flat
  ground ~2 m before the step edge, facing +x (toward the step down).
- Step depth (negative height) is curriculum-driven (difficulty rows) and
  additionally jittered per tile for domain randomization.
- Blind actor (no height scan, 5-frame proprioception history) and
  asymmetric critic, same as the stairs blind tasks.

作者：浮浮酱 (猫娘工程师) ฅ'ω'ฅ
日期：2026-09-09
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.color import brand_ramp

from mjlab.managers.reward_manager import RewardTermCfg

import src.tasks.velocity.mdp as task_mdp
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1_stairs_up_blind.env_cfgs import (
  BLIND_ACTOR_HISTORY_LENGTH,
)

_STEP_RED = (0.95, 0.25, 0.20)  # 下楼梯用红色区分喵～


@dataclass(kw_only=True)
class SingleStepDownTerrainCfg(SubTerrainCfg):
  """A single full-width step DOWN; flat approach ground in front of it."""

  step_depth_range: tuple[float, float] = (0.05, 0.23)
  """Min and max step depth (positive values) in meters, interpolated by difficulty.
  Actual step height will be NEGATIVE (下沉的)."""
  step_length_fraction: float = 0.5
  """Fraction of the tile (along x) occupied by the lower ground."""
  depth_jitter: float = 0.1
  """Per-tile relative jitter of the step depth (domain randomization)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    # Curriculum depth with per-tile jitter for domain randomization.
    d_min, d_max = self.step_depth_range
    step_depth = d_min + difficulty * (d_max - d_min)
    step_depth *= 1.0 + rng.uniform(-self.depth_jitter, self.depth_jitter)
    step_depth = float(np.clip(step_depth, 0.02, 0.30))

    step_x0 = self.size[0] * self.step_length_fraction

    # Strategy: mirror of the UP terrain.
    #   UP:   full base at z=0, raised block on far half (top at +step_height).
    #   DOWN: full base at z=-step_depth, raised block on near half (top at z=0).
    # This way the far half is exposed at z=-step_depth and the near half is z=0.

    # Full-tile base at z = -step_depth (lower level floor).
    base_plane = make_plane(body, self.size, -step_depth, center_zero=False)[0]
    geometries = [
      TerrainGeometry(geom=base_plane, color=(0.5, 0.5, 0.5, 1.0))
    ]

    # Near-half raised platform: top face at z=0 (approach ground).
    # half_height = step_depth/2, center_z = -step_depth/2  =>  top = 0
    near_platform = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(step_x0 / 2.0, self.size[1] / 2.0, step_depth / 2.0),
      pos=(step_x0 / 2.0, self.size[1] / 2.0, -step_depth / 2.0),
    )
    near_rgba = brand_ramp(_STEP_RED, difficulty)
    geometries.append(TerrainGeometry(geom=near_platform, color=near_rgba))

    # Spawn on the near approach platform at z=0.
    origin = np.array([step_x0 / 2.0, self.size[1] / 2.0, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


def single_step_down_terrains_cfg() -> TerrainGeneratorCfg:
  """Single-step-down terrain with depth curriculum + jitter randomization."""
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    sub_terrains={
      "single_step_down": SingleStepDownTerrainCfg(
        proportion=1.0,
        step_depth_range=(0.05, 0.23),
        step_length_fraction=0.5,
        depth_jitter=0.1,
      ),
    },
    add_lights=True,
  )


def unitree_g1_single_step_down_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 single-step-down velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Replace terrain with single-step-down curriculum.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = single_step_down_terrains_cfg()
  # Flat ground + step edge produce more contacts.
  cfg.sim.nconmax = None
  # 对齐上楼梯的并行环境数，保证每个 iteration 数据量一致
  cfg.scene.num_envs = 8192
  if play:
    gen = cfg.scene.terrain.terrain_generator
    gen.curriculum = False
    gen.num_cols = 5
    gen.num_rows = 5
    gen.border_width = 10.0
    # Show mostly full-depth steps when playing.
    gen.difficulty_range = (0.8, 1.0)
    # Blind policy doesn't use height scan; hide debug rays in play view.
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        sensor.debug_vis = False

  # Blind actor: drop height scan from the policy observation.
  del cfg.observations["actor"].terms["height_scan"]
  # 5-frame proprioceptive history, flattened oldest -> newest.
  cfg.observations["actor"].history_length = BLIND_ACTOR_HISTORY_LENGTH

  # Straight-ahead commands: forward only, minimal lateral/yaw.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.ranges.lin_vel_x = (0.2, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
  twist_cmd.ranges.ang_vel_z = (-0.2, 0.2)

  # Spawn in front of the step down, always facing it (+x): no yaw randomization.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.3, 0.3),
    "y": (-0.5, 0.5),
    "z": (0.0, 0.0),
    "yaw": (0.0, 0.0),
  }

  # Fixed command ranges for the whole run.
  cfg.curriculum.pop("command_vel", None)
  if not play and "terrain_levels" in cfg.curriculum:
    # 下台阶布局：spawn在x=2m，台阶在x=4m（tile中心）。默认 move_up=4m 需走到tile末端，
    # 机器人实际速度不足，curriculum永远卡住。改为2.5m：越过台阶后再走0.5m即晋级。
    cfg.curriculum["terrain_levels"].params["move_up_distance"] = 2.5

  # ------------------------------------------------------------------ #
  # Reward overrides for down-step task — 完全对齐 G1DWAQ_Lab 策略喵～  #
  # ------------------------------------------------------------------ #

  # --- 删除对下台阶真正有害的 reward 项 ---
  del cfg.rewards["soft_landing"]      # 惩罚落地冲击 → 下台阶必然触发，有害
  del cfg.rewards["foot_clearance"]    # 惩罚抬腿不足 → 下台阶需向下踏而非向上抬

  # --- 步态塑形 reward 保留但降权，防止步态崩坏 ---
  cfg.rewards["foot_gait"].weight = 0.2         # 原0.5 → 0.2，保留基本交替步态
  cfg.rewards["foot_slip"].weight = -0.1        # 原-0.25 → -0.1，适当放松
  cfg.rewards["pose"].weight = 0.3              # 原1.0 → 0.3，保留手臂/姿态参考
  cfg.rewards["stand_still"].weight = -0.3      # 原-1.0 → -0.3，适当放松
  del cfg.rewards["angular_momentum"]           # 角动量惩罚对下台阶影响小，删除

  # --- 对齐 G1DWAQ 的 reward 权重 ---
  cfg.rewards["track_linear_velocity"].weight = 2.0   # 1.0 → 2.0
  cfg.rewards["track_angular_velocity"].weight = 2.0  # 1.0 → 2.0
  cfg.rewards["action_rate_l2"].weight = -0.01        # -0.05 → -0.01
  cfg.rewards["joint_pos_limits"].weight = -2.0       # -10.0 → -2.0
  cfg.rewards["body_orientation_l2"].weight = -0.3    # -1.0 → -0.3（允许前倾）
  cfg.rewards["is_terminated"].weight = -10.0         # -200 → -10（防 reward hacking）
  # 生存奖励：匹配 G1DWAQ alive=+0.15，是防止早期 collapse 的关键
  cfg.rewards["alive"] = RewardTermCfg(func=task_mdp.alive, weight=0.15)

  if play:
    twist_cmd.ranges.lin_vel_x = (0.4, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
    twist_cmd.ranges.ang_vel_z = (-0.1, 0.1)

  return cfg
