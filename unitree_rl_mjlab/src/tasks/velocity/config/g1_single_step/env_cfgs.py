"""Unitree G1 single-step-up environment configuration.

Task: robot spawns on flat ground directly in front of a single step,
walks straight forward and steps up onto it.

- Single-step terrain: one full-width step on the far half of the tile;
  the near half is flat approach ground. Spawn origin sits on the flat
  ground ~2 m before the step edge, facing +x (toward the step).
- Step height is curriculum-driven (difficulty rows) and additionally
  jittered per tile for domain randomization.
- Blind actor (no height scan, 5-frame proprioception history) and
  asymmetric critic, same as the stairs blind tasks.
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

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1_stairs_up_blind.env_cfgs import (
  BLIND_ACTOR_HISTORY_LENGTH,
)

_STEP_BLUE = (0.20, 0.45, 0.95)


@dataclass(kw_only=True)
class SingleStepUpTerrainCfg(SubTerrainCfg):
  """A single full-width step; flat approach ground in front of it."""

  step_height_range: tuple[float, float] = (0.05, 0.23)
  """Min and max step height in meters, interpolated by difficulty."""
  step_depth_fraction: float = 0.5
  """Fraction of the tile (along x) occupied by the raised step."""
  height_jitter: float = 0.1
  """Per-tile relative jitter of the step height (domain randomization)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    # Curriculum height with per-tile jitter for domain randomization.
    h_min, h_max = self.step_height_range
    step_height = h_min + difficulty * (h_max - h_min)
    step_height *= 1.0 + rng.uniform(-self.height_jitter, self.height_jitter)
    step_height = float(np.clip(step_height, 0.02, 0.30))

    # Flat ground covering the whole tile at z=0.
    ground_box = make_plane(body, self.size, 0.0, center_zero=False)[0]
    geometries = [
      TerrainGeometry(geom=ground_box, color=(0.5, 0.5, 0.5, 1.0))
    ]

    # Single step: raised box on the far half of the tile, full width.
    step_x0 = self.size[0] * self.step_depth_fraction
    step_len = self.size[0] - step_x0
    step_box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(step_len / 2.0, self.size[1] / 2.0, step_height / 2.0),
      pos=(
        step_x0 + step_len / 2.0,
        self.size[1] / 2.0,
        step_height / 2.0,
      ),
    )
    step_rgba = brand_ramp(_STEP_BLUE, difficulty)
    geometries.append(TerrainGeometry(geom=step_box, color=step_rgba))

    # Spawn origin: centered on the flat approach ground, facing +x.
    origin = np.array([step_x0 / 2.0, self.size[1] / 2.0, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


def single_step_up_terrains_cfg() -> TerrainGeneratorCfg:
  """Single-step terrain with height curriculum + jitter randomization."""
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    sub_terrains={
      "single_step": SingleStepUpTerrainCfg(
        proportion=1.0,
        step_height_range=(0.05, 0.23),
        step_depth_fraction=0.5,
        height_jitter=0.1,
      ),
    },
    add_lights=True,
  )


def unitree_g1_single_step_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 single-step-up velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Replace terrain with single-step curriculum (keeps rough sim/contact setup).
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = single_step_up_terrains_cfg()
  # Flat ground + step edge produce more contacts than the pyramid tiles the
  # rough cfg was sized for; let MuJoCo auto-size the contact buffer.
  cfg.sim.nconmax = None
  if play:
    gen = cfg.scene.terrain.terrain_generator
    gen.curriculum = False
    gen.num_cols = 5
    gen.num_rows = 5
    gen.border_width = 10.0
    # Show mostly full-height steps when playing.
    gen.difficulty_range = (0.8, 1.0)
    # Blind policy doesn't use height scan; hide debug rays in play view.
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        sensor.debug_vis = False

  # Blind actor: drop height scan from the policy observation.
  del cfg.observations["actor"].terms["height_scan"]
  # 5-frame proprioceptive history, flattened oldest -> newest.
  cfg.observations["actor"].history_length = BLIND_ACTOR_HISTORY_LENGTH

  # Straight-ahead commands: forward only, minimal lateral/yaw, no heading
  # control (robot always faces the step).
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None  # Must be unset when heading_command=False.
  twist_cmd.ranges.lin_vel_x = (0.2, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
  twist_cmd.ranges.ang_vel_z = (-0.2, 0.2)

  # Spawn in front of the step, always facing it (+x): no yaw randomization.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.3, 0.3),
    "y": (-0.5, 0.5),
    "z": (0.0, 0.0),
    "yaw": (0.0, 0.0),
  }

  # Fixed command ranges for the whole run (no velocity curriculum).
  cfg.curriculum.pop("command_vel", None)

  if play:
    twist_cmd.ranges.lin_vel_x = (0.4, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)  # 修改范围避免 viser GUI 验证失败
    twist_cmd.ranges.ang_vel_z = (-0.1, 0.1)

  return cfg
