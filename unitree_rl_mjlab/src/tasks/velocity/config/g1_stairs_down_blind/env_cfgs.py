"""Unitree G1 blind stair descent environment configuration.

Task design migrated from G1DWAQ_Lab:
- Blind actor: no height scan, proprioception only, 5-frame history.
- Asymmetric critic: keeps privileged observations (base lin vel,
  height scan, foot states).
- Stairs-down (inverted pyramid) terrain with difficulty curriculum:
  step height interpolated 0.0 -> 0.23 m, mixing 0.30 m and 0.34 m
  step widths, plus flat tiles for approach/recovery.
- More conservative command ranges than ascent.
"""

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1_stairs_up_blind.env_cfgs import (
  BLIND_ACTOR_HISTORY_LENGTH,
)


def stairs_down_terrains_cfg() -> TerrainGeneratorCfg:
  """Stairs-down focused terrain curriculum (G1DWAQ ROUGH_TERRAINS_CFG style)."""
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    sub_terrains={
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.2),
      "stairs_down_narrow": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
        proportion=0.4,
        step_height_range=(0.0, 0.23),
        step_width=0.30,
        platform_width=3.0,
        border_width=1.0,
      ),
      "stairs_down_wide": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
        proportion=0.4,
        step_height_range=(0.0, 0.23),
        step_width=0.34,
        platform_width=3.0,
        border_width=1.0,
      ),
    },
    add_lights=True,
  )


def unitree_g1_stairs_down_blind_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 blind stair descent velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Replace terrain with stairs-down curriculum (keeps rough sim/contact setup).
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = stairs_down_terrains_cfg()
  if play:
    gen = cfg.scene.terrain.terrain_generator
    gen.curriculum = False
    gen.num_cols = 5
    gen.num_rows = 5
    gen.border_width = 10.0
    gen.difficulty_range = (0.8, 1.0)
    # Blind policy doesn't use height scan; hide debug rays in play view.
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        sensor.debug_vis = False

  # Blind actor: drop height scan from the policy observation.
  del cfg.observations["actor"].terms["height_scan"]
  cfg.observations["actor"].history_length = BLIND_ACTOR_HISTORY_LENGTH

  # Descent commands: slower forward speeds than ascent, no heading control.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None  # Must be unset when heading_command=False.
  twist_cmd.ranges.lin_vel_x = (0.2, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  cfg.curriculum.pop("command_vel", None)

  if play:
    twist_cmd.ranges.lin_vel_x = (0.2, 0.4)
    twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
    twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  return cfg
