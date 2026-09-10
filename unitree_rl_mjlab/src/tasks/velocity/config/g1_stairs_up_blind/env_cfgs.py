"""Unitree G1 blind stair ascent environment configuration.

Task design migrated from G1DWAQ_Lab:
- Blind actor: no height scan, proprioception only, 5-frame history
  (oldest -> newest, same ordering as the C++ deployment buffer).
- Asymmetric critic: keeps privileged observations (base lin vel,
  height scan, foot states).
- Stairs-up terrain with difficulty curriculum: step height is
  interpolated from 0.0 m (easiest row) to 0.23 m (hardest row),
  mixing 0.28 m and 0.34 m step widths, plus flat tiles for
  approach/recovery phases.
"""

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg

# Actor observation history length (G1DWAQ blind-walking design).
BLIND_ACTOR_HISTORY_LENGTH = 5


def stairs_up_terrains_cfg() -> TerrainGeneratorCfg:
  """Stairs-up focused terrain curriculum (G1DWAQ ROUGH_TERRAINS_CFG style)."""
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    sub_terrains={
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.2),
      "stairs_up_narrow": terrain_gen.BoxPyramidStairsTerrainCfg(
        proportion=0.4,
        step_height_range=(0.0, 0.23),
        step_width=0.28,
        platform_width=3.0,
        border_width=1.0,
      ),
      "stairs_up_wide": terrain_gen.BoxPyramidStairsTerrainCfg(
        proportion=0.4,
        step_height_range=(0.0, 0.23),
        step_width=0.34,
        platform_width=3.0,
        border_width=1.0,
      ),
    },
    add_lights=True,
  )


def unitree_g1_stairs_up_blind_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 blind stair ascent velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Replace terrain with stairs-up curriculum (keeps rough sim/contact setup).
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = stairs_up_terrains_cfg()
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
  # Critic keeps height scan + privileged foot/base terms (history_length=1).

  # Stair-appropriate commands: forward only, small lateral/yaw range,
  # no heading control (keep facing the stairs).
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None  # Must be unset when heading_command=False.
  twist_cmd.ranges.lin_vel_x = (0.2, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  # Keep terrain difficulty curriculum; drop the flat-task velocity widening
  # (stair command ranges stay fixed for the whole run).
  cfg.curriculum.pop("command_vel", None)

  if play:
    twist_cmd.ranges.lin_vel_x = (0.2, 0.6)
    twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
    twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  return cfg
