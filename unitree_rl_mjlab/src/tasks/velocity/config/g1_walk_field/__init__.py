from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_g1_walk_field_env_cfg
from .rl_cfg import unitree_g1_walk_field_ppo_runner_cfg

# Stage 1: arm adaptation (nominal ground, hand-clearance rewards).
register_mjlab_task(
  task_id="Unitree-G1-Walk-Field",
  env_cfg=unitree_g1_walk_field_env_cfg(stage=1),
  play_env_cfg=unitree_g1_walk_field_env_cfg(play=True, stage=3),
  rl_cfg=unitree_g1_walk_field_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Stage 2: ground adaptation (friction + solref + palm mass DR).
register_mjlab_task(
  task_id="Unitree-G1-Walk-Field-DR",
  env_cfg=unitree_g1_walk_field_env_cfg(stage=2),
  play_env_cfg=unitree_g1_walk_field_env_cfg(play=True, stage=3),
  rl_cfg=unitree_g1_walk_field_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Stage 3: strict (DR + hard termination on right-hand/right-thigh contact).
register_mjlab_task(
  task_id="Unitree-G1-Walk-Field-Strict",
  env_cfg=unitree_g1_walk_field_env_cfg(stage=3),
  play_env_cfg=unitree_g1_walk_field_env_cfg(play=True, stage=3),
  rl_cfg=unitree_g1_walk_field_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
