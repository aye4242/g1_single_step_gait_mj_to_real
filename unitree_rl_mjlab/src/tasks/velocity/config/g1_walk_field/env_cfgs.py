"""Unitree G1 walk-field environment configuration (reception-scene gait tuning).

Task: fine-tune the stock blind-walking policy for the real reception site:
- Right wrist carries a Dex3 dexterous hand (fingers welded in fist pose,
  ~0.70 kg heavier than the rubber hand); left hand stays rubber.
- Stage 1 (arm adaptation): nominal ground, hand-clearance rewards active.
- Stage 2 (ground adaptation): adds friction DR (smooth floors) and contact
  solref DR (soft carpet), plus Dex3 palm mass DR.
- Stage 3 (strict): additionally terminates on right-hand/right-thigh
  contact force > 20 N.

Observations match the single-step blind tasks (no height scan, 5-frame
proprioceptive history) so checkpoints can be warm-started from them.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import G1_ACTION_SCALE, get_g1_dex3_robot_cfg
from src.tasks.velocity import mdp
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1_stairs_up_blind.env_cfgs import (
  BLIND_ACTOR_HISTORY_LENGTH,
)
from src.tasks.velocity.mdp.dr import geom_solref
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr


def unitree_g1_walk_field_env_cfg(
  play: bool = False, stage: int = 1
) -> ManagerBasedRlEnvCfg:
  """Create the G1-Dex3 reception walking configuration.

  Args:
    play: Enable play-mode overrides (infinite episode, no push/DR).
    stage: Training stage (1 nominal, 2 ground DR, 3 strict termination).
  """
  # Derive from the rough cfg (NOT flat) so the critic keeps the height-scan
  # privileged term and stays warmstart-compatible with the single-step /
  # stairs blind checkpoints; terrain is then switched to a plane below.
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Flat ground only.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)
  # Plane + feet produce more contacts than the pyramid tiles the rough cfg
  # was sized for; let MuJoCo auto-size the contact buffer.
  cfg.sim.nconmax = None
  # Terrain tiles are gone in play mode; the randomize_terrain event is
  # meaningless on a plane.
  cfg.events.pop("randomize_terrain", None)

  # Dex3 right hand robot (29 actuated joints unchanged).
  cfg.scene.entities = {"robot": get_g1_dex3_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Blind actor: drop height scan from the policy observation; the critic
  # keeps it as privileged information (same as the single-step blind tasks).
  del cfg.observations["actor"].terms["height_scan"]
  # 5-frame proprioceptive history, flattened oldest -> newest.
  cfg.observations["actor"].history_length = BLIND_ACTOR_HISTORY_LENGTH
  # Hide the debug rays (no policy use).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      sensor.debug_vis = False

  # Hand-thigh contact sensor: right Dex3 palm subtree vs right thigh subtree.
  hand_thigh_cfg = ContactSensorCfg(
    name="hand_thigh_contact",
    primary=ContactMatch(
      mode="subtree", pattern=r"^right_hand_palm_link$", entity="robot"
    ),
    secondary=ContactMatch(
      mode="subtree", pattern="right_hip_yaw_link", entity="robot"
    ),    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (hand_thigh_cfg,)

  # Reception-scene velocity commands: gentle omnidirectional walking with
  # significant standing time; lateral motion is a first-class direction.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  twist_cmd.rel_standing_envs = 0.08
  # Fixed command ranges for the whole run (no velocity curriculum).
  cfg.curriculum.pop("command_vel", None)

  ##
  # Events / domain randomization (stage-dependent).
  ##

  if stage >= 2:
    # Smooth floors: widen friction DR down to wet-tile levels.
    cfg.events["foot_friction"].params["ranges"] = (0.10, 1.75)
    # Soft carpet: randomize foot contact solver reference
    # (timeconst 0.01 s = hard tile ... 0.15 s = soft carpet;
    #  dampratio 0.5 ... 1.5).
    cfg.events["ground_solref"] = EventTermCfg(
      func=geom_solref,
      mode="startup",
      params={
        "ranges": {0: (0.01, 0.15), 1: (0.5, 1.5)},
        "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
        "shared_random": True,
      },
    )
    # Dex3 payload variation (e.g. holding objects).
    cfg.events["dex3_palm_mass"] = EventTermCfg(
      func=dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("right_hand_palm_link",)),
        "operation": "scale",
        "ranges": (0.8, 1.2),
      },
    )
  else:
    # Stage 1: nominal ground (feet keep their fixed friction).
    cfg.events.pop("foot_friction", None)

  if play:
    # Never randomize/terminate in play mode.
    cfg.events.pop("ground_solref", None)
    cfg.events.pop("dex3_palm_mass", None)

  ##
  # Rewards.
  ##

  rewards = cfg.rewards

  # Task tracking: soft lateral tracking + hard overshoot penalty (the
  # "hop-and-drift into the wall" failure mode on soft carpet).
  rewards["track_angular_velocity"].weight = 0.8
  rewards["stand_still"].weight = -0.5
  rewards["lateral_overshoot"] = RewardTermCfg(
    func=mdp.lateral_overshoot_penalty,
    weight=-1.0,
    params={"command_name": "twist", "margin": 0.15},
  )

  # Gait quality: phase-locked walking (any jump with both feet airborne
  # mismatches the stance pattern and is penalized), slower cadence.
  rewards["foot_gait"].weight = 0.75
  rewards["foot_gait"].params["period"] = 0.7
  rewards["foot_gait"].params["threshold"] = 0.6
  cfg.observations["actor"].terms["phase"].params["period"] = 0.7
  rewards["feet_air_time"] = RewardTermCfg(
    func=mdp.feet_air_time,
    weight=0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold": 0.35,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  rewards["foot_slip"].weight = -0.5
  rewards["soft_landing"].weight = -2e-3

  # Hand safety: keep the Dex3 right hand clear of the right thigh.
  rewards["hand_thigh_clearance"] = RewardTermCfg(
    func=mdp.body_clearance,
    weight=0.3,
    params={
      "margin": 0.06,
      "asset_cfg_a": SceneEntityCfg("robot", site_names=("right_dex3_palm",)),
      "asset_cfg_b": SceneEntityCfg("robot", site_names=("right_thigh",)),
    },
  )
  rewards["self_collisions"].weight = -2.0
  rewards["self_collisions"].params["force_threshold"] = 5.0

  # Asymmetric arm posture: right arm (Dex3) stays tucked with small swing,
  # left arm swings freely for balance.
  rewards["pose"].params["std_standing"] = {".*": 0.05}
  rewards["pose"].params["std_walking"] = {
    # Lower body (symmetric).
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Right arm (Dex3): tight, tucked close to the nominal abducted pose.
    r"right_shoulder_pitch_joint": 0.08,
    r"right_shoulder_roll_joint": 0.05,
    r"right_shoulder_yaw_joint": 0.1,
    r"right_elbow_joint": 0.08,
    # Left arm (rubber hand): free natural swing.
    r"left_.*shoulder_pitch.*": 0.15,
    r"left_.*shoulder_roll.*": 0.1,
    r"left_.*shoulder_yaw.*": 0.1,
    r"left_elbow_joint": 0.1,
    # Wrists (both): loose.
    r".*_wrist.*": 0.1,
  }
  rewards["pose"].params["std_running"] = dict(
    rewards["pose"].params["std_walking"]
  )
  # Running regime is unused at these command ranges; legs slightly looser.
  rewards["pose"].params["std_running"].update(
    {
      r".*hip_roll.*": 0.25,
      r".*hip_yaw.*": 0.25,
      r".*ankle_pitch.*": 0.25,
    }
  )

  # Smoothness: double action-rate penalty.
  rewards["action_rate_l2"].weight = -0.1
  rewards["angular_momentum"].weight = -0.02

  ##
  # Terminations (stage-dependent).
  ##

  if stage >= 3 and not play:
    cfg.terminations["hand_thigh_contact"] = TerminationTermCfg(
      func=mdp.illegal_contact,
      params={
        "sensor_name": hand_thigh_cfg.name,
        "force_threshold": 20.0,
      },
    )

  return cfg
