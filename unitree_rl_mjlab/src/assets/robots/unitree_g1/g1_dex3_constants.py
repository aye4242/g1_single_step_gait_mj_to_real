"""Unitree G1 constants (29-DOF with Dex3 dexterous right hand).

Reuses the actuator and collision configuration from the standard G1 (the 29
body joints are identical); only the right wrist end-effector differs: the
rubber hand is replaced by a Dex3 whose fingers are welded in a closed-fist
pose (not actuated, controlled independently on the real robot via
``rt/dex3/right/cmd``).
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.entity import EntityCfg

from .g1_constants import (
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
  G1_ACTUATOR_WAIST,
  G1_ARTICULATION,
  FULL_COLLISION,
  get_assets,
)

##
# MJCF and assets.
##

G1_DEX3_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_dex3.xml"
)
assert G1_DEX3_XML.exists()


def get_g1_dex3_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_DEX3_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Keyframe config.
##

# Right shoulder is abducted further out (roll -0.30 vs the stock -0.18) so the
# heavier Dex3 right hand keeps nominal static clearance from the right thigh.
HOME_KEYFRAME_DEX3 = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.30,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

# Same 29 actuated body joints as the stock G1; welded Dex3 fingers add none.


def get_g1_dex3_robot_cfg() -> EntityCfg:
  """Get a fresh G1-Dex3 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME_DEX3,
    collisions=(FULL_COLLISION,),
    spec_fn=get_g1_dex3_spec,
    articulation=G1_ARTICULATION,
  )
