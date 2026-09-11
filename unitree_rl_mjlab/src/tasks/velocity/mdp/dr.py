"""Task-specific domain randomization functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.envs.mdp.dr._core import (
  _DEFAULT_ASSET_CFG,
  Ranges,
  _randomize_model_field,
)
from mjlab.envs.mdp.dr._types import Distribution, Operation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("geom_solref")
def geom_solref(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom contact solver reference (softness).

  Randomizes ``geom_solref = (timeconst, dampratio)`` for the selected geoms.
  Applied to robot foot geoms (which carry ``priority=1`` so their solref
  dominates foot-terrain contacts), this simulates ground compliance ranging
  from hard tile to soft carpet without touching the terrain entity.

  Axis 0 is the time constant (s), axis 1 the damping ratio.
  """
  _randomize_model_field(
    env,
    env_ids,
    "geom_solref",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1],
    valid_axes=[0, 1],
  )
