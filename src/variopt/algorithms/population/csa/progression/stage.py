"""Private staged-bank growth state for CSA."""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from typing_extensions import Self

from .....json_types import JSONDict, JSONValue, require_json_int, require_json_list
from ..indexing import remap_indices_after_removal


@dataclass(frozen=True, slots=True)
class CSAStageState:
    """Private staged-bank growth state for CSA.

    Parameters
    ----------
    base_capacity : int
        Initial bank capacity before staged growth.
    max_capacity : int
        Maximum bank capacity reachable through staged growth.
    stage_index : int, default=0
        Zero-based growth stage index.
    stage_round : int, default=0
        Current round within the stage lifecycle.
    seed_mask : frozenset[int], default=frozenset()
        Active seed indices reserved by the current stage transition.
    partner_mask : frozenset[int], default=frozenset()
        Active partner indices reserved by the current stage transition.
    """

    base_capacity: int
    max_capacity: int
    stage_index: int = 0
    stage_round: int = 0
    seed_mask: frozenset[int] = field(default_factory=frozenset)
    partner_mask: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Reject invalid staged-bank states."""
        if self.base_capacity <= 0:
            msg = "base_capacity must be positive"
            raise ValueError(msg)

        if self.max_capacity < self.base_capacity:
            msg = "max_capacity must be at least base_capacity"
            raise ValueError(msg)

        if self.stage_index < 0:
            msg = "stage_index must be non-negative"
            raise ValueError(msg)

        if self.stage_round not in {0, 1}:
            msg = "stage_round must be 0 or 1"
            raise ValueError(msg)

        if self.current_capacity > self.max_capacity:
            msg = "current stage capacity must not exceed max_capacity"
            raise ValueError(msg)

    @property
    def current_capacity(self) -> int:
        """Return the current target bank capacity for this stage."""
        return min(
            self.max_capacity,
            self.base_capacity * (self.stage_index + 1),
        )

    @property
    def staged_growth_enabled(self) -> bool:
        """Return whether staged bank growth is enabled."""
        return self.max_capacity > self.base_capacity

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the stage state.

        Returns
        -------
        JSONDict
            JSON-safe stage-state snapshot.
        """
        return {
            "base_capacity": self.base_capacity,
            "max_capacity": self.max_capacity,
            "stage_index": self.stage_index,
            "stage_round": self.stage_round,
            "seed_mask": list(self.seed_mask),
            "partner_mask": list(self.partner_mask),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
    ) -> Self:
        """Build a stage state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe stage-state snapshot.

        Returns
        -------
        Self
            Reconstructed stage state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        base_capacity = require_json_int(data.get("base_capacity"), field_name="base_capacity")
        max_capacity = require_json_int(data.get("max_capacity"), field_name="max_capacity")
        stage_index = require_json_int(data.get("stage_index"), field_name="stage_index")
        stage_round = require_json_int(data.get("stage_round"), field_name="stage_round")
        raw_seed_mask = require_json_list(data.get("seed_mask"), field_name="seed_mask")
        raw_partner_mask = require_json_list(data.get("partner_mask"), field_name="partner_mask")

        seed_mask: list[int] = []
        for raw_index in raw_seed_mask:
            if not isinstance(raw_index, int):
                msg = "stage-state snapshot seed_mask values must be integers"
                raise TypeError(msg)
            seed_mask.append(raw_index)

        partner_mask: list[int] = []
        for raw_index in raw_partner_mask:
            if not isinstance(raw_index, int):
                msg = "stage-state snapshot partner_mask values must be integers"
                raise TypeError(msg)
            partner_mask.append(raw_index)

        return cls(
            base_capacity=base_capacity,
            max_capacity=max_capacity,
            stage_index=stage_index,
            stage_round=stage_round,
            seed_mask=frozenset(seed_mask),
            partner_mask=frozenset(partner_mask),
        )

    def next_transition(self) -> tuple[Self, bool] | None:
        """Return the next staged-bank transition, if any."""
        if not self.staged_growth_enabled:
            return None

        if self.stage_index == 0 and self.current_capacity < self.max_capacity:
            return self._growth_transition(next_stage_index=1)

        if self.stage_round == 0:
            return (
                type(self)(
                    base_capacity=self.base_capacity,
                    max_capacity=self.max_capacity,
                    stage_index=self.stage_index,
                    stage_round=1,
                ),
                False,
            )

        if self.current_capacity < self.max_capacity:
            return self._growth_transition(next_stage_index=self.stage_index + 1)

        return None

    def without_updated_seed_mask(self, updated_indices: AbstractSet[int]) -> Self:
        """Return a copy with updated indices removed from the seed mask.

        Parameters
        ----------
        updated_indices : AbstractSet[int]
            Seed indices that have already been updated this round.

        Returns
        -------
        Self
            Stage state with those indices removed from ``seed_mask``.
        """
        if not updated_indices or not self.seed_mask:
            return self

        return self.with_masks(
            seed_mask=frozenset(
                index for index in self.seed_mask if index not in updated_indices
            ),
            partner_mask=self.partner_mask,
        )

    def remove_indices(
        self,
        *,
        removed_indices: AbstractSet[int],
        entry_count: int,
    ) -> Self:
        """Return a copy remapped after bank indices have been removed.

        Parameters
        ----------
        removed_indices : collections.abc.Set[int]
            Bank indices removed from the previous bank snapshot.
        entry_count : int
            Current bank size after removal.

        Returns
        -------
        Self
            Stage state whose masks are aligned with the current bank indexing.

        Raises
        ------
        ValueError
            If ``entry_count`` is negative.
        """
        if entry_count < 0:
            msg = "entry_count must be non-negative"
            raise ValueError(msg)

        if not removed_indices:
            return self

        return self.with_masks(
            seed_mask=remap_indices_after_removal(
                self.seed_mask,
                removed_indices=removed_indices,
                entry_count=entry_count,
            ),
            partner_mask=remap_indices_after_removal(
                self.partner_mask,
                removed_indices=removed_indices,
                entry_count=entry_count,
            ),
        )

    def with_masks(
        self,
        *,
        seed_mask: frozenset[int],
        partner_mask: frozenset[int],
    ) -> Self:
        """Return a copy with replacement stage masks.

        Parameters
        ----------
        seed_mask : frozenset[int]
            Replacement seed mask.
        partner_mask : frozenset[int]
            Replacement partner mask.

        Returns
        -------
        Self
            Stage state with the supplied masks.
        """
        return type(self)(
            base_capacity=self.base_capacity,
            max_capacity=self.max_capacity,
            stage_index=self.stage_index,
            stage_round=self.stage_round,
            seed_mask=seed_mask,
            partner_mask=partner_mask,
        )

    def _growth_transition(self, *, next_stage_index: int) -> tuple[Self, bool]:
        previous_capacity = self.current_capacity
        mask = frozenset(range(previous_capacity))
        return (
            type(self)(
                base_capacity=self.base_capacity,
                max_capacity=self.max_capacity,
                stage_index=next_stage_index,
                stage_round=0,
                seed_mask=mask,
                partner_mask=mask,
            ),
            True,
        )
