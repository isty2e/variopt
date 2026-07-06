"""CSA cutoff runtime-state value objects."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from ......json_types import (
    JSONDict,
    JSONValue,
    require_json_bool,
    require_json_field,
    require_json_int,
    require_json_optional_finite_float,
)


@dataclass(frozen=True, slots=True)
class CSACutoffState:
    """Immutable cutoff runtime-state for CSA progression.

    Parameters
    ----------
    iteration_count : int, default=0
        Number of cutoff iterations completed in the current run phase.
    cycle_count : int, default=0
        Number of completed progression cycles.
    distance_cutoff : float | None, optional
        Active distance cutoff when scheduling has been initialized.
    minimum_distance_cutoff : float | None, optional
        Lower bound for the active distance cutoff.
    cutoff_recover_limit : float | None, optional
        Recover-limit ceiling used by score-gap recovery logic.
    previous_score_gap : float | None, optional
        Score gap observed at the previous cutoff update.
    refresh_in_progress : bool, default=False
        Whether the state currently represents a refresh interval.
    """

    iteration_count: int = 0
    cycle_count: int = 0
    distance_cutoff: float | None = None
    minimum_distance_cutoff: float | None = None
    cutoff_recover_limit: float | None = None
    previous_score_gap: float | None = None
    refresh_in_progress: bool = False

    def __post_init__(self) -> None:
        """Reject invalid cutoff runtime-states."""
        if self.iteration_count < 0:
            msg = "iteration_count must be non-negative"
            raise ValueError(msg)

        if self.cycle_count < 0:
            msg = "cycle_count must be non-negative"
            raise ValueError(msg)

        cutoff_is_none = self.distance_cutoff is None
        minimum_is_none = self.minimum_distance_cutoff is None
        if cutoff_is_none != minimum_is_none:
            msg = "distance_cutoff and minimum_distance_cutoff must be both set or both unset"
            raise ValueError(msg)

        if cutoff_is_none and self.cutoff_recover_limit is not None:
            msg = "cutoff_recover_limit must be unset when cutoff scheduling is uninitialized"
            raise ValueError(msg)

        if cutoff_is_none and self.previous_score_gap is not None:
            msg = "previous_score_gap must be unset when cutoff scheduling is uninitialized"
            raise ValueError(msg)

        if self.refresh_in_progress and (
            not cutoff_is_none
            or self.cutoff_recover_limit is not None
            or self.previous_score_gap is not None
        ):
            msg = (
                "refresh_in_progress states must not carry active cutoff runtime values"
            )
            raise ValueError(msg)

        if self.distance_cutoff is None or self.minimum_distance_cutoff is None:
            return

        distance_cutoff = float(self.distance_cutoff)
        minimum_distance_cutoff = float(self.minimum_distance_cutoff)
        cutoff_recover_limit = self.cutoff_recover_limit
        if cutoff_recover_limit is None:
            cutoff_recover_limit = distance_cutoff

        cutoff_recover_limit = float(cutoff_recover_limit)

        if not np.isfinite(distance_cutoff) or distance_cutoff < 0.0:
            msg = "distance_cutoff must be a finite non-negative float"
            raise ValueError(msg)

        if not np.isfinite(minimum_distance_cutoff) or minimum_distance_cutoff < 0.0:
            msg = "minimum_distance_cutoff must be a finite non-negative float"
            raise ValueError(msg)

        if minimum_distance_cutoff > distance_cutoff:
            msg = "minimum_distance_cutoff must not exceed distance_cutoff"
            raise ValueError(msg)

        if not np.isfinite(cutoff_recover_limit) or cutoff_recover_limit < 0.0:
            msg = "cutoff_recover_limit must be a finite non-negative float"
            raise ValueError(msg)

        previous_score_gap = self.previous_score_gap
        if previous_score_gap is not None:
            previous_score_gap = float(previous_score_gap)
            if not np.isfinite(previous_score_gap) or previous_score_gap < 0.0:
                msg = "previous_score_gap must be a finite non-negative float"
                raise ValueError(msg)

        object.__setattr__(self, "distance_cutoff", distance_cutoff)
        object.__setattr__(
            self,
            "minimum_distance_cutoff",
            minimum_distance_cutoff,
        )
        object.__setattr__(self, "cutoff_recover_limit", cutoff_recover_limit)
        object.__setattr__(self, "previous_score_gap", previous_score_gap)

    @property
    def cutoff_is_initialized(self) -> bool:
        """Return whether cutoff scheduling has been initialized."""
        return self.distance_cutoff is not None

    @property
    def cutoff_at_minimum(self) -> bool:
        """Return whether the active cutoff has reached its minimum."""
        if not self.cutoff_is_initialized:
            return False

        assert self.distance_cutoff is not None
        assert self.minimum_distance_cutoff is not None
        return self.distance_cutoff <= self.minimum_distance_cutoff

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the cutoff state.

        Returns
        -------
        JSONDict
            JSON-safe cutoff-state snapshot.
        """
        return {
            "iteration_count": self.iteration_count,
            "cycle_count": self.cycle_count,
            "distance_cutoff": self.distance_cutoff,
            "minimum_distance_cutoff": self.minimum_distance_cutoff,
            "cutoff_recover_limit": self.cutoff_recover_limit,
            "previous_score_gap": self.previous_score_gap,
            "refresh_in_progress": self.refresh_in_progress,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
    ) -> Self:
        """Build a cutoff state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe cutoff-state snapshot.

        Returns
        -------
        Self
            Reconstructed cutoff state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        iteration_count = require_json_int(
            require_json_field(data, "iteration_count"),
            field_name="iteration_count",
        )
        cycle_count = require_json_int(
            require_json_field(data, "cycle_count"),
            field_name="cycle_count",
        )
        refresh_in_progress = require_json_bool(
            require_json_field(data, "refresh_in_progress"),
            field_name="refresh_in_progress",
        )
        distance_cutoff = require_json_optional_finite_float(
            require_json_field(data, "distance_cutoff"),
            field_name="distance_cutoff",
        )
        minimum_distance_cutoff = require_json_optional_finite_float(
            require_json_field(data, "minimum_distance_cutoff"),
            field_name="minimum_distance_cutoff",
        )
        cutoff_recover_limit = require_json_optional_finite_float(
            require_json_field(data, "cutoff_recover_limit"),
            field_name="cutoff_recover_limit",
        )
        previous_score_gap = require_json_optional_finite_float(
            require_json_field(data, "previous_score_gap"),
            field_name="previous_score_gap",
        )

        return cls(
            iteration_count=iteration_count,
            cycle_count=cycle_count,
            distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=minimum_distance_cutoff,
            cutoff_recover_limit=cutoff_recover_limit,
            previous_score_gap=previous_score_gap,
            refresh_in_progress=refresh_in_progress,
        )

    def initialize_cutoff(
        self,
        *,
        distance_cutoff: float,
        minimum_distance_cutoff: float,
        previous_score_gap: float | None = None,
    ) -> Self:
        """Return a copy with cutoff scheduling initialized.

        Parameters
        ----------
        distance_cutoff : float
            Initial active cutoff.
        minimum_distance_cutoff : float
            Lower bound for later cutoff decay.
        previous_score_gap : float | None, optional
            Optional initial score-gap observation.

        Returns
        -------
        Self
            State with active cutoff scheduling.
        """
        return type(self)(
            iteration_count=self.iteration_count,
            cycle_count=self.cycle_count,
            distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=minimum_distance_cutoff,
            cutoff_recover_limit=distance_cutoff,
            previous_score_gap=previous_score_gap,
            refresh_in_progress=False,
        )

    def advance_iteration(
        self,
        *,
        distance_cutoff: float | None = None,
        cycle_increment: bool = False,
        cutoff_recover_limit: float | None = None,
        previous_score_gap: float | None = None,
    ) -> Self:
        """Return the next cutoff runtime-state.

        Parameters
        ----------
        distance_cutoff : float | None, optional
            Optional next active cutoff.
        cycle_increment : bool, default=False
            Whether this iteration advances the progression cycle count.
        cutoff_recover_limit : float | None, optional
            Optional next recover-limit ceiling.
        previous_score_gap : float | None, optional
            Score gap observed for the just-completed iteration.

        Returns
        -------
        Self
            Advanced cutoff runtime-state.
        """
        next_distance_cutoff = self.distance_cutoff
        if distance_cutoff is not None:
            next_distance_cutoff = distance_cutoff

        next_cutoff_recover_limit = self.cutoff_recover_limit
        if cutoff_recover_limit is not None:
            next_cutoff_recover_limit = cutoff_recover_limit

        return type(self)(
            iteration_count=self.iteration_count + 1,
            cycle_count=self.cycle_count + int(cycle_increment),
            distance_cutoff=next_distance_cutoff,
            minimum_distance_cutoff=self.minimum_distance_cutoff,
            cutoff_recover_limit=next_cutoff_recover_limit,
            previous_score_gap=previous_score_gap,
            refresh_in_progress=False,
        )

    def begin_refresh(self) -> Self:
        """Return a cutoff runtime-state that has entered refresh mode."""
        return type(self)(
            iteration_count=self.iteration_count,
            cycle_count=self.cycle_count,
            distance_cutoff=None,
            minimum_distance_cutoff=None,
            cutoff_recover_limit=None,
            previous_score_gap=None,
            refresh_in_progress=True,
        )

    def complete_refresh(
        self,
        *,
        distance_cutoff: float,
        minimum_distance_cutoff: float,
        previous_score_gap: float | None = None,
    ) -> Self:
        """Return a cutoff runtime-state that has completed refresh.

        Parameters
        ----------
        distance_cutoff : float
            New active cutoff after refresh.
        minimum_distance_cutoff : float
            Lower bound for later cutoff decay.
        previous_score_gap : float | None, optional
            Optional score-gap observation carried into the refreshed state.

        Returns
        -------
        Self
            Refreshed cutoff runtime-state with cycle count reset.
        """
        return type(self)(
            iteration_count=self.iteration_count,
            cycle_count=0,
            distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=minimum_distance_cutoff,
            cutoff_recover_limit=distance_cutoff,
            previous_score_gap=previous_score_gap,
            refresh_in_progress=False,
        )
