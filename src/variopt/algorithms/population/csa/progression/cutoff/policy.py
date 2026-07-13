"""Public CSA cutoff scheduling policy."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

from .observation import CSACutoffObservation
from .state import CSACutoffState

CSAReductionMethod = Literal["exponential", "linear"]
CSARecoverMode = Literal["none", "score_gap_increase", "score_gap_decrease"]


@dataclass(frozen=True, slots=True)
class CSACutoffSchedule:
    """CSA-specific distance-cutoff scheduling policy.

    Parameters
    ----------
    initial_distance_cutoff : float | None, default=None
        Explicit initial cutoff. ``None`` derives the cutoff from the runtime
        average distance.
    minimum_distance_cutoff : float | None, default=None
        Explicit minimum cutoff. ``None`` derives the minimum cutoff from the
        runtime average distance or mirrors ``initial_distance_cutoff`` when it
        is explicit.
    initial_distance_divisor : float, default=2.0
        Divisor used to derive the initial cutoff from average distance.
    minimum_distance_divisor : float, default=5.0
        Divisor used to derive the minimum cutoff from average distance.
    reduction_method : CSAReductionMethod, default="exponential"
        Decay mode used for cutoff reduction.
    reduction_factor : float, default=0.95
        Multiplicative or additive decay factor, depending on
        ``reduction_method``.
    stagnation_update_limit : int, default=0
        Maximum unused-entry count that still triggers cycle increment.
    cycle_increment_requires_minimum_cutoff : bool, default=True
        Whether the cutoff must already be at the minimum before stagnation can
        increment the CSA cycle.
    recover_steps : int, default=0
        Number of inverse decay steps to apply during recovery.
    recover_mode : CSARecoverMode, default="none"
        Score-gap condition that triggers recovery jumps.
    """

    initial_distance_cutoff: float | None = None
    minimum_distance_cutoff: float | None = None
    initial_distance_divisor: float = 2.0
    minimum_distance_divisor: float = 5.0
    reduction_method: CSAReductionMethod = "exponential"
    reduction_factor: float = 0.95
    stagnation_update_limit: int = 0
    cycle_increment_requires_minimum_cutoff: bool = True
    recover_steps: int = 0
    recover_mode: CSARecoverMode = "none"

    def __post_init__(self) -> None:
        """Reject invalid cutoff policy definitions."""
        if (
            self.initial_distance_cutoff is not None
            and self.initial_distance_cutoff < 0.0
        ):
            msg = "initial_distance_cutoff must be non-negative"
            raise ValueError(msg)

        if (
            self.minimum_distance_cutoff is not None
            and self.minimum_distance_cutoff < 0.0
        ):
            msg = "minimum_distance_cutoff must be non-negative"
            raise ValueError(msg)

        if self.initial_distance_divisor <= 0.0:
            msg = "initial_distance_divisor must be positive"
            raise ValueError(msg)

        if self.minimum_distance_divisor <= 0.0:
            msg = "minimum_distance_divisor must be positive"
            raise ValueError(msg)

        if self.reduction_method not in {"exponential", "linear"}:
            msg = "reduction_method must be 'exponential' or 'linear'"
            raise ValueError(msg)

        if self.reduction_method == "exponential":
            if self.reduction_factor <= 0.0 or self.reduction_factor > 1.0:
                msg = "exponential reduction_factor must be in the interval (0.0, 1.0]"
                raise ValueError(msg)
        elif self.reduction_factor < 0.0:
            msg = "linear reduction_factor must be non-negative"
            raise ValueError(msg)

        if self.stagnation_update_limit < 0:
            msg = "stagnation_update_limit must be non-negative"
            raise ValueError(msg)

        if self.recover_steps < 0:
            msg = "recover_steps must be non-negative"
            raise ValueError(msg)

        if self.recover_mode not in {
            "none",
            "score_gap_increase",
            "score_gap_decrease",
        }:
            msg = (
                "recover_mode must be one of "
                "'none', 'score_gap_increase', or 'score_gap_decrease'"
            )
            raise ValueError(msg)

    def build_initial_state(self) -> CSACutoffState:
        """Return the initial CSA cutoff state implied by this schedule.

        Returns
        -------
        CSACutoffState
            Initial cutoff state aligned with the configured schedule.
        """
        if self.initial_distance_cutoff is None:
            return CSACutoffState()

        distance_cutoff, minimum_distance_cutoff = self.resolve_initial_cutoffs(
            average_distance=None,
        )
        return CSACutoffState(
            distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=minimum_distance_cutoff,
            cutoff_recover_limit=distance_cutoff,
        )

    @property
    def requires_average_distance_for_initialization(self) -> bool:
        """Return whether initial cutoff resolution needs average distance."""
        return self.initial_distance_cutoff is None

    def resolve_initial_cutoffs(
        self,
        *,
        average_distance: float | None,
    ) -> tuple[float, float]:
        """Resolve the active and minimum cutoff from explicit or inferred data.

        Parameters
        ----------
        average_distance : float | None
            Runtime average distance used when explicit cutoffs are not set.

        Returns
        -------
        tuple[float, float]
            Active cutoff and minimum cutoff.

        Raises
        ------
        ValueError
            If the schedule requires ``average_distance`` and it is missing.
        """
        distance_cutoff = self.initial_distance_cutoff
        if distance_cutoff is None:
            if average_distance is None:
                msg = (
                    "average_distance is required when initial_distance_cutoff is unset"
                )
                raise ValueError(msg)

            distance_cutoff = average_distance / self.initial_distance_divisor

        minimum_distance_cutoff = self.minimum_distance_cutoff
        if minimum_distance_cutoff is None:
            if self.initial_distance_cutoff is not None:
                minimum_distance_cutoff = distance_cutoff
            else:
                assert average_distance is not None
                minimum_distance_cutoff = (
                    average_distance / self.minimum_distance_divisor
                )

        if minimum_distance_cutoff > distance_cutoff:
            distance_cutoff = minimum_distance_cutoff

        return float(distance_cutoff), float(minimum_distance_cutoff)

    @property
    def requires_bank_crowding(self) -> bool:
        """Return whether cutoff advancement needs current bank geometry."""
        return False

    def resolve_reduction_speed(
        self,
        *,
        observation: CSACutoffObservation,
    ) -> float:
        """Return the multiplier applied to one fixed annealing step.

        Parameters
        ----------
        observation : CSACutoffObservation
            Canonical post-update evidence for the current iteration.

        Returns
        -------
        float
            Positive reduction-speed multiplier. The fixed schedule always
            returns ``1.0``.
        """
        _ = observation
        return 1.0

    def reduce(
        self,
        *,
        distance_cutoff: float,
        minimum_distance_cutoff: float,
        speed: float = 1.0,
    ) -> float:
        """Return the next cutoff after one decay step.

        Parameters
        ----------
        distance_cutoff : float
            Current active cutoff.
        minimum_distance_cutoff : float
            Lower bound for cutoff decay.
        speed : float, default=1.0
            Positive multiplier applied to the configured reduction step.

        Returns
        -------
        float
            Reduced cutoff after clamping to the minimum cutoff.
        """
        if type(speed) is bool or not isinstance(speed, Real):
            msg = "speed must be numeric"
            raise TypeError(msg)
        normalized_speed = float(speed)
        if not isfinite(normalized_speed) or normalized_speed <= 0.0:
            msg = "speed must be finite and positive"
            raise ValueError(msg)

        if normalized_speed == 1.0:
            if self.reduction_method == "linear":
                next_distance_cutoff = distance_cutoff - self.reduction_factor
            else:
                next_distance_cutoff = distance_cutoff * self.reduction_factor
        elif self.reduction_method == "linear":
            next_distance_cutoff = (
                distance_cutoff - self.reduction_factor * normalized_speed
            )
        else:
            next_distance_cutoff = distance_cutoff * (
                self.reduction_factor**normalized_speed
            )

        return max(minimum_distance_cutoff, next_distance_cutoff)

    def recover(self, *, distance_cutoff: float) -> float:
        """Return the cutoff after one recovery jump.

        Parameters
        ----------
        distance_cutoff : float
            Current active cutoff.

        Returns
        -------
        float
            Cutoff after applying the configured recovery jump.
        """
        if self.recover_steps == 0:
            return distance_cutoff

        if self.reduction_method == "linear":
            return distance_cutoff + (self.reduction_factor * self.recover_steps)

        return distance_cutoff / (self.reduction_factor**self.recover_steps)

    def should_increment_cycle(
        self,
        *,
        unused_entry_count: int,
        cutoff_at_minimum: bool,
    ) -> bool:
        """Return whether stagnation should advance the CSA cycle.

        Parameters
        ----------
        unused_entry_count : int
            Number of bank entries left unused by the current selection logic.
        cutoff_at_minimum : bool
            Whether the active cutoff has already reached its minimum.

        Returns
        -------
        bool
            ``True`` when the cutoff schedule allows a cycle increment.
        """
        return unused_entry_count <= self.stagnation_update_limit and (
            not self.cycle_increment_requires_minimum_cutoff or cutoff_at_minimum
        )

    def should_recover(
        self,
        *,
        previous_score_gap: float | None,
        current_score_gap: float | None,
    ) -> bool:
        """Return whether score-gap recovery should fire.

        Parameters
        ----------
        previous_score_gap : float | None
            Score gap recorded before the current refresh or update.
        current_score_gap : float | None
            Score gap recorded after the current refresh or update.

        Returns
        -------
        bool
            ``True`` when the configured recovery mode is satisfied.
        """
        if self.recover_mode == "none" or self.recover_steps == 0:
            return False

        if previous_score_gap is None or current_score_gap is None:
            return False

        if self.recover_mode == "score_gap_increase":
            return current_score_gap > previous_score_gap

        return current_score_gap < previous_score_gap
