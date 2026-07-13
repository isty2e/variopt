"""Public CSA cutoff scheduling policy."""

from dataclasses import dataclass
from math import exp, isfinite, log
from numbers import Real
from typing import Literal

from typing_extensions import override

from .observation import CSACutoffObservation
from .state import CSACutoffState

CSAReductionMethod = Literal["exponential", "linear"]
CSARecoverMode = Literal["none", "score_gap_increase", "score_gap_decrease"]

_MINIMUM_REDUCTION_SPEED = 0.25
_MAXIMUM_REDUCTION_SPEED = 4.0
_MINIMUM_REDUCTION_EXPONENT = log(_MINIMUM_REDUCTION_SPEED)
_MAXIMUM_REDUCTION_EXPONENT = log(_MAXIMUM_REDUCTION_SPEED)


def _normalize_finite_float(value: float | int, *, field_name: str) -> float:
    if type(value) is bool or not isinstance(value, Real):
        msg = f"{field_name} must be numeric"
        raise TypeError(msg)
    normalized_value = float(value)
    if not isfinite(normalized_value):
        msg = f"{field_name} must be finite"
        raise ValueError(msg)
    return normalized_value


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
        initial_distance_cutoff = self.initial_distance_cutoff
        if initial_distance_cutoff is not None:
            initial_distance_cutoff = _normalize_finite_float(
                initial_distance_cutoff,
                field_name="initial_distance_cutoff",
            )
            if initial_distance_cutoff < 0.0:
                msg = "initial_distance_cutoff must be non-negative"
                raise ValueError(msg)
            object.__setattr__(
                self,
                "initial_distance_cutoff",
                initial_distance_cutoff,
            )

        minimum_distance_cutoff = self.minimum_distance_cutoff
        if minimum_distance_cutoff is not None:
            minimum_distance_cutoff = _normalize_finite_float(
                minimum_distance_cutoff,
                field_name="minimum_distance_cutoff",
            )
            if minimum_distance_cutoff < 0.0:
                msg = "minimum_distance_cutoff must be non-negative"
                raise ValueError(msg)
            object.__setattr__(
                self,
                "minimum_distance_cutoff",
                minimum_distance_cutoff,
            )

        initial_distance_divisor = _normalize_finite_float(
            self.initial_distance_divisor,
            field_name="initial_distance_divisor",
        )
        if initial_distance_divisor <= 0.0:
            msg = "initial_distance_divisor must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "initial_distance_divisor", initial_distance_divisor)

        minimum_distance_divisor = _normalize_finite_float(
            self.minimum_distance_divisor,
            field_name="minimum_distance_divisor",
        )
        if minimum_distance_divisor <= 0.0:
            msg = "minimum_distance_divisor must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "minimum_distance_divisor", minimum_distance_divisor)

        if self.reduction_method not in {"exponential", "linear"}:
            msg = "reduction_method must be 'exponential' or 'linear'"
            raise ValueError(msg)

        reduction_factor = _normalize_finite_float(
            self.reduction_factor,
            field_name="reduction_factor",
        )
        if self.reduction_method == "exponential":
            if reduction_factor <= 0.0 or reduction_factor > 1.0:
                msg = "exponential reduction_factor must be in the interval (0.0, 1.0]"
                raise ValueError(msg)
        elif reduction_factor < 0.0:
            msg = "linear reduction_factor must be non-negative"
            raise ValueError(msg)
        object.__setattr__(self, "reduction_factor", reduction_factor)

        if type(self.stagnation_update_limit) is not int:
            msg = "stagnation_update_limit must be an integer"
            raise TypeError(msg)
        if self.stagnation_update_limit < 0:
            msg = "stagnation_update_limit must be non-negative"
            raise ValueError(msg)

        if type(self.cycle_increment_requires_minimum_cutoff) is not bool:
            msg = "cycle_increment_requires_minimum_cutoff must be a bool"
            raise TypeError(msg)

        if type(self.recover_steps) is not int:
            msg = "recover_steps must be an integer"
            raise TypeError(msg)
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
            returns ``1.0``. The optimizer calls this method only when
            ``requires_reduction_observation`` is true.
        """
        _ = observation
        return 1.0

    @property
    def requires_reduction_observation(self) -> bool:
        """Return whether the schedule needs adaptive reduction evidence.

        Returns
        -------
        bool
            ``True`` when the optimizer must construct a
            :class:`CSACutoffObservation` and call
            :meth:`resolve_reduction_speed`. Custom adaptive schedules must
            override this property together with that method. ``False`` keeps
            the fixed-schedule hot path free of observation materialization.
        """
        return False

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
        if type(speed) is float and speed == 1.0:
            if self.reduction_method == "linear":
                next_distance_cutoff = distance_cutoff - self.reduction_factor
            else:
                next_distance_cutoff = distance_cutoff * self.reduction_factor
            return max(minimum_distance_cutoff, next_distance_cutoff)

        if type(speed) is bool or not isinstance(speed, Real):
            msg = "speed must be numeric"
            raise TypeError(msg)
        normalized_speed = float(speed)
        if not isfinite(normalized_speed) or normalized_speed <= 0.0:
            msg = "speed must be finite and positive"
            raise ValueError(msg)

        if self.reduction_method == "linear":
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


@dataclass(frozen=True, slots=True)
class CSALocalRouteCutoffSchedule(CSACutoffSchedule):
    """Bound cutoff annealing by the current local-route share.

    The schedule preserves the fixed exponential annealing backbone and changes
    only the speed of each reduction step. A batch with no full-bank transition
    is neutral. Otherwise, the local share among local, cluster, and far routes
    is compared with ``target_local_route_fraction``. Higher shares accelerate
    decay and lower shares slow it, with the speed bounded to ``[0.25, 4.0]``.

    Parameters
    ----------
    initial_distance_cutoff : float | None, default=None
        Explicit initial cutoff. ``None`` derives it from average bank distance.
    minimum_distance_cutoff : float | None, default=None
        Explicit minimum cutoff. ``None`` derives it from average bank distance.
    initial_distance_divisor : float, default=2.0
        Divisor used to derive the initial cutoff.
    minimum_distance_divisor : float, default=5.0
        Divisor used to derive the minimum cutoff.
    reduction_method : CSAReductionMethod, default="exponential"
        Reduction method. Local-route control requires ``"exponential"``.
    reduction_factor : float, default=0.983912
        Fixed annealing factor raised to the bounded reduction speed.
    stagnation_update_limit : int, default=10
        Maximum unused-entry count that still triggers cycle increment.
    cycle_increment_requires_minimum_cutoff : bool, default=False
        Whether cycle increments require the cutoff to reach its minimum.
    recover_steps : int, default=0
        Number of inverse decay steps applied during recovery.
    recover_mode : CSARecoverMode, default="none"
        Score-gap condition that triggers recovery.
    target_local_route_fraction : float, default=0.25
        Desired local share among current full-bank transition routes.
    response : float, default=2.0
        Positive sensitivity of annealing speed to local-route error.
    """

    reduction_factor: float = 0.983912
    stagnation_update_limit: int = 10
    cycle_increment_requires_minimum_cutoff: bool = False
    target_local_route_fraction: float = 0.25
    response: float = 2.0

    @override
    def __post_init__(self) -> None:
        """Validate the fixed backbone and local-route controller."""
        super(CSALocalRouteCutoffSchedule, self).__post_init__()
        if self.reduction_method != "exponential":
            msg = "local-route cutoff control requires exponential reduction"
            raise ValueError(msg)

        target_fraction = self.target_local_route_fraction
        if type(target_fraction) is bool or not isinstance(target_fraction, Real):
            msg = "target_local_route_fraction must be numeric"
            raise TypeError(msg)
        normalized_target = float(target_fraction)
        if not isfinite(normalized_target) or not 0.0 < normalized_target < 1.0:
            msg = "target_local_route_fraction must be finite and in (0.0, 1.0)"
            raise ValueError(msg)

        response = self.response
        if type(response) is bool or not isinstance(response, Real):
            msg = "response must be numeric"
            raise TypeError(msg)
        normalized_response = float(response)
        if not isfinite(normalized_response) or normalized_response <= 0.0:
            msg = "response must be finite and positive"
            raise ValueError(msg)

        object.__setattr__(self, "target_local_route_fraction", normalized_target)
        object.__setattr__(self, "response", normalized_response)

    @override
    def resolve_reduction_speed(
        self,
        *,
        observation: CSACutoffObservation,
    ) -> float:
        """Return bounded annealing speed from current local-route evidence.

        Parameters
        ----------
        observation : CSACutoffObservation
            Canonical post-update cutoff observation.

        Returns
        -------
        float
            Reduction speed in ``[0.25, 4.0]``. Missing route evidence returns
            exactly ``1.0``.
        """
        local_route_fraction = observation.local_route_fraction
        if local_route_fraction is None:
            return 1.0

        exponent = self.response * (
            local_route_fraction - self.target_local_route_fraction
        )
        if exponent <= _MINIMUM_REDUCTION_EXPONENT:
            return _MINIMUM_REDUCTION_SPEED
        if exponent >= _MAXIMUM_REDUCTION_EXPONENT:
            return _MAXIMUM_REDUCTION_SPEED
        return exp(exponent)

    @property
    @override
    def requires_reduction_observation(self) -> bool:
        """Return that local-route control needs transition evidence.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True
