"""Public CSA score-model configuration objects."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Generic, Literal

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....typevars import CandidateT

CSABiasedSigmaReference = Literal[
    "distance_cutoff",
    "constant",
    "minimum_distance_cutoff",
]


@dataclass(frozen=True, slots=True)
class CSABiasedPotential:
    """Configuration for biased-potential score shaping.

    Parameters
    ----------
    maximum_bias : float | None, default=1.0e6
        Maximum additive bias applied by the biased potential. ``None`` means
        infer the scale from observed score spread.
    sigma : float, default=0.3
        Relative width of the biased-potential kernel.
    sigma_reference : CSABiasedSigmaReference, default="distance_cutoff"
        Runtime distance scale used to interpret ``sigma``.
    """

    maximum_bias: float | None = 1.0e6
    sigma: float = 0.3
    sigma_reference: CSABiasedSigmaReference = "distance_cutoff"

    def __post_init__(self) -> None:
        """Reject invalid biased-potential configuration."""
        if self.maximum_bias is not None and (
            not isfinite(self.maximum_bias) or self.maximum_bias < 0.0
        ):
            msg = "maximum_bias must be a finite non-negative float or None"
            raise ValueError(msg)

        if not isfinite(self.sigma) or self.sigma <= 0.0:
            msg = "sigma must be a finite positive float"
            raise ValueError(msg)

        if self.sigma_reference not in {
            "distance_cutoff",
            "constant",
            "minimum_distance_cutoff",
        }:
            msg = (
                "sigma_reference must be one of "
                "'distance_cutoff', 'constant', or 'minimum_distance_cutoff'"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CSAAdaptivePotentialAxis(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """One adaptive-potential axis anchored at a reference candidate.

    Parameters
    ----------
    reference_candidate : CandidateT
        Candidate used as the distance anchor for the axis.
    minimum_distance : float
        Inclusive lower distance bound.
    maximum_distance : float
        Exclusive upper distance bound.
    bin_count : int
        Number of equal-width bins on the axis.
    """

    reference_candidate: CandidateT
    minimum_distance: float
    maximum_distance: float
    bin_count: int

    def __post_init__(self) -> None:
        """Reject invalid adaptive-potential axis definitions."""
        if not isfinite(self.minimum_distance) or self.minimum_distance < 0.0:
            msg = "minimum_distance must be a finite non-negative float"
            raise ValueError(msg)

        if (
            not isfinite(self.maximum_distance)
            or self.maximum_distance <= self.minimum_distance
        ):
            msg = (
                "maximum_distance must be a finite float greater than minimum_distance"
            )
            raise ValueError(msg)

        if self.bin_count <= 0:
            msg = "bin_count must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CSAAdaptivePotential(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Configuration for adaptive-potential score shaping.

    Parameters
    ----------
    axes : tuple[CSAAdaptivePotentialAxis[CandidateT], ...]
        Adaptive-potential axes used to discretize candidate distances.
    increment : float, default=0.1
        Energy increment applied to the visited bin.
    overflow_energy : float, default=1.0e5
        Energy returned when a candidate falls outside the configured grid.
    """

    axes: tuple[CSAAdaptivePotentialAxis[CandidateT], ...]
    increment: float = 0.1
    overflow_energy: float = 1.0e5

    def __post_init__(self) -> None:
        """Reject invalid adaptive-potential configuration."""
        if len(self.axes) == 0:
            msg = "axes must not be empty"
            raise ValueError(msg)

        if not isfinite(self.increment) or self.increment < 0.0:
            msg = "increment must be a finite non-negative float"
            raise ValueError(msg)

        if not isfinite(self.overflow_energy) or self.overflow_energy < 0.0:
            msg = "overflow_energy must be a finite non-negative float"
            raise ValueError(msg)

    @classmethod
    def from_sequence(
        cls,
        axes: Sequence[CSAAdaptivePotentialAxis[CandidateT]],
        *,
        increment: float = 0.1,
        overflow_energy: float = 1.0e5,
    ) -> Self:
        """Build an adaptive-potential model from a general sequence of axes.

        Parameters
        ----------
        axes : Sequence[CSAAdaptivePotentialAxis[CandidateT]]
            Adaptive-potential axes to materialize.
        increment : float, default=0.1
            Energy increment applied to visited bins.
        overflow_energy : float, default=1.0e5
            Energy returned when candidates fall outside the grid.

        Returns
        -------
        Self
            Materialized adaptive-potential model.
        """
        return cls(
            axes=tuple(axes),
            increment=increment,
            overflow_energy=overflow_energy,
        )


@dataclass(frozen=True, slots=True)
class CSAScoreModel(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """CSA-specific score-shaping configuration.

    Parameters
    ----------
    biased_potential : CSABiasedPotential | None, default=None
        Optional biased-potential configuration.
    adaptive_potential : CSAAdaptivePotential[CandidateT] | None, default=None
        Optional adaptive-potential configuration.
    """

    biased_potential: CSABiasedPotential | None = None
    adaptive_potential: CSAAdaptivePotential[CandidateT] | None = None

    @property
    def has_biased_potential(self) -> bool:
        """Return whether biased-potential shaping is enabled."""
        return self.biased_potential is not None

    @property
    def has_adaptive_potential(self) -> bool:
        """Return whether adaptive-potential shaping is enabled."""
        return self.adaptive_potential is not None


@dataclass(frozen=True, slots=True)
class CSAScoreModelDefaults:
    """Candidate-free defaults for building one CSA score model.

    Parameters
    ----------
    biased_potential : CSABiasedPotential | None, default=None
        Default biased-potential configuration.
    """

    biased_potential: CSABiasedPotential | None = None
