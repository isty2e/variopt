"""Boundary-level profile for the native differential-evolution optimizer."""

from dataclasses import dataclass

from typing_extensions import override

from ...profile import AlgorithmProfile


@dataclass(frozen=True, slots=True)
class DEResolvedProfile:
    """Canonical differential-evolution configuration used by optimizer internals.

    Parameters
    ----------
    mutation_range : tuple[float, float]
        Inclusive range from which mutation factors are sampled.
    recombination_probability : float
        Probability of copying each candidate leaf from the mutant during
        crossover.
    n_cross : int
        Minimum number of leaves forced to cross over into each trial.
    """

    mutation_range: tuple[float, float]
    recombination_probability: float
    n_cross: int


@dataclass(frozen=True, slots=True)
class DEProfile(AlgorithmProfile[DEResolvedProfile]):
    """Boundary-level differential-evolution configuration.

    Parameters
    ----------
    mutation_range : tuple[float, float], default=(0.5, 1.0)
        Inclusive interval used to sample mutation factors.
    recombination_probability : float, default=0.7
        Probability of copying a mutated leaf into the trial candidate.
    n_cross : int, default=1
        Minimum number of leaves that must cross over into each trial.
    """

    mutation_range: tuple[float, float] = (0.5, 1.0)
    recombination_probability: float = 0.7
    n_cross: int = 1

    def __post_init__(self) -> None:
        """Validate boundary-level DE profile fields.

        Raises
        ------
        ValueError
            Raised when the mutation range, crossover probability, or forced
            crossover count is invalid.
        """
        mutation_low, mutation_high = self.mutation_range
        if mutation_low > mutation_high:
            msg = "mutation_range low must not exceed mutation_range high"
            raise ValueError(msg)

        if mutation_low < 0.0:
            msg = "mutation_range low must be non-negative"
            raise ValueError(msg)

        if not 0.0 <= self.recombination_probability <= 1.0:
            msg = "recombination_probability must be between 0.0 and 1.0"
            raise ValueError(msg)

        if self.n_cross <= 0:
            msg = "n_cross must be positive"
            raise ValueError(msg)

    @override
    def resolve(self) -> DEResolvedProfile:
        """Materialize the canonical DE configuration.

        Returns
        -------
        DEResolvedProfile
            Immutable optimizer-ready configuration derived from the boundary
            profile.
        """
        return DEResolvedProfile(
            mutation_range=self.mutation_range,
            recombination_probability=self.recombination_probability,
            n_cross=self.n_cross,
        )
