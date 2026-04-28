"""Boundary-level profile for the clearing genetic algorithm."""

from dataclasses import dataclass

from typing_extensions import override

from ...profile import AlgorithmProfile


@dataclass(frozen=True, slots=True)
class ClearingGAResolvedProfile:
    """Canonical clearing-GA configuration used by optimizer internals.

    Parameters
    ----------
    tournament_size : int
        Number of members sampled in each tournament selection step.
    crossover_probability : float
        Probability of applying crossover when generating a child.
    mutation_probability : float
        Probability of mutating a generated child.
    clearing_radius : float
        Diversity radius below which two individuals are treated as occupying
        the same niche.
    clearing_capacity : int
        Maximum number of survivors allowed inside each clearing niche.
    """

    tournament_size: int
    crossover_probability: float
    mutation_probability: float
    clearing_radius: float
    clearing_capacity: int


@dataclass(frozen=True, slots=True)
class ClearingGAProfile(AlgorithmProfile[ClearingGAResolvedProfile]):
    """Boundary-level configuration for the clearing genetic algorithm.

    Parameters
    ----------
    tournament_size : int, default=2
        Number of population members compared in each tournament.
    crossover_probability : float, default=0.9
        Probability of applying crossover during child generation.
    mutation_probability : float, default=0.1
        Probability of mutating a generated child.
    clearing_radius : float, default=0.25
        Diversity radius used to determine niche occupancy.
    clearing_capacity : int, default=2
        Maximum number of survivors retained in each niche.
    """

    tournament_size: int = 2
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    clearing_radius: float = 0.25
    clearing_capacity: int = 2

    def __post_init__(self) -> None:
        """Validate clearing-GA profile fields.

        Raises
        ------
        ValueError
            Raised when any probability, radius, or capacity falls outside the
            supported range.
        """
        if self.tournament_size <= 0:
            msg = "tournament_size must be positive"
            raise ValueError(msg)

        if not 0.0 <= self.crossover_probability <= 1.0:
            msg = "crossover_probability must be between 0.0 and 1.0"
            raise ValueError(msg)

        if not 0.0 <= self.mutation_probability <= 1.0:
            msg = "mutation_probability must be between 0.0 and 1.0"
            raise ValueError(msg)

        if self.clearing_radius < 0.0:
            msg = "clearing_radius must be non-negative"
            raise ValueError(msg)

        if self.clearing_capacity <= 0:
            msg = "clearing_capacity must be positive"
            raise ValueError(msg)

    @override
    def resolve(self) -> ClearingGAResolvedProfile:
        """Materialize the canonical clearing-GA configuration.

        Returns
        -------
        ClearingGAResolvedProfile
            Immutable optimizer-ready configuration derived from the boundary
            profile.
        """
        return ClearingGAResolvedProfile(
            tournament_size=self.tournament_size,
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability,
            clearing_radius=self.clearing_radius,
            clearing_capacity=self.clearing_capacity,
        )
