"""Boundary-level profile for the species-conserving genetic algorithm."""

from dataclasses import dataclass

from typing_extensions import override

from ...profile import AlgorithmProfile


@dataclass(frozen=True, slots=True)
class SpeciesGAResolvedProfile:
    """Canonical species-GA configuration used by optimizer internals.

    Parameters
    ----------
    tournament_size : int
        Number of members sampled in each parent-selection tournament.
    crossover_probability : float
        Probability of applying crossover during child generation.
    mutation_probability : float
        Probability of mutating a generated child.
    species_radius : float
        Diversity radius used to decide whether candidates belong to the same
        protected species.
    species_capacity : int
        Maximum number of survivors retained for each discovered species.
    """

    tournament_size: int
    crossover_probability: float
    mutation_probability: float
    species_radius: float
    species_capacity: int


@dataclass(frozen=True, slots=True)
class SpeciesGAProfile(AlgorithmProfile[SpeciesGAResolvedProfile]):
    """Boundary-level configuration for the species-conserving GA.

    Parameters
    ----------
    tournament_size : int, default=2
        Number of members compared in each parent-selection tournament.
    crossover_probability : float, default=0.9
        Probability of applying crossover during child generation.
    mutation_probability : float, default=0.1
        Probability of mutating a generated child.
    species_radius : float, default=0.25
        Diversity radius used to group candidates into species.
    species_capacity : int, default=1
        Maximum number of survivors retained per species.
    """

    tournament_size: int = 2
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    species_radius: float = 0.25
    species_capacity: int = 1

    def __post_init__(self) -> None:
        """Validate species-GA profile fields.

        Raises
        ------
        ValueError
            Raised when any probability, radius, or capacity is invalid.
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

        if self.species_radius < 0.0:
            msg = "species_radius must be non-negative"
            raise ValueError(msg)

        if self.species_capacity <= 0:
            msg = "species_capacity must be positive"
            raise ValueError(msg)

    @override
    def resolve(self) -> SpeciesGAResolvedProfile:
        """Materialize the canonical species-GA configuration.

        Returns
        -------
        SpeciesGAResolvedProfile
            Immutable optimizer-ready configuration derived from the boundary
            profile.
        """
        return SpeciesGAResolvedProfile(
            tournament_size=self.tournament_size,
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability,
            species_radius=self.species_radius,
            species_capacity=self.species_capacity,
        )
