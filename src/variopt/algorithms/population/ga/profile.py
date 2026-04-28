"""Boundary-level profile for the native genetic algorithm optimizer."""

from dataclasses import dataclass

from typing_extensions import override

from ...profile import AlgorithmProfile


@dataclass(frozen=True, slots=True)
class GAResolvedProfile:
    """Canonical genetic-algorithm configuration used by optimizer internals.

    Parameters
    ----------
    tournament_size : int
        Number of population members sampled for each tournament selection.
    crossover_probability : float
        Probability of applying the configured crossover operator when
        generating a child.
    mutation_probability : float
        Probability of applying the configured mutation operator after
        crossover or cloning.
    elite_count : int
        Number of best-scoring parents copied directly into the next
        generation.
    """

    tournament_size: int
    crossover_probability: float
    mutation_probability: float
    elite_count: int


@dataclass(frozen=True, slots=True)
class GAProfile(AlgorithmProfile[GAResolvedProfile]):
    """Boundary-level genetic-algorithm configuration.

    Parameters
    ----------
    tournament_size : int, default=2
        Number of candidates drawn for each tournament selection step.
    crossover_probability : float, default=0.9
        Probability of applying crossover when producing a child.
    mutation_probability : float, default=0.1
        Probability of mutating a child after crossover or cloning.
    elite_count : int, default=1
        Number of top parents preserved verbatim between generations.
    """

    tournament_size: int = 2
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    elite_count: int = 1

    def __post_init__(self) -> None:
        """Validate boundary-level GA profile fields.

        Raises
        ------
        ValueError
            Raised when any configured probability, tournament size, or elite
            count falls outside the supported range.
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

        if self.elite_count < 0:
            msg = "elite_count must be non-negative"
            raise ValueError(msg)

    @override
    def resolve(self) -> GAResolvedProfile:
        """Materialize the canonical GA configuration.

        Returns
        -------
        GAResolvedProfile
            Immutable optimizer-ready configuration derived from the boundary
            profile.
        """
        return GAResolvedProfile(
            tournament_size=self.tournament_size,
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability,
            elite_count=self.elite_count,
        )
