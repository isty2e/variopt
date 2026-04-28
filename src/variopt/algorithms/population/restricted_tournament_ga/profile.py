"""Boundary-level profile for the restricted-tournament genetic algorithm."""

from dataclasses import dataclass

from typing_extensions import override

from ...profile import AlgorithmProfile


@dataclass(frozen=True, slots=True)
class RestrictedTournamentGAResolvedProfile:
    """Canonical restricted-tournament GA configuration used by internals.

    Parameters
    ----------
    tournament_size : int
        Number of members sampled in each parent-selection tournament.
    crossover_probability : float
        Probability of applying crossover during child generation.
    mutation_probability : float
        Probability of mutating a generated child.
    restricted_tournament_window_size : int
        Number of incumbents sampled when selecting the restricted-tournament
        replacement competitor.
    """

    tournament_size: int
    crossover_probability: float
    mutation_probability: float
    restricted_tournament_window_size: int


@dataclass(frozen=True, slots=True)
class RestrictedTournamentGAProfile(AlgorithmProfile[RestrictedTournamentGAResolvedProfile]):
    """Boundary-level configuration for the restricted-tournament GA.

    Parameters
    ----------
    tournament_size : int, default=2
        Number of members compared in each parent-selection tournament.
    crossover_probability : float, default=0.9
        Probability of applying crossover during child generation.
    mutation_probability : float, default=0.1
        Probability of mutating a generated child.
    restricted_tournament_window_size : int, default=5
        Number of incumbents sampled when choosing the replacement competitor.
    """

    tournament_size: int = 2
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    restricted_tournament_window_size: int = 5

    def __post_init__(self) -> None:
        """Validate restricted-tournament GA profile fields.

        Raises
        ------
        ValueError
            Raised when any probability, tournament size, or window size is
            invalid.
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

        if self.restricted_tournament_window_size <= 0:
            msg = "restricted_tournament_window_size must be positive"
            raise ValueError(msg)

    @override
    def resolve(self) -> RestrictedTournamentGAResolvedProfile:
        """Materialize the canonical restricted-tournament configuration.

        Returns
        -------
        RestrictedTournamentGAResolvedProfile
            Immutable optimizer-ready configuration derived from the boundary
            profile.
        """
        return RestrictedTournamentGAResolvedProfile(
            tournament_size=self.tournament_size,
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability,
            restricted_tournament_window_size=self.restricted_tournament_window_size,
        )
