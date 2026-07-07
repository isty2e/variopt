"""Restricted-tournament genetic algorithm optimizer."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Observation, Proposal
from ....diversity import DiversityMetric
from ....execution import (
    ExecutionModel,
)
from ....methods import RunMethod
from ....operators import VariationOperator
from ....randomness import (
    RandomSeed,
    RandomStateSnapshot,
    random_state_choice_indices_without_replacement,
)
from ....sampling import CandidateSampler, SearchSpaceSampler
from ....spaces import PermutationSpace, SearchSpace
from ....typevars import CandidateT
from ..generational_ga.lifecycle import (
    GENERATIONAL_GA_EXECUTION_MODELS,
    GenerationalGAGenerationCommit,
    ask_generational_ga,
    create_initial_generational_ga_state,
    require_generational_ga_variant_state,
    sort_generational_ga_population,
    tell_generational_ga,
)
from ..generational_ga.state import (
    GenerationalGAOptimizerState,
    GenerationalGAPopulationMember,
    GenerationalGAVariant,
)
from ..permutation.defaults import derive_permutation_variation_defaults
from .profile import (
    RestrictedTournamentGAProfile,
    RestrictedTournamentGAResolvedProfile,
)

BoundaryT = TypeVar("BoundaryT")


@dataclass(frozen=True, slots=True)
class RestrictedTournamentGeneticAlgorithmOptimizer(
    FrozenGenericSlotsCompat,
    RunMethod[
        GenerationalGAOptimizerState[CandidateT],
        Proposal[CandidateT],
        Observation[CandidateT],
    ],
    Generic[BoundaryT, CandidateT],
):
    """Single-objective GA with restricted-tournament replacement.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, CandidateT]
        Search space used to validate and sample candidates.
    population_size : int
        Number of members maintained in each generation.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to choose restricted-tournament competitors.
    crossover_operator : VariationOperator[CandidateT] | None, optional
        Optional crossover operator applied before mutation.
    mutation_operator : VariationOperator[CandidateT] | None, optional
        Optional mutation operator applied after crossover or cloning.
    profile : RestrictedTournamentGAProfile, optional
        Boundary-level profile controlling tournaments and replacement windows.
    sampler : CandidateSampler[CandidateT] | None, optional
        Optional sampler for initial population generation.
    random_state : RandomSeed, optional
        Seed or random-state object used to initialize optimizer randomness.
    """

    space: SearchSpace[BoundaryT, CandidateT]
    population_size: int
    diversity_metric: DiversityMetric[CandidateT]
    crossover_operator: VariationOperator[CandidateT] | None = field(
        default=None, kw_only=True
    )
    mutation_operator: VariationOperator[CandidateT] | None = field(
        default=None, kw_only=True
    )
    profile: RestrictedTournamentGAProfile = field(
        default_factory=RestrictedTournamentGAProfile, kw_only=True
    )
    sampler: CandidateSampler[CandidateT] | None = field(default=None, kw_only=True)
    random_state: RandomSeed = None
    resolved_profile: RestrictedTournamentGAResolvedProfile = field(
        init=False, repr=False
    )
    resolved_sampler: CandidateSampler[CandidateT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve and validate the canonical restricted-tournament configuration.

        Raises
        ------
        ValueError
            Raised when population size, operator arities, or profile settings
            are inconsistent.
        """
        if self.population_size <= 0:
            msg = "population_size must be positive"
            raise ValueError(msg)

        resolved_profile = self.profile.resolve()
        object.__setattr__(self, "resolved_profile", resolved_profile)
        object.__setattr__(
            self,
            "resolved_sampler",
            SearchSpaceSampler(self.space) if self.sampler is None else self.sampler,
        )

        if self.crossover_operator is None and self.mutation_operator is None:
            msg = "at least one variation operator must be provided"
            raise ValueError(msg)

        if self.crossover_operator is not None and self.crossover_operator.arity < 2:
            msg = "crossover_operator arity must be at least 2"
            raise ValueError(msg)

        if self.mutation_operator is not None and self.mutation_operator.arity != 1:
            msg = "mutation_operator arity must be exactly 1"
            raise ValueError(msg)

        if resolved_profile.tournament_size > self.population_size:
            msg = "tournament_size must not exceed population_size"
            raise ValueError(msg)

        if resolved_profile.restricted_tournament_window_size > self.population_size:
            msg = "restricted_tournament_window_size must not exceed population_size"
            raise ValueError(msg)

    @staticmethod
    def from_permutation_space_defaults(
        *,
        space: PermutationSpace,
        population_size: int,
        profile: RestrictedTournamentGAProfile | None = None,
        sampler: CandidateSampler[tuple[int, ...]] | None = None,
        diversity_metric: DiversityMetric[tuple[int, ...]] | None = None,
        crossover_operator: VariationOperator[tuple[int, ...]] | None = None,
        mutation_operator: VariationOperator[tuple[int, ...]] | None = None,
        random_state: RandomSeed = None,
    ) -> (
        "RestrictedTournamentGeneticAlgorithmOptimizer[Sequence[int], tuple[int, ...]]"
    ):
        """Build a permutation-specialized restricted-tournament GA.

        Parameters
        ----------
        space : PermutationSpace
            Permutation search space optimized by the returned GA.
        population_size : int
            Number of members maintained in each generation.
        profile : RestrictedTournamentGAProfile | None, default=None
            Optional profile override.
        sampler : CandidateSampler[tuple[int, ...]] | None, default=None
            Optional initial-population sampler override.
        diversity_metric : DiversityMetric[tuple[int, ...]] | None, default=None
            Optional diversity metric override.
        crossover_operator : VariationOperator[tuple[int, ...]] | None, default=None
            Optional crossover operator override.
        mutation_operator : VariationOperator[tuple[int, ...]] | None, default=None
            Optional mutation operator override.
        random_state : RandomSeed, optional
            Seed or random-state object used to initialize optimizer
            randomness.

        Returns
        -------
        RestrictedTournamentGeneticAlgorithmOptimizer[Sequence[int], tuple[int, ...]]
            Restricted-tournament GA configured for permutation-safe defaults.
        """
        defaults = derive_permutation_variation_defaults(space)
        return RestrictedTournamentGeneticAlgorithmOptimizer(
            space=space,
            population_size=population_size,
            diversity_metric=(
                defaults.diversity_metric
                if diversity_metric is None
                else diversity_metric
            ),
            crossover_operator=(
                defaults.crossover_operator
                if crossover_operator is None
                else crossover_operator
            ),
            mutation_operator=(
                defaults.mutation_operator
                if mutation_operator is None
                else mutation_operator
            ),
            profile=RestrictedTournamentGAProfile() if profile is None else profile,
            sampler=sampler,
            random_state=random_state,
        )

    @override
    def create_initial_state(self) -> GenerationalGAOptimizerState[CandidateT]:
        """Create the initial immutable optimizer state.

        Returns
        -------
        GenerationalGAOptimizerState[CandidateT]
            State with initialized randomness and no observed population.
        """
        return create_initial_generational_ga_state(
            self.random_state,
            variant=GenerationalGAVariant.RESTRICTED_TOURNAMENT,
        )

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        """Return supported execution models.

        Returns
        -------
        frozenset[ExecutionModel]
            Execution models accepted by the restricted-tournament GA
            ask/tell contract.
        """
        return GENERATIONAL_GA_EXECUTION_MODELS

    @override
    def is_exhausted(self, state: GenerationalGAOptimizerState[CandidateT]) -> bool:
        """Report whether the optimizer can emit more proposals.

        Parameters
        ----------
        state : GenerationalGAOptimizerState[CandidateT]
            Optimizer state to inspect.

        Returns
        -------
        bool
            Always ``False`` for the current unbounded implementation.
        """
        require_generational_ga_variant_state(
            state,
            variant=GenerationalGAVariant.RESTRICTED_TOURNAMENT,
        )
        return False

    @override
    def ask(
        self,
        state: GenerationalGAOptimizerState[CandidateT],
        batch_size: int = 1,
    ) -> tuple[
        tuple[Proposal[CandidateT], ...],
        GenerationalGAOptimizerState[CandidateT],
    ]:
        """Emit the next proposal batch and advanced optimizer state.

        Parameters
        ----------
        state : GenerationalGAOptimizerState[CandidateT]
            Current immutable optimizer state.
        batch_size : int, default=1
            Maximum number of proposals to emit.

        Returns
        -------
        tuple[tuple[Proposal[CandidateT], ...], GenerationalGAOptimizerState[CandidateT]]
            Proposal batch together with the advanced immutable state.

        Raises
        ------
        ValueError
            Raised when ``batch_size`` is not positive.
        RuntimeError
            Raised when outstanding proposals have not yet been observed.
        """
        return ask_generational_ga(
            self,
            state,
            batch_size=batch_size,
            proposal_id_prefix="restricted-tournament-ga-",
            variant=GenerationalGAVariant.RESTRICTED_TOURNAMENT,
        )

    @override
    def tell(
        self,
        state: GenerationalGAOptimizerState[CandidateT],
        observations: Sequence[Observation[CandidateT]],
    ) -> GenerationalGAOptimizerState[CandidateT]:
        """Advance the optimizer state with one observed proposal batch.

        Parameters
        ----------
        state : GenerationalGAOptimizerState[CandidateT]
            Current immutable optimizer state.
        observations : Sequence[Observation[CandidateT]]
            Observations aligned with the currently pending proposals.

        Returns
        -------
        GenerationalGAOptimizerState[CandidateT]
            Updated immutable optimizer state after buffering or committing the
            observed members.

        Raises
        ------
        ValueError
            Raised when observation count or ordering does not match the
            pending proposals.
        """
        return tell_generational_ga(
            state,
            observations,
            population_size=self.population_size,
            build_next_population=self._build_next_population,
            variant=GenerationalGAVariant.RESTRICTED_TOURNAMENT,
        )

    def _build_next_population(
        self,
        *,
        parents: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        offspring: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        random_state: RandomStateSnapshot,
    ) -> GenerationalGAGenerationCommit[CandidateT]:
        materialized_random_state = random_state.materialize()
        next_population = list(sort_generational_ga_population(parents))
        for child_member in offspring:
            competitor_index = self._restricted_tournament_competitor_index(
                population=next_population,
                child_member=child_member,
                random_state=materialized_random_state,
            )
            competitor = next_population[competitor_index]
            if child_member.score <= competitor.score:
                next_population[competitor_index] = child_member

        return GenerationalGAGenerationCommit(
            population=sort_generational_ga_population(tuple(next_population)),
            random_state=RandomStateSnapshot.from_random_state(
                materialized_random_state
            ),
        )

    def _restricted_tournament_competitor_index(
        self,
        *,
        population: Sequence[GenerationalGAPopulationMember[CandidateT]],
        child_member: GenerationalGAPopulationMember[CandidateT],
        random_state: np.random.RandomState,
    ) -> int:
        window_indices = random_state_choice_indices_without_replacement(
            random_state,
            population_size=len(population),
            count=self.resolved_profile.restricted_tournament_window_size,
        )
        return min(
            window_indices,
            key=lambda index: (
                self.diversity_metric.distance(
                    child_member.candidate,
                    population[index].candidate,
                ),
                population[index].score,
            ),
        )
