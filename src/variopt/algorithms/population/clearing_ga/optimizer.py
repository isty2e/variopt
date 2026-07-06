"""Clearing genetic algorithm optimizer."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Observation, Proposal
from ....diversity import DiversityMetric
from ....execution import (
    ExecutionModel,
)
from ....methods import RunMethod
from ....operators import VariationOperator
from ....randomness import RandomSeed, RandomStateSnapshot
from ....sampling import CandidateSampler, SearchSpaceSampler
from ....spaces import PermutationSpace, SearchSpace
from ....typevars import CandidateT
from ..generational_ga.lifecycle import (
    GENERATIONAL_GA_EXECUTION_MODELS,
    GenerationalGAGenerationCommit,
    ask_generational_ga,
    create_initial_generational_ga_state,
    sort_generational_ga_population,
    tell_generational_ga,
)
from ..generational_ga.state import (
    GenerationalGAOptimizerState,
    GenerationalGAPopulationMember,
    GenerationalGAVariant,
)
from ..permutation.defaults import derive_permutation_variation_defaults
from .profile import ClearingGAProfile, ClearingGAResolvedProfile

BoundaryT = TypeVar("BoundaryT")


@dataclass(frozen=True, slots=True)
class ClearingGeneticAlgorithmOptimizer(FrozenGenericSlotsCompat,
    RunMethod[
        GenerationalGAOptimizerState[CandidateT],
        Proposal[CandidateT],
        Observation[CandidateT],
    ],
    Generic[BoundaryT, CandidateT],
):
    """Single-objective GA with clearing-based survival.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, CandidateT]
        Search space used to validate and sample candidates.
    population_size : int
        Number of members maintained in each generation.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to decide niche occupancy during clearing.
    crossover_operator : VariationOperator[CandidateT] | None, optional
        Optional crossover operator applied before mutation.
    mutation_operator : VariationOperator[CandidateT] | None, optional
        Optional mutation operator applied after crossover or cloning.
    profile : ClearingGAProfile, optional
        Boundary-level profile controlling tournaments and clearing behavior.
    sampler : CandidateSampler[CandidateT] | None, optional
        Optional sampler for initial population generation.
    random_state : RandomSeed, optional
        Seed or random-state object used to initialize optimizer randomness.
    """

    space: SearchSpace[BoundaryT, CandidateT]
    population_size: int
    diversity_metric: DiversityMetric[CandidateT]
    crossover_operator: VariationOperator[CandidateT] | None = field(default=None, kw_only=True)
    mutation_operator: VariationOperator[CandidateT] | None = field(default=None, kw_only=True)
    profile: ClearingGAProfile = field(default_factory=ClearingGAProfile, kw_only=True)
    sampler: CandidateSampler[CandidateT] | None = field(default=None, kw_only=True)
    random_state: RandomSeed = None
    resolved_profile: ClearingGAResolvedProfile = field(init=False, repr=False)
    resolved_sampler: CandidateSampler[CandidateT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve and validate the canonical clearing-GA configuration.

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

    @staticmethod
    def from_permutation_space_defaults(
        *,
        space: PermutationSpace,
        population_size: int,
        profile: ClearingGAProfile | None = None,
        sampler: CandidateSampler[tuple[int, ...]] | None = None,
        diversity_metric: DiversityMetric[tuple[int, ...]] | None = None,
        crossover_operator: VariationOperator[tuple[int, ...]] | None = None,
        mutation_operator: VariationOperator[tuple[int, ...]] | None = None,
        random_state: RandomSeed = None,
    ) -> "ClearingGeneticAlgorithmOptimizer[Sequence[int], tuple[int, ...]]":
        """Build a permutation-specialized clearing GA with safe defaults.

        Parameters
        ----------
        space : PermutationSpace
            Permutation search space optimized by the returned GA.
        population_size : int
            Number of members maintained in each generation.
        profile : ClearingGAProfile | None, default=None
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
        ClearingGeneticAlgorithmOptimizer[Sequence[int], tuple[int, ...]]
            Clearing GA configured for permutation-safe defaults.
        """
        defaults = derive_permutation_variation_defaults(space)
        return ClearingGeneticAlgorithmOptimizer(
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
            profile=ClearingGAProfile() if profile is None else profile,
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
            variant=GenerationalGAVariant.CLEARING,
        )

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        """Return supported execution models.

        Returns
        -------
        frozenset[ExecutionModel]
            Execution models accepted by the clearing-GA ask/tell contract.
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
        _ = state
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
            proposal_id_prefix="clearing-ga-",
            variant=GenerationalGAVariant.CLEARING,
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
            variant=GenerationalGAVariant.CLEARING,
        )

    def _build_next_population(
        self,
        *,
        parents: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        offspring: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        random_state: RandomStateSnapshot,
    ) -> GenerationalGAGenerationCommit[CandidateT]:
        _ = random_state
        candidate_pool = sort_generational_ga_population(parents + offspring)
        selected_members: list[GenerationalGAPopulationMember[CandidateT]] = []
        cleared_members: list[GenerationalGAPopulationMember[CandidateT]] = []
        for member in candidate_pool:
            if self._clearing_occupancy(member, selected_members) < self.resolved_profile.clearing_capacity:
                selected_members.append(member)
            else:
                cleared_members.append(member)

        if len(selected_members) >= self.population_size:
            return GenerationalGAGenerationCommit(
                population=tuple(selected_members[: self.population_size]),
            )

        next_population = tuple(selected_members) + self._build_diverse_backfill(
            selected_members=tuple(selected_members),
            overflow_members=tuple(cleared_members),
            count=self.population_size - len(selected_members),
        )
        return GenerationalGAGenerationCommit(
            population=sort_generational_ga_population(next_population),
        )

    def _clearing_occupancy(
        self,
        member: GenerationalGAPopulationMember[CandidateT],
        selected_members: Sequence[GenerationalGAPopulationMember[CandidateT]],
    ) -> int:
        return sum(
            1
            for selected_member in selected_members
            if self.diversity_metric.distance(
                member.candidate,
                selected_member.candidate,
            ) < self.resolved_profile.clearing_radius
        )

    def _build_diverse_backfill(
        self,
        *,
        selected_members: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        overflow_members: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        count: int,
    ) -> tuple[GenerationalGAPopulationMember[CandidateT], ...]:
        if count <= 0 or len(overflow_members) == 0:
            return ()

        chosen_indices: list[int] = []
        remaining_indices = list(range(len(overflow_members)))
        minimum_distances = [
            self._minimum_distance_to_population(member, selected_members)
            for member in overflow_members
        ]

        while len(chosen_indices) < count and len(remaining_indices) > 0:
            next_index = max(
                remaining_indices,
                key=lambda index: (
                    minimum_distances[index],
                    -overflow_members[index].score,
                ),
            )
            chosen_indices.append(next_index)
            remaining_indices.remove(next_index)

            if len(chosen_indices) >= count or len(remaining_indices) == 0:
                continue

            new_anchor = overflow_members[next_index]
            for index in remaining_indices:
                distance_to_anchor = self.diversity_metric.distance(
                    overflow_members[index].candidate,
                    new_anchor.candidate,
                )
                minimum_distances[index] = min(
                    minimum_distances[index],
                    distance_to_anchor,
                )

        return tuple(overflow_members[index] for index in chosen_indices)

    def _minimum_distance_to_population(
        self,
        member: GenerationalGAPopulationMember[CandidateT],
        population: Sequence[GenerationalGAPopulationMember[CandidateT]],
    ) -> float:
        if len(population) == 0:
            return float("inf")

        return min(
            self.diversity_metric.distance(
                member.candidate,
                other_member.candidate,
            )
            for other_member in population
        )
