"""Native single-objective genetic algorithm optimizer."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Observation, Proposal
from ....execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
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
from ..permutation.defaults import derive_permutation_variation_defaults
from .profile import GAProfile, GAResolvedProfile
from .state import GAOptimizerState, GAPopulationMember

BoundaryT = TypeVar("BoundaryT")


@dataclass(frozen=True, slots=True)
class GeneticAlgorithmOptimizer(FrozenGenericSlotsCompat,
    RunMethod[
        GAOptimizerState[CandidateT],
        Proposal[CandidateT],
        Observation[CandidateT],
    ],
    Generic[BoundaryT, CandidateT],
):
    """Stateless native single-objective generational GA.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, CandidateT]
        Search space used to validate and sample candidates.
    population_size : int
        Number of members maintained in each generation.
    crossover_operator : VariationOperator[CandidateT] | None, optional
        Optional crossover operator applied before mutation.
    mutation_operator : VariationOperator[CandidateT] | None, optional
        Optional mutation operator applied after crossover or cloning.
    profile : GAProfile, optional
        Boundary-level GA configuration controlling tournaments and elitism.
    sampler : CandidateSampler[CandidateT] | None, optional
        Optional sampler for initial population generation.
    random_state : RandomSeed, optional
        Seed or random-state object used to initialize optimizer randomness.
    """

    space: SearchSpace[BoundaryT, CandidateT]
    population_size: int
    crossover_operator: VariationOperator[CandidateT] | None = field(default=None, kw_only=True)
    mutation_operator: VariationOperator[CandidateT] | None = field(default=None, kw_only=True)
    profile: GAProfile = field(default_factory=GAProfile, kw_only=True)
    sampler: CandidateSampler[CandidateT] | None = field(default=None, kw_only=True)
    random_state: RandomSeed = None
    resolved_profile: GAResolvedProfile = field(init=False, repr=False)
    resolved_sampler: CandidateSampler[CandidateT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve and validate the canonical GA configuration.

        Raises
        ------
        ValueError
            Raised when the population size, operator arities, or profile
            fields are inconsistent.
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

        if resolved_profile.elite_count > self.population_size:
            msg = "elite_count must not exceed population_size"
            raise ValueError(msg)

    @staticmethod
    def from_permutation_space_defaults(
        *,
        space: PermutationSpace,
        population_size: int,
        profile: GAProfile | None = None,
        sampler: CandidateSampler[tuple[int, ...]] | None = None,
        crossover_operator: VariationOperator[tuple[int, ...]] | None = None,
        mutation_operator: VariationOperator[tuple[int, ...]] | None = None,
        random_state: RandomSeed = None,
    ) -> "GeneticAlgorithmOptimizer[Sequence[int], tuple[int, ...]]":
        """Build a permutation-specialized GA with safe default operators.

        Parameters
        ----------
        space : PermutationSpace
            Permutation search space optimized by the returned GA.
        population_size : int
            Number of members maintained in each generation.
        profile : GAProfile | None, default=None
            Optional GA profile override.
        sampler : CandidateSampler[tuple[int, ...]] | None, default=None
            Optional initial-population sampler override.
        crossover_operator : VariationOperator[tuple[int, ...]] | None, default=None
            Optional crossover operator override.
        mutation_operator : VariationOperator[tuple[int, ...]] | None, default=None
            Optional mutation operator override.
        random_state : RandomSeed, optional
            Seed or random-state object used to initialize optimizer
            randomness.

        Returns
        -------
        GeneticAlgorithmOptimizer[Sequence[int], tuple[int, ...]]
            Genetic algorithm configured for permutation-safe defaults.
        """
        defaults = derive_permutation_variation_defaults(space)
        return GeneticAlgorithmOptimizer(
            space=space,
            population_size=population_size,
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
            profile=GAProfile() if profile is None else profile,
            sampler=sampler,
            random_state=random_state,
        )

    @override
    def create_initial_state(self) -> GAOptimizerState[CandidateT]:
        """Create the initial immutable optimizer state.

        Returns
        -------
        GAOptimizerState[CandidateT]
            State with initialized randomness and no population yet observed.
        """
        return GAOptimizerState(
            random_state=RandomStateSnapshot.from_seed(self.random_state),
        )

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        """Return supported execution models.

        Returns
        -------
        frozenset[ExecutionModel]
            Execution models accepted by the GA ask/tell contract.
        """
        return frozenset(
            {
                SEQUENTIAL_EXECUTION_MODEL,
                SYNC_BATCH_EXECUTION_MODEL,
                EXACT_ASYNC_EXECUTION_MODEL,
            },
        )

    @override
    def is_exhausted(self, state: GAOptimizerState[CandidateT]) -> bool:
        """Report whether the optimizer can emit more proposals.

        Parameters
        ----------
        state : GAOptimizerState[CandidateT]
            Optimizer state to inspect.

        Returns
        -------
        bool
            Always ``False`` for the current unbounded GA implementation.
        """
        _ = state
        return False

    @override
    def ask(
        self,
        state: GAOptimizerState[CandidateT],
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[CandidateT], ...], GAOptimizerState[CandidateT]]:
        """Emit the next proposal batch and advanced GA state.

        Parameters
        ----------
        state : GAOptimizerState[CandidateT]
            Current immutable optimizer state.
        batch_size : int, default=1
            Maximum number of proposals to emit.

        Returns
        -------
        tuple[tuple[Proposal[CandidateT], ...], GAOptimizerState[CandidateT]]
            Proposal batch together with the advanced immutable state.

        Raises
        ------
        ValueError
            Raised when ``batch_size`` is not positive.
        RuntimeError
            Raised when outstanding proposals have not yet been observed.
        """
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)

        if len(state.pending_proposals) > 0:
            msg = "cannot ask while proposals are still pending"
            raise RuntimeError(msg)

        if len(state.population) == 0:
            return self._ask_initial_population(state, batch_size=batch_size)

        next_state = state
        if len(next_state.queued_proposals) == 0:
            next_state = self._materialize_generation(next_state)

        proposal_count = min(batch_size, len(next_state.queued_proposals))
        proposals = next_state.queued_proposals[:proposal_count]
        return proposals, replace(
            next_state,
            queued_proposals=next_state.queued_proposals[proposal_count:],
            pending_proposals=proposals,
        )

    def _ask_initial_population(
        self,
        state: GAOptimizerState[CandidateT],
        *,
        batch_size: int,
    ) -> tuple[tuple[Proposal[CandidateT], ...], GAOptimizerState[CandidateT]]:
        remaining_population = self.population_size - len(state.buffered_members)
        proposal_count = min(batch_size, remaining_population)
        random_state = state.random_state.materialize()
        candidates = tuple(
            self.resolved_sampler.sample(random_state)
            for _ in range(proposal_count)
        )
        next_random_state = RandomStateSnapshot.from_random_state(random_state)
        proposals = tuple(
            Proposal(
                candidate=self._validated_candidate(candidate),
                proposal_id=f"ga-{state.proposal_index + offset}",
            )
            for offset, candidate in enumerate(candidates)
        )
        return proposals, replace(
            state,
            random_state=next_random_state,
            proposal_index=state.proposal_index + len(proposals),
            pending_proposals=proposals,
        )

    def _materialize_generation(
        self,
        state: GAOptimizerState[CandidateT],
    ) -> GAOptimizerState[CandidateT]:
        random_state = state.random_state.materialize()
        proposals = tuple(
            Proposal(
                candidate=self._generate_child(
                    population=state.population,
                    random_state=random_state,
                ),
                proposal_id=f"ga-{state.proposal_index + offset}",
            )
            for offset in range(self.population_size)
        )
        next_random_state = RandomStateSnapshot.from_random_state(random_state)
        return replace(
            state,
            random_state=next_random_state,
            proposal_index=state.proposal_index + len(proposals),
            queued_proposals=proposals,
        )

    def _generate_child(
        self,
        *,
        population: tuple[GAPopulationMember[CandidateT], ...],
        random_state: np.random.RandomState,
    ) -> CandidateT:
        child_candidate: CandidateT
        crossover_operator = self.crossover_operator
        if (
            crossover_operator is not None
            and float(random_state.random_sample())
            < self.resolved_profile.crossover_probability
        ):
            parent_candidates = tuple(
                population[index].candidate
                for index in self._select_parent_indices(
                    random_state=random_state,
                    population=population,
                    count=crossover_operator.arity,
                )
            )
            child_candidate = crossover_operator.apply(parent_candidates, random_state)
        else:
            parent_index = self._select_parent_indices(
                random_state=random_state,
                population=population,
                count=1,
            )[0]
            child_candidate = population[parent_index].candidate

        mutation_operator = self.mutation_operator
        if (
            mutation_operator is not None
            and float(random_state.random_sample())
            < self.resolved_profile.mutation_probability
        ):
            child_candidate = mutation_operator.apply((child_candidate,), random_state)

        return self._validated_candidate(child_candidate)

    def _select_parent_indices(
        self,
        *,
        random_state: np.random.RandomState,
        population: tuple[GAPopulationMember[CandidateT], ...],
        count: int,
    ) -> tuple[int, ...]:
        return tuple(
            self._select_tournament_parent_index(
                random_state=random_state,
                population=population,
            )
            for _ in range(count)
        )

    def _select_tournament_parent_index(
        self,
        *,
        random_state: np.random.RandomState,
        population: tuple[GAPopulationMember[CandidateT], ...],
    ) -> int:
        tournament_indices = random_state_choice_indices_without_replacement(
            random_state,
            population_size=len(population),
            count=self.resolved_profile.tournament_size,
        )
        return min(
            tournament_indices,
            key=lambda index: population[index].score,
        )

    def _validated_candidate(self, candidate: CandidateT) -> CandidateT:
        self.space.validate(candidate)
        return candidate

    @override
    def tell(
        self,
        state: GAOptimizerState[CandidateT],
        observations: Sequence[Observation[CandidateT]],
    ) -> GAOptimizerState[CandidateT]:
        """Advance the GA state with one observed proposal batch.

        Parameters
        ----------
        state : GAOptimizerState[CandidateT]
            Current immutable optimizer state.
        observations : Sequence[Observation[CandidateT]]
            Observations aligned with the currently pending proposals.

        Returns
        -------
        GAOptimizerState[CandidateT]
            Updated immutable optimizer state after buffering or committing the
            observed members.

        Raises
        ------
        ValueError
            Raised when observation count or ordering does not match the
            pending proposals.
        """
        observation_tuple = tuple(observations)
        if len(observation_tuple) != len(state.pending_proposals):
            msg = "observation count must match the number of pending proposals"
            raise ValueError(msg)

        for proposal, observation in zip(
            state.pending_proposals,
            observation_tuple,
            strict=True,
        ):
            if observation.proposal != proposal:
                msg = "observations must align with pending proposal order"
                raise ValueError(msg)

        new_members = tuple(
            GAPopulationMember(
                candidate=observation.candidate,
                value=observation.value,
                score=observation.score,
            )
            for observation in observation_tuple
        )

        buffered_members = state.buffered_members + new_members
        next_state = replace(
            state,
            pending_proposals=(),
            buffered_members=buffered_members,
        )

        if len(state.population) == 0:
            return self._tell_initial_population(next_state)

        return self._tell_generation(next_state)

    def _tell_initial_population(
        self,
        state: GAOptimizerState[CandidateT],
    ) -> GAOptimizerState[CandidateT]:
        if len(state.buffered_members) < self.population_size:
            return state

        if len(state.buffered_members) != self.population_size:
            msg = "initial population buffer exceeded population_size"
            raise RuntimeError(msg)

        return replace(
            state,
            population=self._sort_population(state.buffered_members),
            buffered_members=(),
        )

    def _tell_generation(
        self,
        state: GAOptimizerState[CandidateT],
    ) -> GAOptimizerState[CandidateT]:
        if len(state.buffered_members) < self.population_size:
            return state

        if len(state.buffered_members) != self.population_size:
            msg = "offspring buffer exceeded population_size"
            raise RuntimeError(msg)

        next_population = self._build_next_population(
            parents=state.population,
            offspring=state.buffered_members,
        )
        return replace(
            state,
            generation_index=state.generation_index + 1,
            population=next_population,
            buffered_members=(),
        )

    def _build_next_population(
        self,
        *,
        parents: tuple[GAPopulationMember[CandidateT], ...],
        offspring: tuple[GAPopulationMember[CandidateT], ...],
    ) -> tuple[GAPopulationMember[CandidateT], ...]:
        elite_count = self.resolved_profile.elite_count
        elite_members = self._sort_population(parents)[:elite_count]
        offspring_members = self._sort_population(offspring)[
            : self.population_size - elite_count
        ]
        return self._sort_population(elite_members + offspring_members)

    @staticmethod
    def _sort_population(
        members: tuple[GAPopulationMember[CandidateT], ...],
    ) -> tuple[GAPopulationMember[CandidateT], ...]:
        return tuple(
            sorted(
                members,
                key=lambda member: member.score,
            )
        )
