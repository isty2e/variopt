"""Native single-objective differential-evolution optimizer."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar, cast

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
from ....randomness import (
    RandomSeed,
    RandomStateSnapshot,
    random_state_choice_indices_without_replacement,
)
from ....sampling import CandidateSampler, SearchSpaceSampler
from ....spaces import SearchSpace, StructuredSearchSpace
from ....spaces.types import SpaceCandidateValue
from .profile import DEProfile, DEResolvedProfile
from .state import (
    DEObservedEvaluation,
    DEOptimizerState,
    DEPendingEvaluation,
    DEPopulationMember,
)
from .variation import (
    differential_evolution_variation,
    require_numeric_structured_space,
)

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)


@dataclass(frozen=True, slots=True)
class DifferentialEvolutionOptimizer(FrozenGenericSlotsCompat,
    RunMethod[
        DEOptimizerState[StructuredCandidateT],
        Proposal[StructuredCandidateT],
        Observation[StructuredCandidateT],
    ],
    Generic[BoundaryT, StructuredCandidateT],
):
    """Stateless native single-objective generational differential evolution.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space optimized by the returned DE instance.
    population_size : int
        Number of members maintained in each generation.
    profile : DEProfile, optional
        Boundary-level DE configuration controlling mutation and crossover.
    sampler : CandidateSampler[StructuredCandidateT] | None, optional
        Optional sampler for initial population generation.
    random_state : RandomSeed, optional
        Seed or random-state object used to initialize optimizer randomness.
    """

    space: SearchSpace[BoundaryT, StructuredCandidateT]
    population_size: int
    profile: DEProfile = field(default_factory=DEProfile, kw_only=True)
    sampler: CandidateSampler[StructuredCandidateT] | None = field(default=None, kw_only=True)
    random_state: RandomSeed = None
    resolved_profile: DEResolvedProfile = field(init=False, repr=False)
    resolved_sampler: CandidateSampler[StructuredCandidateT] = field(init=False, repr=False)
    structured_space: StructuredSearchSpace[BoundaryT, StructuredCandidateT] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Resolve and validate the canonical DE configuration.

        Raises
        ------
        ValueError
            Raised when population size or profile settings are inconsistent.
        TypeError
            Raised when the supplied space is not a static-topology numeric
            structured space.
        """
        if self.population_size < 4:
            msg = "population_size must be at least 4 for differential evolution"
            raise ValueError(msg)

        structured_space = require_numeric_structured_space(self.space)
        if not structured_space.has_static_topology():
            msg = (
                "differential evolution optimizer requires a structured search space "
                "with static topology"
            )
            raise TypeError(msg)
        resolved_profile = self.profile.resolve()
        if resolved_profile.n_cross > len(structured_space.leaf_paths()):
            msg = "n_cross must not exceed the number of editable leaves"
            raise ValueError(msg)

        object.__setattr__(self, "structured_space", structured_space)
        object.__setattr__(self, "resolved_profile", resolved_profile)
        object.__setattr__(
            self,
            "resolved_sampler",
            SearchSpaceSampler(self.space) if self.sampler is None else self.sampler,
        )

    @override
    def create_initial_state(self) -> DEOptimizerState[StructuredCandidateT]:
        """Create the initial immutable DE state.

        Returns
        -------
        DEOptimizerState[StructuredCandidateT]
            State with initialized randomness and no observed population.
        """
        return DEOptimizerState(
            random_state=RandomStateSnapshot.from_seed(self.random_state),
        )

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        """Return supported execution models.

        Returns
        -------
        frozenset[ExecutionModel]
            Execution models accepted by the DE ask/tell contract.
        """
        return frozenset(
            {
                SEQUENTIAL_EXECUTION_MODEL,
                SYNC_BATCH_EXECUTION_MODEL,
                EXACT_ASYNC_EXECUTION_MODEL,
            },
        )

    @override
    def is_exhausted(self, state: DEOptimizerState[StructuredCandidateT]) -> bool:
        """Report whether the optimizer can emit more proposals.

        Parameters
        ----------
        state : DEOptimizerState[StructuredCandidateT]
            Optimizer state to inspect.

        Returns
        -------
        bool
            Always ``False`` for the current unbounded DE implementation.
        """
        _ = state
        return False

    @override
    def ask(
        self,
        state: DEOptimizerState[StructuredCandidateT],
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[StructuredCandidateT], ...], DEOptimizerState[StructuredCandidateT]]:
        """Emit the next proposal batch and advanced DE state.

        Parameters
        ----------
        state : DEOptimizerState[StructuredCandidateT]
            Current immutable optimizer state.
        batch_size : int, default=1
            Maximum number of proposals to emit.

        Returns
        -------
        tuple[tuple[Proposal[StructuredCandidateT], ...], DEOptimizerState[StructuredCandidateT]]
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

        if len(state.pending_evaluations) > 0:
            msg = "cannot ask while proposals are still pending"
            raise RuntimeError(msg)

        if len(state.population) == 0:
            return self._ask_initial_population(state, batch_size=batch_size)

        next_state = state
        if len(next_state.queued_evaluations) == 0:
            next_state = self._materialize_generation(next_state)

        evaluation_count = min(batch_size, len(next_state.queued_evaluations))
        pending_evaluations = next_state.queued_evaluations[:evaluation_count]
        return (
            tuple(evaluation.proposal for evaluation in pending_evaluations),
            replace(
                next_state,
                queued_evaluations=next_state.queued_evaluations[evaluation_count:],
                pending_evaluations=pending_evaluations,
            ),
        )

    def _ask_initial_population(
        self,
        state: DEOptimizerState[StructuredCandidateT],
        *,
        batch_size: int,
    ) -> tuple[tuple[Proposal[StructuredCandidateT], ...], DEOptimizerState[StructuredCandidateT]]:
        remaining_population = self.population_size - len(state.buffered_evaluations)
        evaluation_count = min(batch_size, remaining_population)
        random_state = state.random_state.materialize()
        candidates = tuple(
            self.resolved_sampler.sample(random_state)
            for _ in range(evaluation_count)
        )
        pending_evaluations = tuple(
            DEPendingEvaluation(
                proposal=Proposal(
                    candidate=self._validated_candidate(candidate),
                    proposal_id=f"de-{state.proposal_index + offset}",
                ),
            )
            for offset, candidate in enumerate(candidates)
        )
        return (
            tuple(evaluation.proposal for evaluation in pending_evaluations),
            replace(
                state,
                random_state=RandomStateSnapshot.from_random_state(random_state),
                proposal_index=state.proposal_index + len(pending_evaluations),
                pending_evaluations=pending_evaluations,
            ),
        )

    def _materialize_generation(
        self,
        state: DEOptimizerState[StructuredCandidateT],
    ) -> DEOptimizerState[StructuredCandidateT]:
        random_state = state.random_state.materialize()
        queued_evaluations = tuple(
            DEPendingEvaluation(
                proposal=Proposal(
                    candidate=self._generate_trial(
                        population=state.population,
                        target_index=target_index,
                        random_state=random_state,
                    ),
                    proposal_id=f"de-{state.proposal_index + target_index}",
                ),
                target_index=target_index,
            )
            for target_index in range(self.population_size)
        )
        return replace(
            state,
            random_state=RandomStateSnapshot.from_random_state(random_state),
            proposal_index=state.proposal_index + len(queued_evaluations),
            queued_evaluations=queued_evaluations,
        )

    def _generate_trial(
        self,
        *,
        population: tuple[DEPopulationMember[StructuredCandidateT], ...],
        target_index: int,
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        base_index, differential_index_a, differential_index_b = (
            self._select_differential_parent_indices(
                random_state=random_state,
                population_size=len(population),
                target_index=target_index,
            )
        )
        mutation_factor = self._sample_mutation_factor(random_state)
        trial_candidate = differential_evolution_variation(
            space=self.structured_space,
            target_parent=population[target_index].candidate,
            base_parent=population[base_index].candidate,
            differential_parent_a=population[differential_index_a].candidate,
            differential_parent_b=population[differential_index_b].candidate,
            mutation_factor=mutation_factor,
            recombination_probability=self.resolved_profile.recombination_probability,
            n_cross=self.resolved_profile.n_cross,
            random_state=random_state,
        )
        return self._validated_candidate(trial_candidate)

    def _sample_mutation_factor(self, random_state: np.random.RandomState) -> float:
        mutation_low, mutation_high = self.resolved_profile.mutation_range
        mutation_factor = mutation_low
        if mutation_high > mutation_low:
            mutation_factor += (
                (mutation_high - mutation_low)
                * float(random_state.random_sample())
            )
        return mutation_factor

    @staticmethod
    def _select_differential_parent_indices(
        *,
        random_state: np.random.RandomState,
        population_size: int,
        target_index: int,
    ) -> tuple[int, int, int]:
        candidate_indices = tuple(
            index
            for index in range(population_size)
            if index != target_index
        )
        selected_offsets = random_state_choice_indices_without_replacement(
            random_state,
            population_size=len(candidate_indices),
            count=3,
        )
        return cast(
            tuple[int, int, int],
            tuple(candidate_indices[offset] for offset in selected_offsets),
        )

    def _validated_candidate(self, candidate: StructuredCandidateT) -> StructuredCandidateT:
        self.space.validate(candidate)
        return candidate

    @override
    def tell(
        self,
        state: DEOptimizerState[StructuredCandidateT],
        observations: Sequence[Observation[StructuredCandidateT]],
    ) -> DEOptimizerState[StructuredCandidateT]:
        """Advance the DE state with one observed proposal batch.

        Parameters
        ----------
        state : DEOptimizerState[StructuredCandidateT]
            Current immutable optimizer state.
        observations : Sequence[Observation[StructuredCandidateT]]
            Observations aligned with the currently pending proposals.

        Returns
        -------
        DEOptimizerState[StructuredCandidateT]
            Updated immutable optimizer state after buffering or committing the
            observed evaluations.

        Raises
        ------
        ValueError
            Raised when observation count or ordering does not match the
            pending proposals.
        """
        observation_tuple = tuple(observations)
        if len(observation_tuple) != len(state.pending_evaluations):
            msg = "observation count must match the number of pending proposals"
            raise ValueError(msg)

        for pending_evaluation, observation in zip(
            state.pending_evaluations,
            observation_tuple,
            strict=True,
        ):
            if observation.proposal != pending_evaluation.proposal:
                msg = "observations must align with pending proposal order"
                raise ValueError(msg)

        buffered_evaluations = state.buffered_evaluations + tuple(
            DEObservedEvaluation(
                member=DEPopulationMember(
                    candidate=observation.candidate,
                    value=observation.value,
                    score=observation.score,
                ),
                target_index=pending_evaluation.target_index,
            )
            for pending_evaluation, observation in zip(
                state.pending_evaluations,
                observation_tuple,
                strict=True,
            )
        )
        next_state = replace(
            state,
            pending_evaluations=(),
            buffered_evaluations=buffered_evaluations,
        )

        if len(state.population) == 0:
            return self._tell_initial_population(next_state)

        return self._tell_generation(next_state)

    def _tell_initial_population(
        self,
        state: DEOptimizerState[StructuredCandidateT],
    ) -> DEOptimizerState[StructuredCandidateT]:
        if len(state.buffered_evaluations) < self.population_size:
            return state

        if len(state.buffered_evaluations) != self.population_size:
            msg = "initial population buffer exceeded population_size"
            raise RuntimeError(msg)

        if any(
            evaluation.target_index is not None
            for evaluation in state.buffered_evaluations
        ):
            msg = "initial population evaluations must not carry replacement targets"
            raise RuntimeError(msg)

        return replace(
            state,
            population=tuple(
                evaluation.member
                for evaluation in state.buffered_evaluations
            ),
            buffered_evaluations=(),
        )

    def _tell_generation(
        self,
        state: DEOptimizerState[StructuredCandidateT],
    ) -> DEOptimizerState[StructuredCandidateT]:
        if len(state.buffered_evaluations) < self.population_size:
            return state

        if len(state.buffered_evaluations) != self.population_size:
            msg = "offspring buffer exceeded population_size"
            raise RuntimeError(msg)

        next_population = list(state.population)
        for evaluation in state.buffered_evaluations:
            if evaluation.target_index is None:
                msg = "generation evaluations must carry replacement targets"
                raise RuntimeError(msg)

            target_member = next_population[evaluation.target_index]
            if evaluation.member.score <= target_member.score:
                next_population[evaluation.target_index] = evaluation.member

        return replace(
            state,
            generation_index=state.generation_index + 1,
            population=tuple(next_population),
            buffered_evaluations=(),
        )
