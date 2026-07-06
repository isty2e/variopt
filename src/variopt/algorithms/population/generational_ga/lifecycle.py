"""Shared lifecycle algebra for internal generational GA variants."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Generic, Protocol, TypeVar

import numpy as np

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Observation, Proposal
from ....execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
)
from ....operators import VariationOperator
from ....randomness import (
    RandomSeed,
    RandomStateSnapshot,
    random_state_choice_indices_without_replacement,
)
from ....sampling import CandidateSampler
from ....typevars import CandidateT
from .state import (
    GenerationalGAMemberBuffer,
    GenerationalGAOptimizerState,
    GenerationalGAPopulationMember,
    GenerationalGAVariant,
)

ValidatedCandidateT = TypeVar("ValidatedCandidateT", contravariant=True)

GENERATIONAL_GA_EXECUTION_MODELS = frozenset(
    {
        SEQUENTIAL_EXECUTION_MODEL,
        SYNC_BATCH_EXECUTION_MODEL,
        EXACT_ASYNC_EXECUTION_MODEL,
    },
)


class GenerationalGACandidateValidator(Protocol[ValidatedCandidateT]):
    """Minimal candidate-validation contract required by GA lifecycle code."""

    def validate(self, candidate: ValidatedCandidateT) -> None:
        """Validate one canonical candidate."""
        ...


class GenerationalGAResolvedProfile(Protocol):
    """Common resolved-profile fields consumed by generational GA lifecycle."""

    @property
    def tournament_size(self) -> int:
        """Number of population members sampled for parent selection."""
        ...

    @property
    def crossover_probability(self) -> float:
        """Probability of applying crossover during child generation."""
        ...

    @property
    def mutation_probability(self) -> float:
        """Probability of mutating a generated child."""
        ...


class GenerationalGARuntime(Protocol[CandidateT]):
    """Runtime contract shared by generational GA optimizer variants."""

    @property
    def space(self) -> GenerationalGACandidateValidator[CandidateT]:
        """Search-space validation surface for generated candidates."""
        ...

    @property
    def population_size(self) -> int:
        """Number of members maintained in each generation."""
        ...

    @property
    def crossover_operator(self) -> VariationOperator[CandidateT] | None:
        """Optional crossover operator used before mutation."""
        ...

    @property
    def mutation_operator(self) -> VariationOperator[CandidateT] | None:
        """Optional mutation operator used after crossover or cloning."""
        ...

    @property
    def resolved_profile(self) -> GenerationalGAResolvedProfile:
        """Resolved lifecycle profile shared by GA variants."""
        ...

    @property
    def resolved_sampler(self) -> CandidateSampler[CandidateT]:
        """Sampler used to initialize the first population."""
        ...


@dataclass(frozen=True, slots=True)
class GenerationalGAGenerationCommit(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Result of committing one full generational GA offspring buffer.

    Parameters
    ----------
    population : tuple[GenerationalGAPopulationMember[CandidateT], ...]
        Next population selected by the variant-local survival policy.
    random_state : RandomStateSnapshot | None, optional
        Advanced random-state snapshot when survival itself consumes
        randomness, as in restricted-tournament replacement.
    """

    population: tuple[GenerationalGAPopulationMember[CandidateT], ...]
    random_state: RandomStateSnapshot | None = None

    def __post_init__(self) -> None:
        """Reject empty generation commits.

        Raises
        ------
        ValueError
            If the selected population is empty.
        """
        if len(self.population) == 0:
            msg = "population must not be empty"
            raise ValueError(msg)


class GenerationalGAGenerationCommitter(Protocol[CandidateT]):
    """Variant-local survival policy hook for a completed generation."""

    def __call__(
        self,
        *,
        parents: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        offspring: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        random_state: RandomStateSnapshot,
    ) -> GenerationalGAGenerationCommit[CandidateT]:
        """Select the next population from one parent/offspring pool."""
        ...


def create_initial_generational_ga_state(
    random_state: RandomSeed,
    *,
    variant: GenerationalGAVariant,
) -> GenerationalGAOptimizerState[CandidateT]:
    """Create an initial generational GA state.

    Parameters
    ----------
    random_state : RandomSeed
        Seed or random-state object used to initialize optimizer randomness.
    variant : GenerationalGAVariant
        Optimizer variant that owns the resulting state.

    Returns
    -------
    GenerationalGAOptimizerState[CandidateT]
        State with initialized randomness and no observed population.
    """
    return GenerationalGAOptimizerState(
        variant=variant,
        random_state=RandomStateSnapshot.from_seed(random_state),
    )


def ask_generational_ga(
    runtime: GenerationalGARuntime[CandidateT],
    state: GenerationalGAOptimizerState[CandidateT],
    *,
    batch_size: int,
    proposal_id_prefix: str,
    variant: GenerationalGAVariant,
) -> tuple[
    tuple[Proposal[CandidateT], ...],
    GenerationalGAOptimizerState[CandidateT],
]:
    """Emit the next proposal batch for a generational GA variant.

    Parameters
    ----------
    runtime : GenerationalGARuntime[CandidateT]
        Optimizer-owned runtime contract used by the lifecycle.
    state : GenerationalGAOptimizerState[CandidateT]
        Current immutable optimizer state.
    batch_size : int
        Maximum number of proposals to emit.
    proposal_id_prefix : str
        Prefix used to form monotone proposal identifiers.
    variant : GenerationalGAVariant
        Optimizer variant expected to own ``state``.

    Returns
    -------
    tuple[tuple[Proposal[CandidateT], ...], GenerationalGAOptimizerState[CandidateT]]
        Proposal batch together with the advanced immutable state.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive or ``proposal_id_prefix`` is empty.
    RuntimeError
        If outstanding proposals have not yet been observed.
    """
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    if proposal_id_prefix == "":
        msg = "proposal_id_prefix must not be empty"
        raise ValueError(msg)

    require_generational_ga_variant_state(state, variant=variant)

    if len(state.pending_proposals) > 0:
        msg = "cannot ask while proposals are still pending"
        raise RuntimeError(msg)

    if len(state.population) == 0:
        return ask_initial_generational_ga_population(
            runtime,
            state,
            batch_size=batch_size,
            proposal_id_prefix=proposal_id_prefix,
        )

    next_state = state
    if next_state.queued_proposal_index == len(next_state.queued_proposals):
        next_state = materialize_generational_ga_generation(
            runtime,
            next_state,
            proposal_id_prefix=proposal_id_prefix,
        )

    queued_start = next_state.queued_proposal_index
    queued_stop = min(batch_size + queued_start, len(next_state.queued_proposals))
    proposals = next_state.queued_proposals[queued_start:queued_stop]
    remaining_queue_is_empty = queued_stop == len(next_state.queued_proposals)
    return proposals, replace(
        next_state,
        queued_proposals=() if remaining_queue_is_empty else next_state.queued_proposals,
        queued_proposal_index=0 if remaining_queue_is_empty else queued_stop,
        pending_proposals=proposals,
    )


def tell_generational_ga(
    state: GenerationalGAOptimizerState[CandidateT],
    observations: Sequence[Observation[CandidateT]],
    *,
    population_size: int,
    build_next_population: GenerationalGAGenerationCommitter[CandidateT],
    variant: GenerationalGAVariant,
) -> GenerationalGAOptimizerState[CandidateT]:
    """Advance generational GA state from one aligned observation batch.

    Parameters
    ----------
    state : GenerationalGAOptimizerState[CandidateT]
        Current immutable optimizer state.
    observations : Sequence[Observation[CandidateT]]
        Observations aligned with the currently pending proposals.
    population_size : int
        Number of members required for a complete population/generation.
    build_next_population : GenerationalGAGenerationCommitter[CandidateT]
        Variant-local survival policy hook.
    variant : GenerationalGAVariant
        Optimizer variant expected to own ``state``.

    Returns
    -------
    GenerationalGAOptimizerState[CandidateT]
        Updated immutable optimizer state after buffering or committing the
        observed members.

    Raises
    ------
    ValueError
        If observation count or ordering does not match pending proposals.
    RuntimeError
        If a buffer exceeds ``population_size`` or the survival hook returns
        the wrong number of members.
    """
    require_generational_ga_variant_state(state, variant=variant)

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
        GenerationalGAPopulationMember(
            candidate=observation.candidate,
            value=observation.value,
            score=observation.score,
        )
        for observation in observation_tuple
    )
    next_state = replace(
        state,
        pending_proposals=(),
        buffered_member_buffer=state.buffered_member_buffer.append(new_members),
    )

    if len(state.population) == 0:
        return tell_initial_generational_ga_population(
            next_state,
            population_size=population_size,
        )

    return tell_generational_ga_generation(
        next_state,
        population_size=population_size,
        build_next_population=build_next_population,
    )


def ask_initial_generational_ga_population(
    runtime: GenerationalGARuntime[CandidateT],
    state: GenerationalGAOptimizerState[CandidateT],
    *,
    batch_size: int,
    proposal_id_prefix: str,
) -> tuple[
    tuple[Proposal[CandidateT], ...],
    GenerationalGAOptimizerState[CandidateT],
]:
    """Emit proposals for an incomplete initial population."""
    remaining_population = (
        runtime.population_size - state.buffered_member_buffer.member_count
    )
    proposal_count = min(batch_size, remaining_population)
    random_state = state.random_state.materialize()
    candidates = tuple(
        runtime.resolved_sampler.sample(random_state)
        for _ in range(proposal_count)
    )
    next_random_state = RandomStateSnapshot.from_random_state(random_state)
    proposals = tuple(
        Proposal(
            candidate=validate_generational_ga_candidate(runtime, candidate),
            proposal_id=f"{proposal_id_prefix}{state.proposal_index + offset}",
        )
        for offset, candidate in enumerate(candidates)
    )
    return proposals, replace(
        state,
        random_state=next_random_state,
        proposal_index=state.proposal_index + len(proposals),
        pending_proposals=proposals,
    )


def materialize_generational_ga_generation(
    runtime: GenerationalGARuntime[CandidateT],
    state: GenerationalGAOptimizerState[CandidateT],
    *,
    proposal_id_prefix: str,
) -> GenerationalGAOptimizerState[CandidateT]:
    """Materialize one full generation of queued offspring proposals."""
    random_state = state.random_state.materialize()
    proposals = tuple(
        Proposal(
            candidate=generate_generational_ga_child(
                runtime,
                population=state.population,
                random_state=random_state,
            ),
            proposal_id=f"{proposal_id_prefix}{state.proposal_index + offset}",
        )
        for offset in range(runtime.population_size)
    )
    next_random_state = RandomStateSnapshot.from_random_state(random_state)
    return replace(
        state,
        random_state=next_random_state,
        proposal_index=state.proposal_index + len(proposals),
        queued_proposals=proposals,
        queued_proposal_index=0,
    )


def generate_generational_ga_child(
    runtime: GenerationalGARuntime[CandidateT],
    *,
    population: tuple[GenerationalGAPopulationMember[CandidateT], ...],
    random_state: np.random.RandomState,
) -> CandidateT:
    """Generate and validate one child candidate from a population."""
    child_candidate: CandidateT
    crossover_operator = runtime.crossover_operator
    if (
        crossover_operator is not None
        and float(random_state.random_sample())
        < runtime.resolved_profile.crossover_probability
    ):
        parent_candidates = tuple(
            population[index].candidate
            for index in select_generational_ga_parent_indices(
                runtime,
                random_state=random_state,
                population=population,
                count=crossover_operator.arity,
            )
        )
        child_candidate = crossover_operator.apply(parent_candidates, random_state)
    else:
        parent_index = select_generational_ga_parent_indices(
            runtime,
            random_state=random_state,
            population=population,
            count=1,
        )[0]
        child_candidate = population[parent_index].candidate

    mutation_operator = runtime.mutation_operator
    if (
        mutation_operator is not None
        and float(random_state.random_sample())
        < runtime.resolved_profile.mutation_probability
    ):
        child_candidate = mutation_operator.apply((child_candidate,), random_state)

    return validate_generational_ga_candidate(runtime, child_candidate)


def select_generational_ga_parent_indices(
    runtime: GenerationalGARuntime[CandidateT],
    *,
    random_state: np.random.RandomState,
    population: tuple[GenerationalGAPopulationMember[CandidateT], ...],
    count: int,
) -> tuple[int, ...]:
    """Select parent indices through repeated tournament selection."""
    return tuple(
        select_generational_ga_tournament_parent_index(
            runtime,
            random_state=random_state,
            population=population,
        )
        for _ in range(count)
    )


def select_generational_ga_tournament_parent_index(
    runtime: GenerationalGARuntime[CandidateT],
    *,
    random_state: np.random.RandomState,
    population: tuple[GenerationalGAPopulationMember[CandidateT], ...],
) -> int:
    """Select one parent index by tournament over minimization scores."""
    tournament_indices = random_state_choice_indices_without_replacement(
        random_state,
        population_size=len(population),
        count=runtime.resolved_profile.tournament_size,
    )
    return min(
        tournament_indices,
        key=lambda index: population[index].score,
    )


def validate_generational_ga_candidate(
    runtime: GenerationalGARuntime[CandidateT],
    candidate: CandidateT,
) -> CandidateT:
    """Validate and return one generated candidate."""
    runtime.space.validate(candidate)
    return candidate


def tell_initial_generational_ga_population(
    state: GenerationalGAOptimizerState[CandidateT],
    *,
    population_size: int,
) -> GenerationalGAOptimizerState[CandidateT]:
    """Commit buffered observations as the initial sorted population."""
    if state.buffered_member_buffer.member_count < population_size:
        return state

    if state.buffered_member_buffer.member_count != population_size:
        msg = "initial population buffer exceeded population_size"
        raise RuntimeError(msg)

    empty_buffer: GenerationalGAMemberBuffer[CandidateT] = GenerationalGAMemberBuffer()
    return replace(
        state,
        population=sort_generational_ga_population(
            state.buffered_member_buffer.materialize(),
        ),
        buffered_member_buffer=empty_buffer,
    )


def tell_generational_ga_generation(
    state: GenerationalGAOptimizerState[CandidateT],
    *,
    population_size: int,
    build_next_population: GenerationalGAGenerationCommitter[CandidateT],
) -> GenerationalGAOptimizerState[CandidateT]:
    """Commit a full offspring buffer through variant-local survival."""
    if state.buffered_member_buffer.member_count < population_size:
        return state

    if state.buffered_member_buffer.member_count != population_size:
        msg = "offspring buffer exceeded population_size"
        raise RuntimeError(msg)

    offspring = state.buffered_member_buffer.materialize()
    commit = build_next_population(
        parents=state.population,
        offspring=offspring,
        random_state=state.random_state,
    )
    if len(commit.population) != population_size:
        msg = "next population size must match population_size"
        raise RuntimeError(msg)

    empty_buffer: GenerationalGAMemberBuffer[CandidateT] = GenerationalGAMemberBuffer()
    if commit.random_state is None:
        return replace(
            state,
            generation_index=state.generation_index + 1,
            population=commit.population,
            buffered_member_buffer=empty_buffer,
        )

    return replace(
        state,
        random_state=commit.random_state,
        generation_index=state.generation_index + 1,
        population=commit.population,
        buffered_member_buffer=empty_buffer,
    )


def sort_generational_ga_population(
    members: tuple[GenerationalGAPopulationMember[CandidateT], ...],
) -> tuple[GenerationalGAPopulationMember[CandidateT], ...]:
    """Return members sorted by ascending minimization score."""
    return tuple(
        sorted(
            members,
            key=lambda member: member.score,
        )
    )


def require_generational_ga_variant_state(
    state: GenerationalGAOptimizerState[CandidateT],
    *,
    variant: GenerationalGAVariant,
) -> None:
    """Reject a state owned by a different generational GA variant.

    Parameters
    ----------
    state : GenerationalGAOptimizerState[CandidateT]
        State to validate before an optimizer transition.
    variant : GenerationalGAVariant
        Expected owning optimizer variant.

    Raises
    ------
    ValueError
        If ``state`` belongs to another variant.
    """
    if state.variant is not variant:
        msg = "state variant does not match optimizer variant"
        raise ValueError(msg)
