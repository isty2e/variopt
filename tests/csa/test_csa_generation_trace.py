"""Generation tracing must observe CSA without changing its execution."""

from dataclasses import replace

import pytest

from variopt import IntegerSpace, Observation, Proposal
from variopt.algorithms.population.csa import (
    CSACutoffSchedule,
    CSAOptimizer,
    CSAPerturbationSchedule,
    CSAPerturbationSpec,
    CSAProfile,
    CSAProposalPolicy,
    RandomResetMutation,
    UniformCrossover,
)
from variopt.algorithms.population.csa.engine.state import CSAEngineState
from variopt.algorithms.population.csa.generation.proposal.logic import (
    mutation_family_weights,
)
from variopt.algorithms.population.csa.trace.events import CSAEventTraceState

_SPACE = IntegerSpace(-100, 100)
_REGULAR = CSAPerturbationSpec(RandomResetMutation(_SPACE), count=3)
_INITIAL = CSAPerturbationSpec(UniformCrossover(_SPACE), count=3)
_SCHEDULES = (
    CSAPerturbationSchedule(regular_family=(_REGULAR,)),
    CSAPerturbationSchedule(initial_family=(_INITIAL,)),
    CSAPerturbationSchedule(regular_family=(_REGULAR,), initial_family=(_INITIAL,)),
    CSAPerturbationSchedule(
        mutation_family=(
            CSAPerturbationSpec(RandomResetMutation(_SPACE), count=1),
            CSAPerturbationSpec(RandomResetMutation(_SPACE), count=2),
        ),
    ),
)
_SCHEDULE_IDS = ("regular", "initial", "regular-initial", "mutation")


def _optimizer(
    schedule: CSAPerturbationSchedule[int],
    *,
    adaptation_enabled: bool,
) -> CSAOptimizer[int, int]:
    return CSAOptimizer.from_space_defaults(
        space=_SPACE,
        bank_capacity=4,
        profile=CSAProfile(
            perturbation_schedule=schedule,
            seed_count=2,
            cutoff_schedule=CSACutoffSchedule(initial_distance_cutoff=1.0),
            proposal_policy=CSAProposalPolicy(enabled=adaptation_enabled),
        ),
        random_state=31,
    )


def _evaluate(proposals: tuple[Proposal[int], ...]) -> tuple[Observation[int], ...]:
    return tuple(
        Observation(
            proposal=proposal,
            candidate=proposal.candidate,
            value=float(proposal.candidate**2),
            score=float(proposal.candidate**2),
        )
        for proposal in proposals
    )


def _step_pair(
    optimizer: CSAOptimizer[int, int],
    traced_state: CSAEngineState[int],
    plain_state: CSAEngineState[int],
    *,
    batch_size: int,
) -> tuple[CSAEngineState[int], CSAEngineState[int]]:
    traced_proposals, traced_state = optimizer.ask(traced_state, batch_size=batch_size)
    plain_proposals, plain_state = optimizer.ask(plain_state, batch_size=batch_size)
    assert traced_proposals
    assert traced_proposals == plain_proposals
    assert replace(traced_state, trace_state=None) == plain_state

    traced_state = optimizer.tell(traced_state, _evaluate(traced_proposals))
    plain_state = optimizer.tell(plain_state, _evaluate(plain_proposals))
    assert replace(traced_state, trace_state=None) == plain_state
    return traced_state, plain_state


@pytest.mark.parametrize("schedule", _SCHEDULES, ids=_SCHEDULE_IDS)
@pytest.mark.parametrize("adaptation_enabled", [False, True])
@pytest.mark.parametrize("shuffle_children", [False, True])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_generation_trace_preserves_proposals_and_state(
    schedule: CSAPerturbationSchedule[int],
    adaptation_enabled: bool,
    shuffle_children: bool,
    batch_size: int,
) -> None:
    optimizer = _optimizer(
        replace(schedule, shuffle_children=shuffle_children),
        adaptation_enabled=adaptation_enabled,
    )
    traced_state = optimizer.create_state(trace_state=CSAEventTraceState[int]())
    plain_state = optimizer.create_initial_state()

    for _ in range(64):
        previous_proposal_state = traced_state.proposal_state
        was_active = traced_state.generation_state.is_active
        traced_state, plain_state = _step_pair(
            optimizer, traced_state, plain_state, batch_size=batch_size
        )
        trace = traced_state.trace_state
        assert trace is not None

        if not was_active and (
            trace.active_generation is not None or trace.completed_generations
        ):
            generation = (
                trace.completed_generations[-1]
                if trace.active_generation is None
                else trace.active_generation
            )
            family_traces = generation.proposal_families_before
            if schedule.mutation_family:
                weights = mutation_family_weights(
                    state=previous_proposal_state, family=schedule.mutation_family
                )
                assert tuple(item.mutation_weight for item in family_traces) == weights
                assert tuple(item.family_key for item in family_traces) == (
                    "mutation:0",
                    "mutation:1",
                )
                counts_by_key = {
                    stat.family_key: stat.observation_count
                    for stat in previous_proposal_state.family_stats
                }
                assert tuple(item.observation_count for item in family_traces) == tuple(
                    counts_by_key.get(item.family_key, 0) for item in family_traces
                )
            else:
                assert family_traces == ()

        if len(trace.completed_generations) == 4:
            break
    else:
        pytest.fail("four generations did not finish")

    expected_families = {
        name
        for name, family in (
            ("regular", schedule.regular_family),
            ("initial", schedule.initial_family),
            ("mutation", schedule.mutation_family),
        )
        if family
    }
    for generation in trace.completed_generations:
        assert {child.family for child in generation.child_pool} == expected_families
        assert len(generation.child_pool) == len(generation.seed_batch) * sum(
            spec.count
            for family in (
                schedule.regular_family,
                schedule.initial_family,
                schedule.mutation_family,
            )
            for spec in family
        )
        assert sorted(generation.shuffled_pool) == sorted(
            child.candidate for child in generation.child_pool
        )
        if not schedule.mutation_family:
            assert generation.proposal_families_after == ()

    if schedule.mutation_family and adaptation_enabled:
        assert traced_state.proposal_state.family_stats
        assert all(
            item.observation_count > 0
            for item in trace.completed_generations[-1].proposal_families_before
        )

    assert optimizer.state_to_dict(traced_state) == optimizer.state_to_dict(plain_state)


@pytest.mark.parametrize("schedule", _SCHEDULES[:3], ids=_SCHEDULE_IDS[:3])
@pytest.mark.parametrize("adaptation_enabled", [False, True])
def test_empty_mutation_trace_survives_checkpoint_continuation(
    schedule: CSAPerturbationSchedule[int],
    adaptation_enabled: bool,
) -> None:
    optimizer = _optimizer(schedule, adaptation_enabled=adaptation_enabled)
    traced_state = optimizer.create_state(trace_state=CSAEventTraceState[int]())
    plain_state = optimizer.create_initial_state()

    for _ in range(32):
        traced_state, plain_state = _step_pair(
            optimizer, traced_state, plain_state, batch_size=1
        )
        trace = traced_state.trace_state
        assert trace is not None
        if trace.completed_generations:
            break
    else:
        pytest.fail("first generation did not finish")

    resumed_state = optimizer.state_from_dict(optimizer.state_to_dict(traced_state))
    assert resumed_state == plain_state
    for _ in range(12):
        traced_state, resumed_state = _step_pair(
            optimizer, traced_state, resumed_state, batch_size=1
        )
