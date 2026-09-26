"""Behavioral tests for the issued-proposal hook and immutable ask transitions."""

from dataclasses import dataclass, field, replace

import pytest
from typing_extensions import override

from tests.csa_support import (
    AbsoluteDistance,
    RepeatParent,
    make_optimizer,
    perturbation_schedule,
    schedule,
)
from variopt import IntegerSpace, Observation, Proposal
from variopt.algorithms.population.csa import CSAOptimizer
from variopt.algorithms.population.csa.engine import CSAEngineState
from variopt.algorithms.population.csa.generation.proposal import CSAProposalPolicy
from variopt.algorithms.population.csa.generation.proposal.state.attribution import (
    PlannedProposalProvenance,
)
from variopt.algorithms.population.csa.trace.events import CSAEventTraceState


@dataclass(frozen=True, slots=True)
class RecordingEmitter(CSAOptimizer[int, int]):
    seen_states: list[CSAEngineState[int]] = field(default_factory=list, compare=False)
    emitted: list[Proposal[int]] = field(default_factory=list, compare=False)
    fail_at_index: int | None = None

    @override
    def emit_proposal(
        self, state: CSAEngineState[int]
    ) -> tuple[
        Proposal[int], bool, PlannedProposalProvenance | None, CSAEngineState[int]
    ]:
        self.seen_states.append(state)
        if state.proposal_index == self.fail_at_index:
            raise RuntimeError("injected emission failure")

        result = super(RecordingEmitter, self).emit_proposal(state)
        self.emitted.append(result[0])
        return result


def recording_emitter(*, adaptation: bool) -> RecordingEmitter:
    base = make_optimizer(
        space=IntegerSpace(0, 100),
        diversity_metric=AbsoluteDistance(),
        variation_operator=RepeatParent(),
        mutation_operators=(RepeatParent(),),
        bank_capacity=4,
        seed_count=1,
        cutoff_schedule=schedule(initial_distance_cutoff=1.0),
        perturbation_schedule=perturbation_schedule(
            regular_children_per_seed=2,
            initial_children_per_seed=0,
            shuffle_children=False,
        ),
        proposal_policy=CSAProposalPolicy(enabled=adaptation),
        random_state=31,
    ).optimizer
    return RecordingEmitter(
        space=base.space,
        diversity_metric=base.diversity_metric,
        bank_capacity=base.bank_capacity,
        profile=base.profile,
        random_state=base.random_state,
    )


def square_observations(
    proposals: tuple[Proposal[int], ...],
) -> tuple[Observation[int], ...]:
    return tuple(
        Observation(
            proposal=proposal,
            candidate=proposal.candidate,
            value=float(proposal.candidate**2),
            score=float(proposal.candidate**2),
        )
        for proposal in proposals
    )


@pytest.mark.parametrize("adaptation", [False, True])
def test_ask_uses_issued_proposals_without_reallocation(adaptation: bool) -> None:
    optimizer = recording_emitter(adaptation=adaptation)
    initial = optimizer.create_initial_state()

    proposals, state = optimizer.ask(initial, batch_size=4)

    assert [proposal.proposal_id for proposal in proposals] == [
        "csa-0",
        "csa-1",
        "csa-2",
        "csa-3",
    ]
    assert state.proposal_index == 4
    for index, proposal in enumerate(proposals):
        assert proposal is optimizer.emitted[index]
        assert state.pending_proposals.get(f"csa-{index}") is proposal
        assert optimizer.seen_states[index].proposal_index == index
        assert len(optimizer.seen_states[index].pending_proposals.proposals) == index
        assert optimizer.seen_states[index].proposal_state.pending_attributions == ()

    assert len(state.proposal_state.pending_attributions) == (4 if adaptation else 0)
    assert initial.pending_proposals.is_empty
    assert initial.proposal_index == 0
    assert not state.generation_state.is_active


@pytest.mark.parametrize("adaptation", [False, True])
def test_later_emission_failure_preserves_caller_snapshot(adaptation: bool) -> None:
    optimizer = replace(recording_emitter(adaptation=adaptation), fail_at_index=1)
    state = optimizer.create_initial_state()
    snapshot = optimizer.state_to_dict(state)

    with pytest.raises(RuntimeError, match="injected emission failure"):
        optimizer.ask(state, batch_size=3)

    assert optimizer.state_to_dict(state) == snapshot
    retry, retried_state = optimizer.ask(state, batch_size=1)
    assert retry[0] == optimizer.emitted[0]
    assert retried_state.proposal_index == 1
    assert retried_state.pending_proposals.proposals == retry


@pytest.mark.parametrize("adaptation", [False, True])
def test_partial_initial_feedback_keeps_unobserved_samples_pending(
    adaptation: bool,
) -> None:
    optimizer = recording_emitter(adaptation=adaptation)
    initial = optimizer.create_initial_state()
    samples, sampled_state = optimizer.ask(initial, batch_size=6)
    partial = optimizer.tell(sampled_state, square_observations(samples[:2]))

    extra, expanded = optimizer.ask(partial, batch_size=2)

    assert [proposal.proposal_id for proposal in extra] == ["csa-6", "csa-7"]
    assert expanded.pending_proposals.proposals == samples[2:] + extra
    assert expanded.proposal_index == 8
    assert not expanded.generation_state.is_active
    assert partial.pending_proposals.proposals == samples[2:]
    assert partial.proposal_index == 6
    assert partial.random_state is sampled_state.random_state

    completed = optimizer.tell(expanded, square_observations(samples[2:] + extra))

    assert completed.pending_proposals.is_empty
    assert completed.proposal_state.pending_attributions == ()
    assert completed.banking_state.bank.is_full


@pytest.mark.parametrize("adaptation", [False, True])
def test_later_queued_emission_failure_preserves_unissued_children(
    adaptation: bool,
) -> None:
    optimizer = recording_emitter(adaptation=adaptation)
    samples, state = optimizer.ask(optimizer.create_initial_state(), batch_size=4)
    state = optimizer.tell(state, square_observations(samples))
    first, partial = optimizer.ask(state, batch_size=1)
    failing = replace(optimizer, fail_at_index=6)
    queue = partial.generation_state.queue

    with pytest.raises(RuntimeError, match="injected emission failure"):
        failing.ask(partial, batch_size=2)

    assert partial.proposal_index == 5
    assert partial.pending_proposals.proposals == first
    assert queue.head_index == 1
    assert partial.generation_state.queue is queue
    assert partial.generation_state.pending_proposal_ids == frozenset({"csa-4"})
    retry, retried = optimizer.ask(partial, batch_size=2)
    assert [proposal.proposal_id for proposal in retry] == ["csa-5", "csa-6"]
    assert retry[0].candidate is queue.candidates[1].candidate
    assert retried.pending_proposals.proposals == first + retry
    assert retried.random_state is partial.random_state
    assert retried.generation_state.queue.is_empty


@pytest.mark.parametrize("adaptation", [False, True])
@pytest.mark.parametrize("trace", [False, True])
def test_generation_forks_match_split_emission_and_keep_batch_end_provenance(
    adaptation: bool, trace: bool
) -> None:
    optimizer = recording_emitter(adaptation=adaptation)
    initial = optimizer.create_state(
        trace_state=CSAEventTraceState() if trace else None
    )
    samples, sampled_state = optimizer.ask(initial, batch_size=4)
    full_bank_state = optimizer.tell(sampled_state, square_observations(samples))
    full_bank_snapshot = optimizer.state_to_dict(full_bank_state)
    optimizer.seen_states.clear()
    optimizer.emitted.clear()

    whole_pool, whole_state = optimizer.ask(full_bank_state, batch_size=10)
    assert len(whole_pool) == 3
    assert len(optimizer.seen_states) == 3
    assert all(
        state.proposal_state.pending_attributions == ()
        for state in optimizer.seen_states
    )
    assert all(
        proposal is emitted
        for proposal, emitted in zip(whole_pool, optimizer.emitted, strict=True)
    )

    first, partial_state = optimizer.ask(full_bank_state, batch_size=1)
    assert len(partial_state.proposal_state.pending_attributions) == (
        1 if adaptation else 0
    )
    queue = partial_state.generation_state.queue
    remaining, split_state = optimizer.ask(partial_state, batch_size=10)
    retry_remaining, retry_state = optimizer.ask(partial_state, batch_size=10)

    assert first + remaining == whole_pool
    assert retry_remaining == remaining
    assert split_state == whole_state == retry_state
    assert split_state.random_state is partial_state.random_state
    assert queue.head_index == 1
    assert partial_state.generation_state.queue is queue
    assert partial_state.proposal_index == 5
    assert optimizer.state_to_dict(full_bank_state) == full_bank_snapshot

    completed = optimizer.tell(whole_state, square_observations(whole_pool))
    split_completed = optimizer.tell(
        split_state, square_observations(first + remaining)
    )
    assert completed == split_completed
    assert not completed.generation_state.is_active
    assert completed.pending_proposals.is_empty
    restored = optimizer.state_from_dict(optimizer.state_to_dict(completed))
    next_pool, continued = optimizer.ask(completed, batch_size=10)
    resumed_pool, resumed = optimizer.ask(restored, batch_size=10)
    assert next_pool == resumed_pool
    assert replace(continued, trace_state=None) == resumed


@pytest.mark.parametrize("adaptation", [False, True])
def test_emit_proposal_returns_registered_state_but_unbound_provenance(
    adaptation: bool,
) -> None:
    optimizer = recording_emitter(adaptation=adaptation)
    state = optimizer.create_initial_state()

    proposal, tracks_generation, planned, issued_state = optimizer.emit_proposal(state)

    assert not tracks_generation
    assert (planned is not None) == adaptation
    assert issued_state.proposal_state.pending_attributions == ()
    assert issued_state.pending_proposals.get("csa-0") is proposal
    assert issued_state.proposal_index == 1
    assert issued_state.random_state != state.random_state


def test_exhausted_engine_rejects_emission_before_sampling() -> None:
    optimizer = recording_emitter(adaptation=True)
    initial = optimizer.create_initial_state()
    state = replace(
        initial,
        progression_state=replace(initial.progression_state, is_exhausted=True),
    )

    with pytest.raises(RuntimeError, match="exhausted CSAOptimizer"):
        optimizer.emit_proposal(state)

    assert state.random_state is initial.random_state
    assert state.proposal_index == 0
    assert state.pending_proposals.is_empty
