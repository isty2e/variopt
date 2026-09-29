"""Immutable pending/buffer transitions at the CSA tell boundary."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import pytest

from tests.csa_support import (
    AbsoluteDistance,
    CSAAcceptancePolicy,
    RepeatParent,
    make_optimizer,
    perturbation_schedule,
    schedule,
)
from variopt import IntegerSpace, Observation, Proposal
from variopt.algorithms.population.csa import CSAOptimizer
from variopt.algorithms.population.csa.banking.bank import BankEntry
from variopt.algorithms.population.csa.engine import CSAEngineState
from variopt.algorithms.population.csa.generation.proposal import CSAProposalPolicy
from variopt.algorithms.population.csa.trace.events import CSAEventTraceState


def _observations(
    proposals: Sequence[Proposal[int]],
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


def _prepared_generation(
    *,
    adaptation: bool = False,
    trace: bool = False,
    issued_count: int = 8,
    temperature: float = 0.0,
) -> tuple[CSAOptimizer[int, int], CSAEngineState[int], tuple[Proposal[int], ...]]:
    optimizer = make_optimizer(
        space=IntegerSpace(0, 100),
        diversity_metric=AbsoluteDistance(),
        variation_operator=RepeatParent(),
        bank_capacity=4,
        seed_count=1,
        cutoff_schedule=schedule(initial_distance_cutoff=1.0),
        perturbation_schedule=perturbation_schedule(
            regular_children_per_seed=8,
            initial_children_per_seed=0,
            shuffle_children=False,
        ),
        proposal_policy=CSAProposalPolicy(enabled=adaptation),
        acceptance_policy=CSAAcceptancePolicy(initial_temperature=temperature),
        random_state=31,
    ).optimizer
    initial = optimizer.create_state(
        trace_state=CSAEventTraceState() if trace else None
    )
    samples, state = optimizer.ask(initial, batch_size=4)
    state = optimizer.tell(state, _observations(samples))
    proposals, state = optimizer.ask(state, batch_size=issued_count)
    assert len(proposals) == issued_count
    assert state.generation_state.is_active
    return optimizer, state, proposals


@pytest.mark.parametrize("batch_size", [1, 7])
def test_partial_tell_constructs_one_coherent_engine_snapshot(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    optimizer, state, proposals = _prepared_generation(adaptation=True, trace=True)
    observations = _observations(proposals[:batch_size])
    constructed: list[CSAEngineState[int]] = []
    original = CSAEngineState.__post_init__

    def record_construction(self: CSAEngineState[int]) -> None:
        original(self)
        constructed.append(self)

    monkeypatch.setattr(CSAEngineState, "__post_init__", record_construction)
    updated = optimizer.tell(state, observations)

    assert len(constructed) == 1
    assert constructed[0] is updated
    assert updated.pending_proposals.proposals == proposals[batch_size:]
    assert updated.generation_state.pending_proposal_ids == frozenset(
        proposal.proposal_id for proposal in proposals[batch_size:]
    )
    assert (
        tuple(
            evaluation.observation
            for evaluation in updated.generation_state.buffered_evaluations
        )
        == observations
    )
    assert updated.banking_state is state.banking_state
    assert updated.progression_state is state.progression_state
    assert updated.selection_state is state.selection_state
    assert updated.proposal_state is state.proposal_state
    assert updated.scoring_state is state.scoring_state
    assert updated.trace_state is state.trace_state
    assert updated.random_state is state.random_state
    assert updated.proposal_index == state.proposal_index
    assert state.pending_proposals.proposals == proposals
    assert state.generation_state.buffered_evaluations == ()


def test_full_tell_materializes_idle_generation_before_banking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer, state, proposals = _prepared_generation(adaptation=True, trace=True)
    observations = _observations(proposals)
    expected = optimizer.tell(state, observations)
    partial = optimizer.tell(state, observations[:3])
    constructed: list[CSAEngineState[int]] = []
    original = CSAEngineState.__post_init__

    def record_construction(self: CSAEngineState[int]) -> None:
        original(self)
        constructed.append(self)

    monkeypatch.setattr(CSAEngineState, "__post_init__", record_construction)
    completed = optimizer.tell(partial, observations[3:])

    assert constructed
    assert all(snapshot.pending_proposals.is_empty for snapshot in constructed)
    assert all(not snapshot.generation_state.is_active for snapshot in constructed)
    assert constructed[0].banking_state is partial.banking_state
    assert completed == expected
    assert partial.pending_proposals.proposals == proposals[3:]
    assert len(partial.generation_state.buffered_evaluations) == 3


@pytest.mark.parametrize(
    "invalid_kind", ["missing", "unknown", "mismatch", "duplicate", "candidate"]
)
def test_invalid_later_observation_does_not_consume_or_buffer_earlier_feedback(
    invalid_kind: Literal["missing", "unknown", "mismatch", "duplicate", "candidate"],
) -> None:
    optimizer, state, proposals = _prepared_generation(adaptation=True, trace=True)
    observations = _observations(proposals)
    invalid = observations[1]
    if invalid_kind == "missing":
        invalid = replace(
            invalid,
            request=replace(
                invalid.request, proposal=replace(proposals[1], proposal_id=None)
            ),
        )
        message = "must reference proposal ids"
    elif invalid_kind == "unknown":
        invalid = replace(
            invalid,
            request=replace(
                invalid.request, proposal=replace(proposals[1], proposal_id="stale")
            ),
        )
        message = "does not correspond to a pending proposal"
    elif invalid_kind == "mismatch":
        invalid = replace(
            invalid,
            request=replace(
                invalid.request,
                proposal=replace(
                    proposals[1], candidate=(proposals[1].candidate + 1) % 101
                ),
            ),
        )
        message = "does not match the pending proposal"
    elif invalid_kind == "duplicate":
        invalid = observations[0]
        message = "distinct proposal ids"
    else:
        invalid = replace(invalid, candidate=101)
        message = "bounds"

    with pytest.raises(ValueError, match=message):
        optimizer.tell(state, (observations[0], invalid))

    assert state.pending_proposals.proposals == proposals
    assert state.generation_state.buffered_evaluations == ()
    assert len(state.generation_state.pending_proposal_ids) == len(proposals)
    assert len(state.proposal_state.pending_attributions) == len(proposals)
    assert optimizer.tell(state, observations) == optimizer.tell(state, observations)


@pytest.mark.parametrize("adaptation", [False, True])
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_split_feedback_preserves_arrival_order_trace_and_checkpoint_continuation(
    adaptation: bool, trace: bool, reverse: bool
) -> None:
    optimizer, state, proposals = _prepared_generation(
        adaptation=adaptation, trace=trace
    )
    observations = tuple(
        Observation(
            proposal=proposal,
            candidate=20 + index,
            value=-float(index),
            score=-float(index),
        )
        for index, proposal in enumerate(proposals)
    )
    if reverse:
        observations = observations[::-1]
    expected = optimizer.tell(state, observations)
    split = state
    for index, observation in enumerate(observations):
        split = optimizer.tell(split, (observation,))
        if index < len(observations) - 1:
            assert (
                tuple(
                    evaluation.observation
                    for evaluation in split.generation_state.buffered_evaluations
                )
                == observations[: index + 1]
            )
            with pytest.raises(ValueError, match="pending proposal"):
                optimizer.state_to_dict(split)

    assert split == expected
    restored = optimizer.state_from_dict(optimizer.state_to_dict(split))
    for _ in range(3):
        proposals, split = optimizer.ask(split, batch_size=8)
        restored_proposals, restored = optimizer.ask(restored, batch_size=8)
        assert proposals == restored_proposals
        split = optimizer.tell(split, _observations(proposals))
        restored = optimizer.tell(restored, _observations(restored_proposals))
        assert replace(split, trace_state=None) == restored

    assert state.pending_proposals.proposals
    assert state.generation_state.buffered_evaluations == ()


def test_split_tell_keeps_probabilistic_acceptance_rng_and_temperature() -> None:
    optimizer, state, proposals = _prepared_generation(temperature=10_000.0)
    observations = tuple(
        replace(
            observation, value=observation.value + 10.0, score=observation.score + 10.0
        )
        for observation in _observations(proposals)
    )
    whole = optimizer.tell(state, observations)
    split = state
    for observation in observations:
        split = optimizer.tell(split, (observation,))

    assert split == whole
    assert split.random_state != state.random_state
    assert not split.generation_state.is_active


def test_unissued_children_and_empty_tell_do_not_commit_buffered_successes() -> None:
    optimizer, state, proposals = _prepared_generation(issued_count=1)
    partial = optimizer.tell(state, _observations(proposals))

    assert partial.pending_proposals.is_empty
    assert partial.generation_state.pending_proposal_ids == frozenset()
    assert not partial.generation_state.queue.is_empty
    assert partial.banking_state is state.banking_state
    assert optimizer.tell(partial, ()) == partial
    with pytest.raises(ValueError, match="generation runtime"):
        optimizer.state_to_dict(partial)

    remaining, issued = optimizer.ask(partial, batch_size=8)
    assert len(remaining) == 7
    completed = optimizer.tell(issued, _observations(remaining))
    assert not completed.generation_state.is_active
    assert optimizer.is_checkpoint_safe_state(completed)


def test_repeated_feedback_cannot_reconsume_a_buffered_proposal() -> None:
    optimizer, state, proposals = _prepared_generation()
    observations = _observations(proposals)
    partial = optimizer.tell(state, observations[:1])

    with pytest.raises(ValueError, match="does not correspond to a pending proposal"):
        optimizer.tell(partial, (observations[1], observations[0]))

    assert len(partial.generation_state.buffered_evaluations) == 1
    assert partial.pending_proposals.proposals == proposals[1:]
    assert optimizer.tell(partial, observations[1:]) == optimizer.tell(
        state, observations
    )


def test_callback_failure_after_banking_preserves_input_and_retry_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer, state, proposals = _prepared_generation(adaptation=True, trace=True)
    observations = _observations(proposals)
    expected = optimizer.tell(state, observations)
    partial = optimizer.tell(state, observations[:2])
    seen_entries: list[tuple[BankEntry[int], ...]] = []

    def fail_score_gap(
        self: CSAOptimizer[int, int], entries: Sequence[BankEntry[int]]
    ) -> float | None:
        seen_entries.append(tuple(entries))
        raise RuntimeError("injected score-gap failure")

    with monkeypatch.context() as patch:
        patch.setattr(CSAOptimizer, "infer_score_gap_for_entries", fail_score_gap)
        with pytest.raises(RuntimeError, match="injected score-gap failure"):
            optimizer.tell(partial, observations[2:])

    assert seen_entries == [expected.banking_state.bank.entries]
    assert partial.pending_proposals.proposals == proposals[2:]
    assert len(partial.generation_state.buffered_evaluations) == 2
    assert optimizer.tell(partial, observations[2:]) == expected
