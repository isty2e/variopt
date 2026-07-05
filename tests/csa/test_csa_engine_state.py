"""Tests for canonical CSA engine-state aggregates."""

from dataclasses import replace

import numpy as np
import pytest

from variopt import Observation, Proposal
from variopt.algorithms.population.csa.banking.bank import Bank, BankEntry
from variopt.algorithms.population.csa.banking.clustering import (
    CSAClusteringPolicy,
    CSAClusteringState,
)
from variopt.algorithms.population.csa.banking.growth import (
    CSABankGrowthPolicy,
    CSABankGrowthState,
)
from variopt.algorithms.population.csa.banking.reference import (
    ReferenceBank,
)
from variopt.algorithms.population.csa.engine import (
    CSAAskPlan,
    CSABankingState,
    CSAEngineState,
    CSAMaterializedGeneration,
    CSAPendingProposals,
    CSAScoringState,
    commit_materialized_generation,
    plan_next_ask,
)
from variopt.algorithms.population.csa.generation.proposal import (
    CSAProposalPolicy,
    CSAProposalState,
)
from variopt.algorithms.population.csa.generation.proposal.state import (
    ProposalAttribution,
)
from variopt.algorithms.population.csa.generation.state import (
    GeneratedCandidate,
    GenerationQueue,
    GenerationRuntimeState,
)
from variopt.algorithms.population.csa.progression.cutoff.state import (
    CSACutoffState,
)
from variopt.algorithms.population.csa.progression.stage import (
    CSAStageState,
)
from variopt.algorithms.population.csa.progression.state import (
    CSAProgressionState,
)
from variopt.algorithms.population.csa.scoring.acceptance import (
    CSAAcceptancePolicy,
)
from variopt.algorithms.population.csa.scoring.acceptance_state import (
    CSAAcceptanceState,
)
from variopt.algorithms.population.csa.scoring.model import CSAScoreModel
from variopt.algorithms.population.csa.scoring.model_state import (
    CSAScoreModelState,
)
from variopt.algorithms.population.csa.selection.state import (
    SeedSelectionState,
)
from variopt.randomness import RandomStateSnapshot


class CSAPendingProposalsTests:
    """Regression tests for canonical pending-proposal registry behavior."""

    def test_rejects_proposals_without_ids(self) -> None:
        with pytest.raises(ValueError, match="proposal ids"):
            _ = CSAPendingProposals[int](proposals=(Proposal(candidate=1),))

    def test_rejects_duplicate_proposal_ids(self) -> None:
        proposal = Proposal(candidate=1, proposal_id="csa-0")

        with pytest.raises(ValueError, match="distinct proposal ids"):
            _ = CSAPendingProposals[int](proposals=(proposal, proposal))


class GenerationQueueTests:
    """Regression tests for CSA generated-candidate queue behavior."""

    def test_dequeue_advances_head_without_copying_candidates(self) -> None:
        queue = GenerationQueue(
            candidates=(
                GeneratedCandidate(candidate=11),
                GeneratedCandidate(candidate=12),
            ),
        )

        first_candidate, next_queue = queue.dequeue()
        second_candidate, empty_queue = next_queue.dequeue()

        assert first_candidate.candidate == 11
        assert second_candidate.candidate == 12
        assert next_queue.candidates is queue.candidates
        assert empty_queue.candidates is queue.candidates
        assert next_queue.head_index == 1
        assert empty_queue.head_index == 2
        assert empty_queue.is_empty

    def test_rejects_invalid_head_index(self) -> None:
        with pytest.raises(ValueError, match="head_index"):
            _ = GenerationQueue[int](
                candidates=(GeneratedCandidate(candidate=11),),
                head_index=2,
            )

    def test_rejects_negative_head_index(self) -> None:
        with pytest.raises(ValueError, match="head_index"):
            _ = GenerationQueue[int](
                candidates=(GeneratedCandidate(candidate=11),),
                head_index=-1,
            )

    def test_exhausted_nonempty_queue_behaves_as_empty(self) -> None:
        queue = GenerationQueue(
            candidates=(GeneratedCandidate(candidate=11),),
            head_index=1,
        )

        assert queue.is_empty
        with pytest.raises(RuntimeError, match="empty generation queue"):
            _ = queue.dequeue()
        with pytest.raises(ValueError, match="empty queue"):
            _ = GenerationRuntimeState[int]().begin(queue)

    def test_runtime_with_exhausted_queue_and_no_buffers_is_inactive(self) -> None:
        runtime: GenerationRuntimeState[int] = GenerationRuntimeState(
            queue=GenerationQueue(
                candidates=(GeneratedCandidate(candidate=11),),
                head_index=1,
            ),
        )

        assert not runtime.is_active
        assert not runtime.ready_to_commit

    def test_runtime_with_exhausted_queue_and_buffer_is_ready_to_commit(self) -> None:
        observation: Observation[int] = Observation(
            proposal=Proposal(candidate=11, proposal_id="csa-0"),
            candidate=11,
            value=121.0,
            score=121.0,
        )
        runtime: GenerationRuntimeState[int] = GenerationRuntimeState(
            queue=GenerationQueue(
                candidates=(GeneratedCandidate(candidate=11),),
                head_index=1,
            ),
            buffered_observations=(observation,),
        )

        buffered_observations, idle_runtime = runtime.release_buffer()

        assert runtime.ready_to_commit
        assert buffered_observations == (observation,)
        assert not idle_runtime.is_active

    def test_shuffled_queue_preserves_all_candidates_with_zero_head_index(self) -> None:
        candidates: tuple[GeneratedCandidate[int], ...] = (
            GeneratedCandidate(candidate=1),
            GeneratedCandidate(candidate=2),
            GeneratedCandidate(candidate=3),
        )

        queue: GenerationQueue[int] = GenerationQueue[int].from_candidates(
            candidates,
            shuffle=True,
            random_state=np.random.RandomState(0),
        )
        first_candidate, queue = queue.dequeue()
        second_candidate, queue = queue.dequeue()
        third_candidate, queue = queue.dequeue()

        assert queue.head_index == 3
        assert queue.is_empty
        assert queue.candidates is not candidates
        assert sorted(entry.candidate for entry in queue.candidates) == [1, 2, 3]
        assert sorted(
            (
                first_candidate.candidate,
                second_candidate.candidate,
                third_candidate.candidate,
            )
        ) == [1, 2, 3]


class CSAEngineStateTests:
    """Regression tests for CSAEngineState invariants and helpers."""

    def test_allocate_proposal_id_advances_index(self) -> None:
        state = build_engine_state()

        proposal_id, next_state = state.allocate_proposal_id()

        assert proposal_id == "csa-0"
        assert next_state.proposal_index == 1
        assert state.proposal_index == 0

    def test_issue_proposal_registers_pending_and_generation(self) -> None:
        state = build_engine_state()
        proposal = Proposal(candidate=7, proposal_id="csa-0")

        next_state = state.issue_proposal(proposal, tracks_generation=True)

        assert next_state.pending_proposals.get("csa-0") == proposal
        assert next_state.generation_state.pending_proposal_ids == frozenset({"csa-0"})

    def test_consume_pending_proposals_removes_registered_ids(self) -> None:
        proposal = Proposal(candidate=7, proposal_id="csa-0")
        state = build_engine_state().issue_proposal(proposal, tracks_generation=False)

        next_state = state.consume_pending_proposals({"csa-0"})

        assert next_state.pending_proposals.is_empty
        assert state.pending_proposals.get("csa-0") == proposal

    def test_consume_failed_pending_proposals_removes_all_inflight_registries(self) -> None:
        proposal = Proposal(candidate=7, proposal_id="csa-0")
        state = build_engine_state()
        state = replace(
            state,
            proposal_state=state.proposal_state.register_pending_attribution(
                ProposalAttribution(
                    proposal_id="csa-0",
                    source_score=10.0,
                    proposal_family_key="regular",
                ),
            ),
        ).issue_proposal(proposal, tracks_generation=True)

        next_state = state.consume_failed_pending_proposals({"csa-0"})

        assert next_state.pending_proposals.is_empty
        assert next_state.generation_state.pending_proposal_ids == frozenset()
        assert next_state.proposal_state.pending_attributions == ()
        assert state.proposal_state.pending_attributions != ()

    def test_progression_masks_merge_stage_and_refresh_masks(self) -> None:
        state = build_engine_state()
        progression_state = replace(
            state.progression_state,
            stage_state=state.progression_state.stage_state.with_masks(
                seed_mask=frozenset({1}),
                partner_mask=frozenset({2}),
            ),
        ).with_refresh_mask(frozenset({0}))

        assert progression_state.seed_mask == frozenset({0, 1})
        assert progression_state.partner_mask == frozenset({0, 2})

    def test_without_updated_seed_mask_removes_refresh_mask_entries(self) -> None:
        progression_state = build_engine_state().progression_state.with_refresh_mask(
            frozenset({0, 1}),
        )

        next_state = progression_state.without_updated_seed_mask({1})

        assert next_state.refresh_mask == frozenset({0})

    def test_progression_masks_remap_after_bank_removal(self) -> None:
        state = build_engine_state()
        progression_state = replace(
            state.progression_state,
            stage_state=state.progression_state.stage_state.with_masks(
                seed_mask=frozenset({0, 2, 4}),
                partner_mask=frozenset({1, 3, 4}),
            ),
        ).with_refresh_mask(frozenset({2, 4}))

        next_state = progression_state.remove_indices(
            removed_indices=frozenset({1, 4}),
            entry_count=3,
        )

        assert next_state.stage_state.seed_mask == frozenset({0, 1})
        assert next_state.stage_state.partner_mask == frozenset({2})
        assert next_state.refresh_mask == frozenset({1})


class CSAAskEngineTests:
    """Regression tests for extracted ask-side engine planning."""

    def test_plan_next_ask_samples_space_before_bank_fill(self) -> None:
        plan = plan_next_ask(build_engine_state())

        assert plan == CSAAskPlan(kind="space_sample")

    def test_plan_next_ask_prefers_generation_dequeue_for_active_queue(self) -> None:
        state = build_engine_state()
        state = replace(
            state,
            banking_state=replace(
                state.banking_state,
                bank=Bank[int](
                    capacity=4,
                    entries=(
                        BankEntry(candidate=1, value=1.0),
                        BankEntry(candidate=2, value=4.0),
                        BankEntry(candidate=3, value=9.0),
                        BankEntry(candidate=4, value=16.0),
                    ),
                ),
                reference_bank=ReferenceBank[int](
                    capacity=4,
                    entries=(
                        BankEntry(candidate=1, value=1.0),
                        BankEntry(candidate=2, value=4.0),
                        BankEntry(candidate=3, value=9.0),
                        BankEntry(candidate=4, value=16.0),
                    ),
                ),
            ),
            generation_state=GenerationRuntimeState(
                queue=GenerationQueue(
                    candidates=(
                        GeneratedCandidate(candidate=11),
                        GeneratedCandidate(candidate=12),
                    ),
                ),
            ),
        )

        plan = plan_next_ask(state)

        assert plan == CSAAskPlan(kind="dequeue_generation")

    def test_commit_materialized_generation_begins_pool_and_dequeues_first_candidate(self) -> None:
        state = build_engine_state()
        materialized_generation = CSAMaterializedGeneration(
            selection_state=SeedSelectionState(),
            generation_queue=GenerationQueue(
                candidates=(
                    GeneratedCandidate(candidate=11),
                    GeneratedCandidate(candidate=12),
                ),
            ),
            trace_state=None,
        )

        candidate, next_state = commit_materialized_generation(
            state,
            materialized_generation,
        )

        assert candidate.candidate == 11
        assert next_state.generation_state.queue.candidates == (
            GeneratedCandidate(candidate=11),
            GeneratedCandidate(candidate=12),
        )
        assert next_state.generation_state.queue.head_index == 1


def build_engine_state() -> CSAEngineState[int]:
    return CSAEngineState(
        random_state=RandomStateSnapshot.from_seed(0),
        banking_state=CSABankingState(
            bank=Bank[int](capacity=4),
            reference_bank=ReferenceBank[int](capacity=4),
            refresh_state=None,
            growth_state=CSABankGrowthState[int].from_policy(
                CSABankGrowthPolicy(),
            ),
            clustering_state=CSAClusteringState[int](
                policy=CSAClusteringPolicy(),
            ),
        ),
        progression_state=CSAProgressionState(
            cutoff_state=CSACutoffState(),
            stage_state=CSAStageState(base_capacity=4, max_capacity=4),
        ),
        selection_state=SeedSelectionState(),
        generation_state=GenerationRuntimeState[int](),
        proposal_state=CSAProposalState.from_policy(CSAProposalPolicy()),
        scoring_state=CSAScoringState(
            acceptance_state=CSAAcceptanceState.from_policy(
                CSAAcceptancePolicy(),
            ),
            model_state=CSAScoreModelState(score_model=CSAScoreModel()),
        ),
        pending_proposals=CSAPendingProposals[int](),
    )
