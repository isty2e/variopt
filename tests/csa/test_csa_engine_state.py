"""Tests for canonical CSA engine-state aggregates."""

from dataclasses import replace

import numpy as np
import pytest
from typing_extensions import override

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
from variopt.algorithms.population.csa.generation.proposal.evidence import (
    CSAProposalEvaluation,
)
from variopt.algorithms.population.csa.generation.proposal.state import (
    PlannedNonAdaptiveProposalAttribution,
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


def generated_candidate(candidate: int) -> GeneratedCandidate[int]:
    """Return a synthetic regular child with explicit non-adaptive provenance."""
    return GeneratedCandidate(
        candidate=candidate,
        planned_attribution=PlannedNonAdaptiveProposalAttribution(reason="regular"),
    )


class EqualityHostileCandidate:
    """Candidate that fails the test if proposal equality is used accidentally."""

    @override
    def __eq__(self, other: object) -> bool:
        del other
        raise AssertionError("candidate equality must not be used")


class CSAPendingProposalsTests:
    """Regression tests for canonical pending-proposal registry behavior."""

    def test_rejects_proposals_without_ids(self) -> None:
        with pytest.raises(ValueError, match="proposal ids"):
            _ = CSAPendingProposals[int](proposals=(Proposal(candidate=1),))

    def test_rejects_duplicate_proposal_ids(self) -> None:
        proposal = Proposal(candidate=1, proposal_id="csa-0")

        with pytest.raises(ValueError, match="distinct proposal ids"):
            _ = CSAPendingProposals[int](proposals=(proposal, proposal))

    def test_lookup_returns_identity_without_candidate_equality(self) -> None:
        proposals: tuple[Proposal[EqualityHostileCandidate], ...] = (
            Proposal(candidate=EqualityHostileCandidate(), proposal_id="csa-0"),
            Proposal(candidate=EqualityHostileCandidate(), proposal_id="csa-1"),
            Proposal(candidate=EqualityHostileCandidate(), proposal_id="csa-2"),
        )
        registry = CSAPendingProposals[EqualityHostileCandidate](proposals=proposals)

        assert registry.get("csa-1") is proposals[1]
        assert registry.get("missing") is None

    def test_add_rejects_duplicate_ids_without_candidate_equality(self) -> None:
        registry = CSAPendingProposals[EqualityHostileCandidate](
            proposals=(
                Proposal(candidate=EqualityHostileCandidate(), proposal_id="csa-0"),
            ),
        )
        duplicate = Proposal(
            candidate=EqualityHostileCandidate(),
            proposal_id="csa-0",
        )

        with pytest.raises(ValueError, match="distinct proposal ids"):
            _ = registry.add(duplicate)

    def test_add_appends_and_updates_lookup_index(self) -> None:
        first = Proposal(candidate=1, proposal_id="csa-0")
        second = Proposal(candidate=2, proposal_id="csa-1")
        registry = CSAPendingProposals[int](proposals=(first,))

        next_registry = registry.add(second)

        assert next_registry.proposals == (first, second)
        assert next_registry.get("csa-0") is first
        assert next_registry.get("csa-1") is second
        assert registry.get("csa-1") is None

    def test_remove_many_preserves_order_and_rebuilds_lookup_index(self) -> None:
        proposals = tuple(
            Proposal(candidate=index, proposal_id=f"csa-{index}") for index in range(5)
        )
        registry = CSAPendingProposals[int](proposals=proposals)

        next_registry = registry.remove_many({"csa-1", "csa-3", "missing"})

        assert next_registry.proposals == (proposals[0], proposals[2], proposals[4])
        assert next_registry.get("csa-0") is proposals[0]
        assert next_registry.get("csa-1") is None
        assert next_registry.get("csa-2") is proposals[2]
        assert next_registry.get("csa-3") is None
        assert next_registry.get("csa-4") is proposals[4]


class GenerationQueueTests:
    """Regression tests for CSA generated-candidate queue behavior."""

    def test_dequeue_advances_head_without_copying_candidates(self) -> None:
        queue = GenerationQueue(
            candidates=(
                generated_candidate(11),
                generated_candidate(12),
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
                candidates=(generated_candidate(11),),
                head_index=2,
            )

    def test_rejects_negative_head_index(self) -> None:
        with pytest.raises(ValueError, match="head_index"):
            _ = GenerationQueue[int](
                candidates=(generated_candidate(11),),
                head_index=-1,
            )

    def test_exhausted_nonempty_queue_behaves_as_empty(self) -> None:
        queue = GenerationQueue(
            candidates=(generated_candidate(11),),
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
                candidates=(generated_candidate(11),),
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
                candidates=(generated_candidate(11),),
                head_index=1,
            ),
            buffered_evaluations=(CSAProposalEvaluation.from_observation(observation),),
        )

        buffered_evaluations, idle_runtime = runtime.release_buffer()

        assert runtime.ready_to_commit
        assert buffered_evaluations == (
            CSAProposalEvaluation.from_observation(observation),
        )
        assert not idle_runtime.is_active

    def test_shuffled_queue_preserves_all_candidates_with_zero_head_index(self) -> None:
        candidates: tuple[GeneratedCandidate[int], ...] = (
            generated_candidate(1),
            generated_candidate(2),
            generated_candidate(3),
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


class GenerationIssuanceTests:
    def test_issue_preserves_candidate_identity_and_buffered_feedback(self) -> None:
        child = EqualityHostileCandidate()
        generated = GeneratedCandidate(candidate=child, planned_attribution=None)
        observation = Observation(
            proposal=Proposal(candidate=child, proposal_id="csa-0"),
            candidate=child,
            value=1.0,
            score=1.0,
        )
        feedback = (CSAProposalEvaluation.from_observation(observation),)
        runtime = GenerationRuntimeState(
            queue=GenerationQueue(candidates=(generated, generated)),
            pending_proposal_ids=frozenset({"csa-1"}),
            buffered_evaluations=feedback,
        )

        candidate, next_runtime = runtime.issue_next("csa-2")

        assert candidate is generated
        assert candidate.candidate is child
        assert next_runtime.queue.candidates is runtime.queue.candidates
        assert next_runtime.queue.head_index == 1
        assert next_runtime.pending_proposal_ids == frozenset({"csa-1", "csa-2"})
        assert next_runtime.buffered_evaluations is feedback
        assert runtime.queue.head_index == 0
        assert runtime.pending_proposal_ids == frozenset({"csa-1"})

    def test_duplicate_id_does_not_consume_queue(self) -> None:
        runtime = GenerationRuntimeState(
            queue=GenerationQueue(candidates=(generated_candidate(7),)),
            pending_proposal_ids=frozenset({"csa-0"}),
        )

        with pytest.raises(ValueError, match="already pending"):
            runtime.issue_next("csa-0")

        assert runtime.queue.head_index == 0
        assert runtime.pending_proposal_ids == frozenset({"csa-0"})

    def test_empty_queue_does_not_register_an_id(self) -> None:
        runtime = GenerationRuntimeState[int](
            pending_proposal_ids=frozenset({"csa-0"}),
        )

        with pytest.raises(RuntimeError, match="empty generation queue"):
            runtime.issue_next("csa-1")

        assert runtime.pending_proposal_ids == frozenset({"csa-0"})


class CSAEngineStateTests:
    """Regression tests for CSAEngineState invariants and helpers."""

    def test_sample_issuance_commits_id_pending_and_rng_together(self) -> None:
        state = build_engine_state()
        next_random_state = RandomStateSnapshot.from_seed(1)

        proposal, next_state = state.issue_sampled_proposal(
            7,
            random_state=next_random_state,
        )

        assert proposal == Proposal(candidate=7, proposal_id="csa-0")
        assert next_state.pending_proposals.get("csa-0") is proposal
        assert next_state.random_state is next_random_state
        assert next_state.generation_state is state.generation_state
        assert next_state.proposal_index == 1
        assert state.proposal_index == 0
        assert state.pending_proposals.is_empty
        assert state.random_state == RandomStateSnapshot.from_seed(0)

    def test_generation_issuance_advances_queue_and_both_registries(self) -> None:
        generated = generated_candidate(7)
        state = replace(
            build_engine_state(),
            proposal_index=41,
            generation_state=GenerationRuntimeState(
                queue=GenerationQueue(candidates=(generated,)),
            ),
        )

        proposal, planned, next_state = state.issue_generation_proposal()

        assert proposal.candidate is generated.candidate
        assert proposal.proposal_id == "csa-41"
        assert planned is generated.planned_attribution
        assert next_state.proposal_index == 42
        assert next_state.pending_proposals.get("csa-41") is proposal
        assert next_state.generation_state.pending_proposal_ids == frozenset({"csa-41"})
        assert next_state.generation_state.queue.is_empty
        assert next_state.random_state is state.random_state
        assert next_state.selection_state is state.selection_state
        assert next_state.proposal_state is state.proposal_state
        assert state.generation_state.queue.head_index == 0
        assert state.pending_proposals.is_empty
        assert state.proposal_index == 41

    def test_pending_collision_leaves_generation_and_counter_unchanged(self) -> None:
        state = replace(
            build_engine_state(),
            pending_proposals=CSAPendingProposals(
                proposals=(Proposal(candidate=1, proposal_id="csa-0"),),
            ),
            generation_state=GenerationRuntimeState(
                queue=GenerationQueue(candidates=(generated_candidate(7),)),
            ),
        )

        with pytest.raises(ValueError, match="distinct proposal ids"):
            state.issue_generation_proposal()

        assert state.generation_state.queue.head_index == 0
        assert state.generation_state.pending_proposal_ids == frozenset()
        assert state.proposal_index == 0
        assert len(state.pending_proposals.proposals) == 1

    def test_consume_pending_proposals_removes_registered_ids(self) -> None:
        initial_state = build_engine_state()
        proposal, state = initial_state.issue_sampled_proposal(
            7, random_state=initial_state.random_state
        )

        next_state = state.consume_pending_proposals({"csa-0"})

        assert next_state.pending_proposals.is_empty
        assert state.pending_proposals.get("csa-0") == proposal

    def test_consume_failed_pending_proposals_removes_all_inflight_registries(
        self,
    ) -> None:
        state = build_engine_state()
        state = replace(
            state,
            generation_state=GenerationRuntimeState(
                queue=GenerationQueue(candidates=(generated_candidate(7),)),
            ),
            proposal_state=state.proposal_state.register_pending_attribution(
                ProposalAttribution(
                    proposal_id="csa-0",
                    proposal_family_key="regular",
                ),
            ),
        )
        _, _, state = state.issue_generation_proposal()

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
                        generated_candidate(11),
                        generated_candidate(12),
                    ),
                ),
            ),
        )

        plan = plan_next_ask(state)

        assert plan == CSAAskPlan(kind="dequeue_generation")

    def test_commit_materialized_generation_preserves_first_child_until_issuance(
        self,
    ) -> None:
        state = build_engine_state()
        materialized_generation = CSAMaterializedGeneration(
            selection_state=SeedSelectionState(),
            generation_queue=GenerationQueue(
                candidates=(
                    generated_candidate(11),
                    generated_candidate(12),
                ),
            ),
            trace_state=None,
        )

        random_state = RandomStateSnapshot.from_seed(5)
        next_state = commit_materialized_generation(
            state,
            materialized_generation,
            random_state=random_state,
        )

        assert (
            next_state.generation_state.queue
            is materialized_generation.generation_queue
        )
        assert next_state.generation_state.queue.head_index == 0
        assert next_state.selection_state is materialized_generation.selection_state
        assert next_state.random_state is random_state
        assert next_state.pending_proposals is state.pending_proposals
        assert next_state.proposal_index == state.proposal_index
        assert not state.generation_state.is_active

        proposal, _, issued_state = next_state.issue_generation_proposal()

        assert proposal.candidate == 11
        assert issued_state.generation_state.queue.head_index == 1


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
