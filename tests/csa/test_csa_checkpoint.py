"""Regression tests for CSA safe-boundary checkpoint snapshots."""

from dataclasses import replace

import pytest

from variopt import IntegerSpace, Observation, Proposal
from variopt.algorithms.population.csa import CSAOptimizer, CSAProfile
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
    ReferenceRefreshState,
)
from variopt.algorithms.population.csa.engine import (
    CSABankingState,
    CSAEngineState,
    CSAPendingProposals,
    CSAScoringState,
)
from variopt.algorithms.population.csa.generation.proposal import (
    CSAProposalPolicy,
    CSAProposalState,
)
from variopt.algorithms.population.csa.generation.proposal.state import (
    ProposalAttribution,
    ProposalFamilyStat,
    ProposalLeafStat,
    ProposalNumericSubspaceCovarianceStat,
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
    PendingBoundaryAction,
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
from variopt.json_types import JSONValue
from variopt.randomness import RandomStateSnapshot


def _int_candidate_to_dict(candidate: int) -> JSONValue:
    return candidate


def _int_candidate_from_dict(data: JSONValue) -> int:
    if type(data) is not int:
        msg = "integer candidate snapshot must be an integer"
        raise TypeError(msg)
    return data


def _evaluate_proposals(
    proposals: tuple[Proposal[int], ...],
) -> tuple[Observation[int], ...]:
    return tuple(
        Observation(
            proposal=proposal,
            candidate=proposal.candidate,
            value=float(proposal.candidate * proposal.candidate),
            score=float(proposal.candidate * proposal.candidate),
        )
        for proposal in proposals
    )


def _advance_to_safe_boundary(
    optimizer: CSAOptimizer[int, int],
    state: CSAEngineState[int],
) -> tuple[
    tuple[tuple[tuple[Proposal[int], ...], tuple[Observation[int], ...]], ...],
    CSAEngineState[int],
]:
    trace: list[tuple[tuple[Proposal[int], ...], tuple[Observation[int], ...]]] = []
    next_state = state
    while True:
        proposals, next_state = optimizer.ask(next_state, batch_size=1)
        observations = _evaluate_proposals(proposals)
        next_state = optimizer.tell(next_state, observations)
        trace.append((proposals, observations))
        if next_state.pending_proposals.is_empty and not next_state.generation_state.is_active:
            return tuple(trace), next_state


def build_populated_engine_state() -> CSAEngineState[int]:
    growth_policy = CSABankGrowthPolicy(
        enabled=True,
        maximum_capacity=8,
        initial_energy_gap_limit=3.0,
    )
    clustering_policy = CSAClusteringPolicy(enabled=True)
    proposal_policy = CSAProposalPolicy(enabled=True, numeric_covariance_strength=0.5)
    acceptance_policy = CSAAcceptancePolicy(
        initial_temperature=2.0,
        reduction_factor=0.8,
        minimum_temperature=0.25,
    )
    score_model: CSAScoreModel[int] = CSAScoreModel()
    next_stage = CSAStageState(base_capacity=4, max_capacity=8, stage_index=2)

    return CSAEngineState(
        random_state=RandomStateSnapshot.from_seed(17),
        banking_state=CSABankingState(
            bank=Bank[int](
                capacity=4,
                entries=(
                    BankEntry(candidate=1, value=1.0, proposal_id="csa-0"),
                    BankEntry(candidate=2, value=4.0, proposal_id="csa-1"),
                    BankEntry(candidate=3, value=9.0, proposal_id="csa-2"),
                    BankEntry(candidate=4, value=16.0, proposal_id="csa-3"),
                ),
            ),
            reference_bank=ReferenceBank[int](
                capacity=4,
                entries=(
                    BankEntry(candidate=1, value=1.0, proposal_id="csa-0"),
                    BankEntry(candidate=2, value=4.0, proposal_id="csa-1"),
                    BankEntry(candidate=3, value=9.0, proposal_id="csa-2"),
                    BankEntry(candidate=4, value=16.0, proposal_id="csa-3"),
                ),
            ),
            refresh_state=None,
            growth_state=CSABankGrowthState[int](
                policy=growth_policy,
                active_energy_gap_limit=1.5,
                generation_growth_count=2,
            ),
            clustering_state=CSAClusteringState[int](
                policy=clustering_policy,
                cluster_distance=0.75,
                cluster_labels=(1, 1, 2, 2),
            ),
        ),
        progression_state=CSAProgressionState(
            cutoff_state=CSACutoffState(
                iteration_count=3,
                cycle_count=1,
                distance_cutoff=1.25,
                minimum_distance_cutoff=0.5,
                cutoff_recover_limit=1.75,
                previous_score_gap=0.2,
            ),
            stage_state=CSAStageState(
                base_capacity=4,
                max_capacity=8,
                stage_index=1,
                stage_round=1,
                seed_mask=frozenset({0, 1}),
                partner_mask=frozenset({2}),
            ),
            base_cycle_limit=5,
            restart_lite=False,
            pending_action=PendingBoundaryAction.stage_transition_action(
                (next_stage, True),
            ),
            stage_transition_count=2,
            refresh_count=1,
            refresh_mask=frozenset({3}),
        ),
        selection_state=SeedSelectionState(
            used_entry_indices=frozenset({1, 3}),
            bank_status=(False, True, False, True),
            active_seed_indices=(0, 2),
            next_seed_offset=1,
        ),
        generation_state=GenerationRuntimeState[int](),
        proposal_state=CSAProposalState(
            policy=proposal_policy,
            family_stats=(
                ProposalFamilyStat(
                    family_key="mutation",
                    observation_count=4,
                    discounted_score_credit=2.5,
                    last_update_index=7,
                ),
            ),
            leaf_stats=(
                ProposalLeafStat(
                    path=("x",),
                    observation_count=3,
                    discounted_score_credit=1.25,
                    last_update_index=7,
                    recent_failure_streak=1,
                ),
            ),
            local_displacement_leaf_stats=(
                ProposalLeafStat(
                    path=("y",),
                    observation_count=2,
                    discounted_score_credit=0.5,
                    last_update_index=7,
                ),
            ),
            numeric_covariance_stats=(
                ProposalNumericSubspaceCovarianceStat(
                    leaf_paths=(("x",), ("y",)),
                    observation_count=3,
                    discounted_weight=1.5,
                    discounted_displacement_sum=(0.3, -0.1),
                    discounted_outer_product_sum=((0.4, 0.05), (0.05, 0.2)),
                    last_update_index=7,
                ),
            ),
            update_index=7,
        ),
        scoring_state=CSAScoringState(
            acceptance_state=CSAAcceptanceState(
                policy=acceptance_policy,
                temperature=1.2,
            ),
            model_state=CSAScoreModelState(
                score_model=score_model,
                biased_potential_max=3.5,
            ),
        ),
        pending_proposals=CSAPendingProposals[int](),
        proposal_index=11,
    )


class CSAEngineCheckpointTests:
    """Regression tests for safe-boundary engine checkpoint snapshots."""

    def test_round_trips_populated_engine_state(self) -> None:
        state = build_populated_engine_state()

        snapshot = state.to_dict(candidate_to_dict=_int_candidate_to_dict)
        restored = CSAEngineState[int].from_dict(
            snapshot,
            candidate_from_dict=_int_candidate_from_dict,
            growth_policy=state.banking_state.growth_state.policy,
            clustering_policy=state.banking_state.clustering_state.policy,
            proposal_policy=state.proposal_state.policy,
            acceptance_policy=state.scoring_state.acceptance_state.policy,
            score_model=state.scoring_state.model_state.score_model,
        )

        assert restored.to_dict(candidate_to_dict=_int_candidate_to_dict) == snapshot

    def test_checkpoint_serializes_adaptation_state_not_refinement_payload(
        self,
    ) -> None:
        state = build_populated_engine_state()

        snapshot = state.to_dict(candidate_to_dict=_int_candidate_to_dict)

        proposal_state = snapshot["proposal_state"]
        assert isinstance(proposal_state, dict)
        assert proposal_state["local_displacement_leaf_stats"] == [
            {
                "path": ["y"],
                "observation_count": 2,
                "discounted_score_credit": 0.5,
                "last_update_index": 7,
                "recent_failure_streak": 0,
            },
        ]
        assert "refinement" not in repr(snapshot).lower()

    def test_rejects_checkpoint_when_pending_proposals_exist(self) -> None:
        state = build_populated_engine_state().issue_proposal(
            Proposal(candidate=7, proposal_id="csa-12"),
            tracks_generation=False,
        )

        with pytest.raises(ValueError, match="pending proposal registry"):
            _ = state.to_dict(candidate_to_dict=_int_candidate_to_dict)

    def test_rejects_checkpoint_when_generation_runtime_is_active(self) -> None:
        state = replace(
            build_populated_engine_state(),
            generation_state=GenerationRuntimeState[int]().begin(
                GenerationQueue(
                    candidates=(GeneratedCandidate(candidate=9),),
                ),
            ),
        )

        with pytest.raises(ValueError, match="generation runtime"):
            _ = state.to_dict(candidate_to_dict=_int_candidate_to_dict)

    def test_rejects_checkpoint_when_reference_refresh_is_active(self) -> None:
        state = replace(
            build_populated_engine_state(),
            banking_state=replace(
                build_populated_engine_state().banking_state,
                refresh_state=ReferenceRefreshState[int](target_capacity=4),
            ),
        )

        with pytest.raises(ValueError, match="reference refresh"):
            _ = state.to_dict(candidate_to_dict=_int_candidate_to_dict)

    def test_rejects_checkpoint_when_pending_attributions_exist(self) -> None:
        state = replace(
            build_populated_engine_state(),
            proposal_state=build_populated_engine_state().proposal_state.register_pending_attribution(
                ProposalAttribution(
                    proposal_id="csa-20",
                    source_score=1.0,
                ),
            ),
        )

        with pytest.raises(ValueError, match="pending attribution"):
            _ = state.to_dict(candidate_to_dict=_int_candidate_to_dict)


class CSAOptimizerCheckpointTests:
    """Regression tests for public optimizer checkpoint helpers."""

    def test_structured_optimizer_checkpoint_round_trip_matches_uninterrupted_run(self) -> None:
        optimizer = CSAOptimizer.from_space_defaults(
            space=IntegerSpace(-10, 10),
            bank_capacity=6,
            profile=CSAProfile(seed_count=3),
            random_state=7,
        )

        uninterrupted_state = optimizer.create_initial_state()
        for _ in range(2):
            _, uninterrupted_state = _advance_to_safe_boundary(
                optimizer,
                uninterrupted_state,
            )

        snapshot = optimizer.state_to_dict(uninterrupted_state)
        resumed_state = optimizer.state_from_dict(snapshot)

        assert optimizer.state_to_dict(resumed_state) == snapshot

        for _ in range(3):
            uninterrupted_trace, uninterrupted_state = _advance_to_safe_boundary(
                optimizer,
                uninterrupted_state,
            )
            resumed_trace, resumed_state = _advance_to_safe_boundary(
                optimizer,
                resumed_state,
            )
            assert uninterrupted_trace == resumed_trace

        assert optimizer.state_to_dict(resumed_state) == optimizer.state_to_dict(
            uninterrupted_state,
        )
