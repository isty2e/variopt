"""Regression tests for CSA safe-boundary checkpoint snapshots."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest
from typing_extensions import override

from variopt import (
    IntegerSpace,
    Objective,
    Observation,
    Problem,
    Proposal,
    RealSpace,
    RecordSpace,
    Study,
    TupleSpace,
)
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
from variopt.evaluators import SequentialEvaluator
from variopt.json_types import JSONValue
from variopt.randomness import RandomStateSnapshot
from variopt.spaces import RecordCandidate, SpaceBoundaryValue, SpaceCandidateValue


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


class _SquareObjective(Objective[int]):
    """Integer square objective used for CSA study checkpoint tests."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


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
        if (
            next_state.pending_proposals.is_empty
            and not next_state.generation_state.is_active
        ):
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

    def test_clustering_state_snapshot_rejects_bool_and_non_finite_numbers(self) -> None:
        policy = CSAClusteringPolicy(enabled=True)

        with pytest.raises(TypeError, match="cluster_distance must be a JSON number"):
            _ = CSAClusteringState[int].from_dict(
                {"cluster_distance": True, "cluster_labels": []},
                policy=policy,
            )

        with pytest.raises(ValueError, match="cluster_distance must be finite"):
            _ = CSAClusteringState[int].from_dict(
                {"cluster_distance": float("inf"), "cluster_labels": []},
                policy=policy,
            )

        with pytest.raises(TypeError, match=r"cluster_labels\[0\] must be a JSON integer"):
            _ = CSAClusteringState[int].from_dict(
                {"cluster_distance": 1.0, "cluster_labels": [True]},
                policy=policy,
            )

    def test_scoring_state_snapshots_reject_bool_and_non_finite_numbers(self) -> None:
        acceptance_policy = CSAAcceptancePolicy()

        with pytest.raises(TypeError, match="temperature must be a JSON number"):
            _ = CSAAcceptanceState.from_dict(
                {"temperature": True},
                policy=acceptance_policy,
            )

        with pytest.raises(ValueError, match="temperature must be finite"):
            _ = CSAAcceptanceState.from_dict(
                {"temperature": float("nan")},
                policy=acceptance_policy,
            )

        score_model: CSAScoreModel[int] = CSAScoreModel()

        with pytest.raises(TypeError, match="biased_potential_max must be a JSON number"):
            _ = CSAScoreModelState[int].from_dict(
                {
                    "biased_potential_max": True,
                    "adaptive_potential_state": None,
                },
                score_model=score_model,
            )

        with pytest.raises(ValueError, match="biased_potential_max must be finite"):
            _ = CSAScoreModelState[int](
                score_model=score_model,
                biased_potential_max=float("inf"),
            )

    def test_proposal_stat_snapshots_reject_bool_and_non_finite_numbers(self) -> None:
        with pytest.raises(TypeError, match="observation_count must be a JSON integer"):
            _ = ProposalFamilyStat.from_dict(
                {
                    "family_key": "mutation",
                    "observation_count": True,
                    "discounted_score_credit": 0.0,
                    "last_update_index": 0,
                },
            )

        with pytest.raises(ValueError, match="discounted_score_credit must be finite"):
            _ = ProposalFamilyStat.from_dict(
                {
                    "family_key": "mutation",
                    "observation_count": 0,
                    "discounted_score_credit": float("inf"),
                    "last_update_index": 0,
                },
            )

        with pytest.raises(TypeError, match="recent_failure_streak must be a JSON integer"):
            _ = ProposalLeafStat.from_dict(
                {
                    "path": ["x"],
                    "observation_count": 0,
                    "discounted_score_credit": 0.0,
                    "last_update_index": 0,
                    "recent_failure_streak": True,
                },
            )

        with pytest.raises(ValueError, match="discounted_score_credit must be finite"):
            _ = ProposalLeafStat(
                path=("x",),
                discounted_score_credit=float("nan"),
            )

        with pytest.raises(TypeError, match="discounted_weight must be a JSON number"):
            _ = ProposalNumericSubspaceCovarianceStat.from_dict(
                {
                    "leaf_paths": [["x"]],
                    "observation_count": 0,
                    "discounted_weight": True,
                    "discounted_displacement_sum": [0.0],
                    "discounted_outer_product_sum": [[0.0]],
                    "last_update_index": 0,
                },
            )

        with pytest.raises(ValueError, match="discounted_displacement_sum\\[0\\] must be finite"):
            _ = ProposalNumericSubspaceCovarianceStat.from_dict(
                {
                    "leaf_paths": [["x"]],
                    "observation_count": 0,
                    "discounted_weight": 1.0,
                    "discounted_displacement_sum": [float("inf")],
                    "discounted_outer_product_sum": [[0.0]],
                    "last_update_index": 0,
                },
            )

        with pytest.raises(
            ValueError,
            match=r"discounted_outer_product_sum\[0\]\[0\] must be finite",
        ):
            _ = ProposalNumericSubspaceCovarianceStat.from_dict(
                {
                    "leaf_paths": [["x"]],
                    "observation_count": 0,
                    "discounted_weight": 1.0,
                    "discounted_displacement_sum": [0.0],
                    "discounted_outer_product_sum": [[float("nan")]],
                    "last_update_index": 0,
                },
            )

    def test_checkpoint_snapshots_reject_bool_integer_fields(self) -> None:
        with pytest.raises(TypeError, match="capacity must be a JSON integer"):
            _ = Bank[int].from_dict(
                {"capacity": True, "entries": []},
                candidate_from_dict=_int_candidate_from_dict,
            )

        with pytest.raises(TypeError, match="capacity must be a JSON integer"):
            _ = ReferenceBank[int].from_dict(
                {"capacity": True, "entries": [], "initialized": False},
                candidate_from_dict=_int_candidate_from_dict,
            )

        with pytest.raises(TypeError, match=r"seed_mask\[0\] must be a JSON integer"):
            _ = CSAStageState.from_dict(
                {
                    "base_capacity": 1,
                    "max_capacity": 2,
                    "stage_index": 0,
                    "stage_round": 0,
                    "seed_mask": [True],
                    "partner_mask": [],
                },
            )

        progression_snapshot = build_populated_engine_state().progression_state.to_dict()
        progression_snapshot["refresh_mask"] = [True]
        with pytest.raises(TypeError, match=r"refresh_mask\[0\] must be a JSON integer"):
            _ = CSAProgressionState.from_dict(progression_snapshot)

        with pytest.raises(TypeError, match=r"used_entry_indices\[0\] must be a JSON integer"):
            _ = SeedSelectionState.from_dict(
                {
                    "used_entry_indices": [True],
                    "bank_status": [],
                    "active_seed_indices": [],
                    "next_seed_offset": 0,
                },
            )

        with pytest.raises(TypeError, match="update_index must be a JSON integer"):
            _ = CSAProposalState.from_dict(
                {
                    "pending_attributions": [],
                    "family_stats": [],
                    "leaf_stats": [],
                    "local_displacement_leaf_stats": [],
                    "numeric_covariance_stats": [],
                    "update_index": True,
                },
                policy=CSAProposalPolicy(),
            )

        state = build_populated_engine_state()
        engine_snapshot = state.to_dict(candidate_to_dict=_int_candidate_to_dict)
        engine_snapshot["proposal_index"] = True
        with pytest.raises(TypeError, match="proposal_index must be a JSON integer"):
            _ = CSAEngineState[int].from_dict(
                engine_snapshot,
                candidate_from_dict=_int_candidate_from_dict,
                growth_policy=state.banking_state.growth_state.policy,
                clustering_policy=state.banking_state.clustering_state.policy,
                proposal_policy=state.proposal_state.policy,
                acceptance_policy=state.scoring_state.acceptance_state.policy,
                score_model=state.scoring_state.model_state.score_model,
            )

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

    def test_structured_optimizer_checkpoint_round_trip_matches_uninterrupted_run(
        self,
    ) -> None:
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

    def test_study_optimize_can_stop_at_checkpoint_safe_boundary(self) -> None:
        optimizer = CSAOptimizer.from_space_defaults(
            space=IntegerSpace(-10, 10),
            bank_capacity=6,
            profile=CSAProfile(seed_count=3),
            random_state=7,
        )
        study = Study(
            problem=Problem(
                space=IntegerSpace(-10, 10),
                objective=_SquareObjective(),
            ),
            run_method=optimizer,
            evaluator=SequentialEvaluator[int, int](),
        )

        full_result, unsafe_state = study.optimize(max_evaluations=7)
        safe_result, safe_state = study.optimize(
            max_evaluations=7,
            stop_at_checkpoint_boundary=True,
        )

        assert full_result.evaluation_count == 7
        assert not optimizer.is_checkpoint_safe_state(unsafe_state)
        with pytest.raises(ValueError):
            _ = optimizer.state_to_dict(unsafe_state)

        assert 0 < safe_result.evaluation_count < full_result.evaluation_count
        assert optimizer.is_checkpoint_safe_state(safe_state)
        snapshot = optimizer.state_to_dict(safe_state)
        restored_state = optimizer.state_from_dict(snapshot)
        assert optimizer.state_to_dict(restored_state) == snapshot

    def test_record_space_checkpoint_restore_preserves_record_candidates(self) -> None:
        space = RecordSpace(
            x=RealSpace(0.0, 1.0),
            depth=IntegerSpace(0, 5),
        )
        candidate = space.normalize({"x": 0.5, "depth": 2})
        optimizer: CSAOptimizer[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ] = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            profile=CSAProfile(seed_count=1),
            random_state=7,
        )
        entries = (
            BankEntry(
                candidate=candidate,
                value=1.25,
                proposal_id="csa-0",
            ),
        )
        initial_state = optimizer.create_initial_state()
        state = replace(
            initial_state,
            banking_state=replace(
                initial_state.banking_state,
                bank=Bank[RecordCandidate](capacity=4, entries=entries),
                reference_bank=ReferenceBank[RecordCandidate](
                    capacity=4,
                    entries=entries,
                ),
            ),
        )

        snapshot = optimizer.state_to_dict(state)
        restored = optimizer.state_from_dict(snapshot)

        restored_candidate = restored.banking_state.bank.entries[0].candidate
        assert isinstance(restored_candidate, RecordCandidate)
        assert restored_candidate == candidate
        space.validate(restored_candidate)
        assert optimizer.state_to_dict(restored) == snapshot

    def test_nested_record_space_checkpoint_restore_preserves_record_candidates(
        self,
    ) -> None:
        space = TupleSpace(
            RecordSpace(depth=IntegerSpace(0, 5)),
            IntegerSpace(0, 5),
        )
        candidate = space.normalize([{"depth": 2}, 1])
        optimizer: CSAOptimizer[
            Sequence[SpaceBoundaryValue],
            tuple[SpaceCandidateValue, ...],
        ] = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            profile=CSAProfile(seed_count=1),
            random_state=7,
        )
        entries = (
            BankEntry(
                candidate=candidate,
                value=1.25,
                proposal_id="csa-0",
            ),
        )
        initial_state = optimizer.create_initial_state()
        state = replace(
            initial_state,
            banking_state=replace(
                initial_state.banking_state,
                bank=Bank[tuple[SpaceCandidateValue, ...]](
                    capacity=4,
                    entries=entries,
                ),
                reference_bank=ReferenceBank[tuple[SpaceCandidateValue, ...]](
                    capacity=4,
                    entries=entries,
                ),
            ),
        )

        snapshot = optimizer.state_to_dict(state)
        restored = optimizer.state_from_dict(snapshot)

        restored_candidate = restored.banking_state.bank.entries[0].candidate
        assert isinstance(restored_candidate, tuple)
        assert isinstance(restored_candidate[0], RecordCandidate)
        assert restored_candidate == candidate
        space.validate(restored_candidate)
        assert optimizer.state_to_dict(restored) == snapshot
