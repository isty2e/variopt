"""Regression tests for CSA safe-boundary checkpoint snapshots."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest
from typing_extensions import override

from variopt import (
    CategoricalSpace,
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
from variopt.algorithms.local_search import StructuredStochasticNeighborhoodKernel
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
    PlannedNonAdaptiveProposalAttribution,
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
from variopt.algorithms.population.csa.scoring.model import (
    CSAAdaptivePotential,
    CSAAdaptivePotentialAxis,
    CSAScoreModel,
)
from variopt.algorithms.population.csa.scoring.model_state import (
    CSAScoreModelState,
)
from variopt.algorithms.population.csa.selection.state import (
    SeedSelectionState,
)
from variopt.algorithms.population.csa.trace.events import CSAEventTraceState
from variopt.artifacts import ObservationPayload
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


class _CategoricalRankObjective(Objective[int]):
    """Categorical objective where lower integer labels are better."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate)


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


def _build_adaptive_score_model() -> CSAScoreModel[int]:
    return CSAScoreModel(
        adaptive_potential=CSAAdaptivePotential(
            axes=(
                CSAAdaptivePotentialAxis(
                    reference_candidate=0,
                    minimum_distance=0.0,
                    maximum_distance=1.0,
                    bin_count=2,
                ),
                CSAAdaptivePotentialAxis(
                    reference_candidate=1,
                    minimum_distance=0.0,
                    maximum_distance=1.0,
                    bin_count=2,
                ),
            ),
        ),
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

    def test_checkpoint_omits_trace_reducer_state(self) -> None:
        state = replace(
            build_populated_engine_state(),
            trace_state=CSAEventTraceState[int](),
        )

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

        assert "trace_state" not in snapshot
        assert restored.trace_state is None
        assert restored.to_dict(candidate_to_dict=_int_candidate_to_dict) == snapshot

    def test_checkpoint_rejects_trace_reducer_payload(self) -> None:
        state = build_populated_engine_state()
        snapshot = state.to_dict(candidate_to_dict=_int_candidate_to_dict)
        snapshot["trace_state"] = {"completed_generations": []}

        with pytest.raises(ValueError, match="trace_state"):
            _ = CSAEngineState[int].from_dict(
                snapshot,
                candidate_from_dict=_int_candidate_from_dict,
                growth_policy=state.banking_state.growth_state.policy,
                clustering_policy=state.banking_state.clustering_state.policy,
                proposal_policy=state.proposal_state.policy,
                acceptance_policy=state.scoring_state.acceptance_state.policy,
                score_model=state.scoring_state.model_state.score_model,
            )

    def test_clustering_state_snapshot_rejects_bool_and_non_finite_numbers(
        self,
    ) -> None:
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

        with pytest.raises(
            TypeError, match=r"cluster_labels\[0\] must be a JSON integer"
        ):
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

        with pytest.raises(
            TypeError, match="biased_potential_max must be a JSON number"
        ):
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

    def test_adaptive_potential_snapshot_restores_valid_potential(self) -> None:
        score_model = _build_adaptive_score_model()

        restored = CSAScoreModelState[int].from_dict(
            {
                "biased_potential_max": None,
                "adaptive_potential_state": {
                    "potential": [[1, 2.5], [3.25, 4]],
                },
            },
            score_model=score_model,
        )

        assert restored.adaptive_potential_state is not None
        assert restored.adaptive_potential_state.to_dict() == {
            "potential": [[1.0, 2.5], [3.25, 4.0]],
        }

    @pytest.mark.parametrize(
        ("potential", "expected_error", "match"),
        [
            (
                [[True, 0.0], [0.0, 0.0]],
                TypeError,
                r"potential\[0\]\[0\]",
            ),
            (
                [[float("nan"), 0.0], [0.0, 0.0]],
                ValueError,
                r"potential\[0\]\[0\]",
            ),
            (
                [[float("inf"), 0.0], [0.0, 0.0]],
                ValueError,
                r"potential\[0\]\[0\]",
            ),
            (
                [[0.0], [0.0, 0.0]],
                ValueError,
                r"potential\[0\]",
            ),
            (
                [[0.0, 0.0]],
                ValueError,
                "potential",
            ),
        ],
    )
    def test_adaptive_potential_snapshot_rejects_malformed_potential(
        self,
        potential: JSONValue,
        expected_error: type[Exception],
        match: str,
    ) -> None:
        score_model = _build_adaptive_score_model()

        with pytest.raises(expected_error, match=match):
            _ = CSAScoreModelState[int].from_dict(
                {
                    "biased_potential_max": None,
                    "adaptive_potential_state": {"potential": potential},
                },
                score_model=score_model,
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

        with pytest.raises(
            TypeError, match="recent_failure_streak must be a JSON integer"
        ):
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

        with pytest.raises(
            TypeError, match=r"path\[0\] must be a JSON integer or string"
        ):
            _ = ProposalLeafStat.from_dict(
                {
                    "path": [True],
                    "observation_count": 0,
                    "discounted_score_credit": 0.0,
                    "last_update_index": 0,
                    "recent_failure_streak": 0,
                },
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

        with pytest.raises(
            ValueError, match="discounted_displacement_sum\\[0\\] must be finite"
        ):
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

        progression_snapshot = (
            build_populated_engine_state().progression_state.to_dict()
        )
        progression_snapshot["refresh_mask"] = [True]
        with pytest.raises(
            TypeError, match=r"refresh_mask\[0\] must be a JSON integer"
        ):
            _ = CSAProgressionState.from_dict(progression_snapshot)

        with pytest.raises(
            TypeError, match=r"used_entry_indices\[0\] must be a JSON integer"
        ):
            _ = SeedSelectionState.from_dict(
                {
                    "used_entry_indices": [True],
                    "bank_status": [],
                    "active_seed_indices": [],
                    "next_seed_offset": 0,
                },
            )

        with pytest.raises(TypeError, match=r"bank_status\[0\] must be a JSON boolean"):
            _ = SeedSelectionState.from_dict(
                {
                    "used_entry_indices": [],
                    "bank_status": [1],
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

    def test_checkpoint_snapshots_reject_malformed_json_containers(self) -> None:
        with pytest.raises(TypeError, match="entries must be a JSON array"):
            _ = Bank[int].from_dict(
                {"capacity": 1, "entries": {}},
                candidate_from_dict=_int_candidate_from_dict,
            )

        with pytest.raises(TypeError, match=r"entries\[0\] must be a JSON object"):
            _ = Bank[int].from_dict(
                {"capacity": 1, "entries": [0]},
                candidate_from_dict=_int_candidate_from_dict,
            )

        with pytest.raises(TypeError, match=r"family_stats\[0\] must be a JSON object"):
            _ = CSAProposalState.from_dict(
                {
                    "pending_attributions": [],
                    "family_stats": [0],
                    "leaf_stats": [],
                    "local_displacement_leaf_stats": [],
                    "numeric_covariance_stats": [],
                    "update_index": 0,
                },
                policy=CSAProposalPolicy(),
            )

        state = build_populated_engine_state()
        engine_snapshot = state.to_dict(candidate_to_dict=_int_candidate_to_dict)
        engine_snapshot["random_state"] = []
        with pytest.raises(TypeError, match="random_state must be a JSON object"):
            _ = CSAEngineState[int].from_dict(
                engine_snapshot,
                candidate_from_dict=_int_candidate_from_dict,
                growth_policy=state.banking_state.growth_state.policy,
                clustering_policy=state.banking_state.clustering_state.policy,
                proposal_policy=state.proposal_state.policy,
                acceptance_policy=state.scoring_state.acceptance_state.policy,
                score_model=state.scoring_state.model_state.score_model,
            )

    def test_checkpoint_snapshots_reject_missing_required_fields(self) -> None:
        state = build_populated_engine_state()
        engine_snapshot = state.to_dict(candidate_to_dict=_int_candidate_to_dict)
        del engine_snapshot["random_state"]

        with pytest.raises(TypeError, match="random_state is required"):
            _ = CSAEngineState[int].from_dict(
                engine_snapshot,
                candidate_from_dict=_int_candidate_from_dict,
                growth_policy=state.banking_state.growth_state.policy,
                clustering_policy=state.banking_state.clustering_state.policy,
                proposal_policy=state.proposal_state.policy,
                acceptance_policy=state.scoring_state.acceptance_state.policy,
                score_model=state.scoring_state.model_state.score_model,
            )

        proposal_snapshot = state.proposal_state.to_dict()
        del proposal_snapshot["leaf_stats"]
        with pytest.raises(TypeError, match="leaf_stats is required"):
            _ = CSAProposalState.from_dict(
                proposal_snapshot,
                policy=state.proposal_state.policy,
            )

        numeric_stat_snapshot = ProposalNumericSubspaceCovarianceStat(
            leaf_paths=(("x",),),
            observation_count=1,
            discounted_weight=1.0,
            discounted_displacement_sum=(0.0,),
            discounted_outer_product_sum=((0.0,),),
            last_update_index=2,
        ).to_dict()
        del numeric_stat_snapshot["discounted_weight"]
        with pytest.raises(TypeError, match="discounted_weight is required"):
            _ = ProposalNumericSubspaceCovarianceStat.from_dict(numeric_stat_snapshot)

        action_snapshot: dict[str, JSONValue] = {
            "kind": "stage_transition",
            "stage_transition": {
                "stage_state": CSAStageState(
                    base_capacity=1,
                    max_capacity=2,
                ).to_dict(),
            },
        }
        with pytest.raises(
            TypeError,
            match=r"stage_transition\.refresh_required is required",
        ):
            _ = PendingBoundaryAction.from_dict(action_snapshot)

    def test_checkpoint_snapshots_distinguish_null_required_fields(self) -> None:
        state = build_populated_engine_state()
        engine_snapshot = state.to_dict(candidate_to_dict=_int_candidate_to_dict)
        engine_snapshot["random_state"] = None

        with pytest.raises(TypeError, match="random_state must be a JSON object"):
            _ = CSAEngineState[int].from_dict(
                engine_snapshot,
                candidate_from_dict=_int_candidate_from_dict,
                growth_policy=state.banking_state.growth_state.policy,
                clustering_policy=state.banking_state.clustering_state.policy,
                proposal_policy=state.proposal_state.policy,
                acceptance_policy=state.scoring_state.acceptance_state.policy,
                score_model=state.scoring_state.model_state.score_model,
            )

        with pytest.raises(TypeError, match="leaf_stats must be a JSON array"):
            _ = CSAProposalState.from_dict(
                {
                    "pending_attributions": [],
                    "family_stats": [],
                    "leaf_stats": None,
                    "local_displacement_leaf_stats": [],
                    "numeric_covariance_stats": [],
                    "update_index": 0,
                },
                policy=state.proposal_state.policy,
            )

        with pytest.raises(TypeError, match="discounted_weight must be a JSON number"):
            _ = ProposalNumericSubspaceCovarianceStat.from_dict(
                {
                    "leaf_paths": [["x"]],
                    "observation_count": 1,
                    "discounted_weight": None,
                    "discounted_displacement_sum": [0.0],
                    "discounted_outer_product_sum": [[0.0]],
                    "last_update_index": 2,
                },
            )

    def test_checkpoint_snapshot_domain_guards_run_before_deep_shape_checks(
        self,
    ) -> None:
        state = build_populated_engine_state()
        engine_snapshot = state.to_dict(candidate_to_dict=_int_candidate_to_dict)
        engine_snapshot["format"] = "other"
        engine_snapshot["random_state"] = []
        with pytest.raises(ValueError, match="unsupported CSA checkpoint format"):
            _ = CSAEngineState[int].from_dict(
                engine_snapshot,
                candidate_from_dict=_int_candidate_from_dict,
                growth_policy=state.banking_state.growth_state.policy,
                clustering_policy=state.banking_state.clustering_state.policy,
                proposal_policy=state.proposal_state.policy,
                acceptance_policy=state.scoring_state.acceptance_state.policy,
                score_model=state.scoring_state.model_state.score_model,
            )

        version_snapshot = state.to_dict(candidate_to_dict=_int_candidate_to_dict)
        version_snapshot["version"] = 2
        version_snapshot["banking_state"] = []
        with pytest.raises(ValueError, match="unsupported CSA checkpoint version"):
            _ = CSAEngineState[int].from_dict(
                version_snapshot,
                candidate_from_dict=_int_candidate_from_dict,
                growth_policy=state.banking_state.growth_state.policy,
                clustering_policy=state.banking_state.clustering_state.policy,
                proposal_policy=state.proposal_state.policy,
                acceptance_policy=state.scoring_state.acceptance_state.policy,
                score_model=state.scoring_state.model_state.score_model,
            )

        with pytest.raises(ValueError, match="reference refresh"):
            _ = CSABankingState[int].from_dict(
                {
                    "bank": {"capacity": 1, "entries": []},
                    "reference_bank": {"capacity": 1, "entries": []},
                    "refresh_state": {},
                    "growth_state": [],
                    "clustering_state": [],
                },
                candidate_from_dict=_int_candidate_from_dict,
                growth_policy=CSABankGrowthPolicy(),
                clustering_policy=CSAClusteringPolicy(),
            )

        with pytest.raises(ValueError, match="pending attribution"):
            _ = CSAProposalState.from_dict(
                {
                    "pending_attributions": [{"proposal_id": "csa-1"}],
                    "family_stats": {},
                    "leaf_stats": {},
                    "local_displacement_leaf_stats": {},
                    "numeric_covariance_stats": {},
                    "update_index": True,
                },
                policy=CSAProposalPolicy(),
            )

    def test_checkpoint_snapshot_nested_container_boundaries_use_field_paths(
        self,
    ) -> None:
        with pytest.raises(TypeError, match=r"leaf_paths\[0\] must be a JSON array"):
            _ = ProposalNumericSubspaceCovarianceStat.from_dict(
                {
                    "leaf_paths": [0],
                    "observation_count": 0,
                    "discounted_weight": 0.0,
                    "discounted_displacement_sum": [],
                    "discounted_outer_product_sum": [],
                    "last_update_index": 0,
                },
            )

        with pytest.raises(TypeError, match=r"entries\[0\] must be a JSON object"):
            _ = ReferenceBank[int].from_dict(
                {"capacity": 1, "entries": [0], "initialized": False},
                candidate_from_dict=_int_candidate_from_dict,
            )

        with pytest.raises(TypeError, match="initialized must be a JSON boolean"):
            _ = ReferenceBank[int].from_dict(
                {"capacity": 1, "entries": [], "initialized": 1},
                candidate_from_dict=_int_candidate_from_dict,
            )

        with pytest.raises(TypeError, match="leaf_stats must be a JSON array"):
            _ = CSAProposalState.from_dict(
                {
                    "pending_attributions": [],
                    "family_stats": [],
                    "leaf_stats": {},
                    "local_displacement_leaf_stats": [],
                    "numeric_covariance_stats": [],
                    "update_index": 0,
                },
                policy=CSAProposalPolicy(),
            )

    @pytest.mark.parametrize(
        ("field_name", "field_value", "expected_error", "match"),
        [
            (
                "distance_cutoff",
                True,
                TypeError,
                "distance_cutoff must be a JSON number",
            ),
            (
                "distance_cutoff",
                float("inf"),
                ValueError,
                "distance_cutoff must be finite",
            ),
            (
                "minimum_distance_cutoff",
                float("nan"),
                ValueError,
                "minimum_distance_cutoff must be finite",
            ),
            (
                "cutoff_recover_limit",
                float("inf"),
                ValueError,
                "cutoff_recover_limit must be finite",
            ),
            (
                "previous_score_gap",
                float("nan"),
                ValueError,
                "previous_score_gap must be finite",
            ),
        ],
    )
    def test_cutoff_state_snapshot_rejects_invalid_optional_numbers(
        self,
        field_name: str,
        field_value: JSONValue,
        expected_error: type[Exception],
        match: str,
    ) -> None:
        snapshot = CSACutoffState(
            distance_cutoff=1.0,
            minimum_distance_cutoff=0.5,
            cutoff_recover_limit=1.0,
            previous_score_gap=0.25,
        ).to_dict()
        snapshot[field_name] = field_value

        with pytest.raises(expected_error, match=match):
            _ = CSACutoffState.from_dict(snapshot)

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
                    candidates=(
                        GeneratedCandidate(
                            candidate=9,
                            planned_attribution=PlannedNonAdaptiveProposalAttribution(
                                reason="regular",
                            ),
                        ),
                    ),
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

    @pytest.mark.parametrize("version", [True, 1.0])
    def test_optimizer_checkpoint_rejects_non_integer_version(
        self,
        version: JSONValue,
    ) -> None:
        optimizer = CSAOptimizer.from_space_defaults(
            space=IntegerSpace(-10, 10),
            bank_capacity=6,
            profile=CSAProfile(seed_count=3),
            random_state=7,
        )
        snapshot = optimizer.state_to_dict(optimizer.create_initial_state())
        snapshot["version"] = version

        with pytest.raises(TypeError, match="version must be a JSON integer"):
            _ = optimizer.state_from_dict(snapshot)

    @pytest.mark.parametrize(
        ("field_name", "field_value", "expected_error", "match"),
        [
            ("position", True, TypeError, "position"),
            ("has_gaussian", True, TypeError, "has_gaussian"),
            ("cached_gaussian", float("inf"), ValueError, "cached_gaussian"),
            ("algorithm", "PCG64", ValueError, "algorithm"),
            ("key_hex", "not-hex", ValueError, "key_hex"),
            ("key_hex", "00000000", ValueError, "MT19937"),
        ],
    )
    def test_optimizer_checkpoint_rejects_malformed_random_state(
        self,
        field_name: str,
        field_value: JSONValue,
        expected_error: type[Exception],
        match: str,
    ) -> None:
        optimizer = CSAOptimizer.from_space_defaults(
            space=IntegerSpace(-10, 10),
            bank_capacity=6,
            profile=CSAProfile(seed_count=3),
            random_state=7,
        )
        snapshot = optimizer.state_to_dict(optimizer.create_initial_state())
        random_state_snapshot = snapshot["random_state"]
        assert isinstance(random_state_snapshot, dict)
        random_state_snapshot[field_name] = field_value

        with pytest.raises(expected_error, match=match):
            _ = optimizer.state_from_dict(snapshot)

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

    def test_trace_reducer_state_is_not_required_for_checkpoint_continuation(
        self,
    ) -> None:
        optimizer = CSAOptimizer.from_space_defaults(
            space=IntegerSpace(-10, 10),
            bank_capacity=6,
            profile=CSAProfile(seed_count=3),
            random_state=7,
        )

        traced_state = optimizer.create_state(trace_state=CSAEventTraceState[int]())
        _, traced_state = _advance_to_safe_boundary(optimizer, traced_state)
        snapshot = optimizer.state_to_dict(traced_state)
        resumed_state = optimizer.state_from_dict(snapshot)

        assert traced_state.trace_state is not None
        assert "trace_state" not in snapshot
        assert resumed_state.trace_state is None

        for _ in range(3):
            traced_trace, traced_state = _advance_to_safe_boundary(
                optimizer,
                traced_state,
            )
            resumed_trace, resumed_state = _advance_to_safe_boundary(
                optimizer,
                resumed_state,
            )
            assert resumed_trace == traced_trace

        assert optimizer.state_to_dict(resumed_state) == optimizer.state_to_dict(
            traced_state,
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

    def test_structured_local_search_checkpoint_resume_matches_continuation(
        self,
    ) -> None:
        space = CategoricalSpace(tuple(range(10)))

        def build_study() -> tuple[
            Study[int, int, CSAEngineState[int], ObservationPayload, Observation[int]],
            CSAOptimizer[int, int],
        ]:
            optimizer = CSAOptimizer.from_space_defaults(
                space=space,
                bank_capacity=6,
                profile=CSAProfile(seed_count=3),
                random_state=7,
            )
            study = Study(
                problem=Problem(
                    space=space,
                    objective=_CategoricalRankObjective(),
                ),
                run_method=optimizer,
                evaluator=SequentialEvaluator[int, int](),
                kernel=StructuredStochasticNeighborhoodKernel[int, int](
                    max_steps=1,
                    max_neighbors_per_step=1,
                    random_state=13,
                ),
            )
            return study, optimizer

        continuous_study, continuous_optimizer = build_study()
        _, checkpoint_state = continuous_study.run(
            max_evaluations=12,
            stop_at_checkpoint_boundary=True,
        )
        snapshot = continuous_optimizer.state_to_dict(checkpoint_state)
        continuous_report, continuous_state = continuous_study.run(
            max_evaluations=12,
            initial_state=checkpoint_state,
            stop_at_checkpoint_boundary=True,
        )

        resumed_study, resumed_optimizer = build_study()
        restored_state = resumed_optimizer.state_from_dict(snapshot)
        resumed_report, resumed_state = resumed_study.run(
            max_evaluations=12,
            initial_state=restored_state,
            stop_at_checkpoint_boundary=True,
        )

        assert resumed_report == continuous_report
        assert resumed_optimizer.state_to_dict(
            resumed_state,
        ) == continuous_optimizer.state_to_dict(continuous_state)

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

    def test_custom_checkpoint_decoder_must_return_valid_space_candidate(self) -> None:
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

        def invalid_candidate_from_dict(_data: JSONValue) -> RecordCandidate:
            return RecordCandidate(entries=(("depth", 2), ("x", 0.5)))

        with pytest.raises(ValueError, match="record candidate keys"):
            _ = optimizer.state_from_dict(
                snapshot,
                candidate_from_dict=invalid_candidate_from_dict,
            )

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
