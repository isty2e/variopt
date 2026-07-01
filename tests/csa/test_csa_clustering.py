"""Tests for CSA clustering policy and runtime behavior."""

from collections.abc import Callable, Sequence

import numpy as np
from typing_extensions import override

from variopt import Observation, Proposal
from variopt.algorithms.population.csa import (
    Bank,
    CSAAcceptancePolicy,
    CSAAdaptivePotential,
    CSAAdaptivePotentialAxis,
    CSABankGrowthPolicy,
    CSABankUpdatePolicy,
    CSABiasedPotential,
    CSAClusteringPolicy,
    CSACutoffSchedule,
    CSANicheQualityPolicy,
    CSAScoreModel,
)
from variopt.algorithms.population.csa.banking.bank import BankEntry
from variopt.algorithms.population.csa.banking.clustering import (
    CSAClusteringState,
)
from variopt.algorithms.population.csa.banking.growth import (
    CSABankGrowthState,
)
from variopt.algorithms.population.csa.banking.update.logic import (
    apply_bank_update_batch,
    initialize_cutoff_if_needed,
)
from variopt.algorithms.population.csa.banking.update.result import (
    BankUpdateResult,
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
from variopt.algorithms.population.csa.scoring.acceptance_state import (
    CSAAcceptanceState,
)
from variopt.algorithms.population.csa.scoring.model_state import (
    CSAScoreModelState,
)
from variopt.diversity import DiversityMetric


class AbsoluteDistance(DiversityMetric[int]):
    """Absolute-value distance for integer candidates."""

    @override
    def distance(self, left: int, right: int) -> float:
        return float(abs(left - right))


class CountingAbsoluteDistance(AbsoluteDistance):
    """Absolute-value distance that records every concrete distance call."""

    call_count: int

    def __init__(self) -> None:
        self.call_count = 0

    @override
    def distance(self, left: int, right: int) -> float:
        self.call_count += 1
        return super().distance(left, right)


class CSAClusteringRuntimeTests:
    """Regression tests for CSA clustering semantics."""

    def test_explicit_cutoff_initialization_does_not_infer_average_distance(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=0, value=0.0),
                BankEntry(candidate=10, value=10.0),
            ),
        )
        state = CSAProgressionState(
            cutoff_state=CSACutoffState(),
            stage_state=CSAStageState(
                base_capacity=bank.capacity,
                max_capacity=bank.capacity,
            ),
        )

        next_state = initialize_cutoff_if_needed(
            bank=bank,
            state=state,
            infer_average_distance=reject_average_distance,
            infer_score_gap=infer_constant_score_gap,
            cutoff_schedule=CSACutoffSchedule(
                initial_distance_cutoff=2.0,
                minimum_distance_cutoff=1.0,
            ),
        )

        assert next_state.distance_cutoff == 2.0
        assert next_state.minimum_distance_cutoff == 1.0
        assert next_state.previous_score_gap == 10.0

    def test_explicit_initial_cutoff_mirrors_missing_minimum_without_inference(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=0, value=0.0),
                BankEntry(candidate=10, value=10.0),
            ),
        )
        state = CSAProgressionState(
            cutoff_state=CSACutoffState(),
            stage_state=CSAStageState(
                base_capacity=bank.capacity,
                max_capacity=bank.capacity,
            ),
        )

        next_state = initialize_cutoff_if_needed(
            bank=bank,
            state=state,
            infer_average_distance=reject_average_distance,
            infer_score_gap=infer_constant_score_gap,
            cutoff_schedule=CSACutoffSchedule(
                initial_distance_cutoff=2.0,
                minimum_distance_cutoff=None,
            ),
        )

        assert next_state.distance_cutoff == 2.0
        assert next_state.minimum_distance_cutoff == 2.0
        assert next_state.previous_score_gap == 10.0

    def test_disabled_clustering_does_not_infer_average_distance(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=0, value=0.0),
                BankEntry(candidate=10, value=10.0),
            ),
        )

        result = run_empty_cluster_batch(
            bank=bank,
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(enabled=False),
            ),
            infer_average_distance=reject_average_distance,
        )

        assert result.clustering_state == CSAClusteringState(
            policy=CSAClusteringPolicy(enabled=False),
        )

    def test_aligned_clustering_does_not_infer_average_distance(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=0, value=0.0),
                BankEntry(candidate=10, value=10.0),
            ),
        )
        clustering_state: CSAClusteringState[int] = CSAClusteringState(
            policy=CSAClusteringPolicy(enabled=True),
            cluster_distance=2.0,
            cluster_labels=(1, 2),
        )

        result = run_empty_cluster_batch(
            bank=bank,
            clustering_state=clustering_state,
            infer_average_distance=reject_average_distance,
        )

        assert result.clustering_state.cluster_distance == 2.0
        assert len(result.clustering_state.cluster_labels) == len(bank.entries)

    def test_aligned_empty_clustering_does_not_infer_average_distance(self) -> None:
        result = run_empty_cluster_batch(
            bank=Bank[int](capacity=2),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(enabled=True),
                cluster_distance=2.0,
                cluster_labels=(),
            ),
            infer_average_distance=reject_average_distance,
        )

        assert result.clustering_state.cluster_distance == 2.0
        assert result.clustering_state.cluster_labels == ()

    def test_misaligned_clustering_infers_average_distance_once(self) -> None:
        call_count = 0

        def count_average_distance(entries: Sequence[BankEntry[int]]) -> float:
            nonlocal call_count
            call_count += 1
            assert tuple(entry.candidate for entry in entries) == (0, 10)
            return 6.0

        result = run_empty_cluster_batch(
            bank=Bank(
                capacity=2,
                entries=(
                    BankEntry(candidate=0, value=0.0),
                    BankEntry(candidate=10, value=10.0),
                ),
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(enabled=True),
                cluster_distance=2.0,
                cluster_labels=(1,),
            ),
            infer_average_distance=count_average_distance,
        )

        assert call_count == 1
        assert result.clustering_state.cluster_distance == 4.0
        assert len(result.clustering_state.cluster_labels) == 2

    def test_inferred_cutoff_initialization_infers_average_distance_once(self) -> None:
        call_count = 0
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=0, value=0.0),
                BankEntry(candidate=10, value=10.0),
            ),
        )
        state = CSAProgressionState(
            cutoff_state=CSACutoffState(),
            stage_state=CSAStageState(
                base_capacity=bank.capacity,
                max_capacity=bank.capacity,
            ),
        )

        def count_average_distance(entries: Sequence[BankEntry[int]]) -> float:
            nonlocal call_count
            call_count += 1
            assert tuple(entry.candidate for entry in entries) == (0, 10)
            return 10.0

        next_state = initialize_cutoff_if_needed(
            bank=bank,
            state=state,
            infer_average_distance=count_average_distance,
            infer_score_gap=infer_constant_score_gap,
            cutoff_schedule=CSACutoffSchedule(),
        )

        assert call_count == 1
        assert next_state.distance_cutoff == 5.0
        assert next_state.minimum_distance_cutoff == 2.0
        assert next_state.previous_score_gap == 10.0

    def test_bank_update_initial_cutoff_uses_supplied_score_gap_callback(self) -> None:
        call_count = 0

        def count_score_gap(entries: Sequence[BankEntry[int]]) -> float:
            nonlocal call_count
            call_count += 1
            assert tuple(entry.candidate for entry in entries) == (0, 10)
            return 99.0

        growth_policy = CSABankGrowthPolicy()
        result = apply_bank_update_batch(
            bank=Bank(
                capacity=2,
                entries=(BankEntry(candidate=0, value=0.0),),
            ),
            state=CSAProgressionState(
                cutoff_state=CSACutoffState(),
                stage_state=CSAStageState(base_capacity=2, max_capacity=2),
            ),
            observations=(
                Observation(
                    proposal=Proposal(candidate=10, proposal_id="p-1"),
                    candidate=10,
                    value=10.0,
                    score=10.0,
                ),
            ),
            diversity_metric=AbsoluteDistance(),
            infer_average_distance=reject_average_distance,
            infer_score_gap=count_score_gap,
            cutoff_schedule=CSACutoffSchedule(
                initial_distance_cutoff=2.0,
                minimum_distance_cutoff=1.0,
            ),
            update_policy=CSABankUpdatePolicy(),
            acceptance_state=CSAAcceptanceState.from_policy(CSAAcceptancePolicy()),
            score_model_state=CSAScoreModelState(score_model=CSAScoreModel()),
            growth_state=CSABankGrowthState[int](
                policy=growth_policy,
                active_energy_gap_limit=growth_policy.initial_energy_gap_limit,
            ),
            clustering_state=CSAClusteringState(policy=CSAClusteringPolicy(enabled=False)),
            base_bank_capacity=2,
            masked_seed_indices=frozenset(),
            random_state=np.random.RandomState(0),
        )

        assert call_count == 1
        assert result.state.previous_score_gap == 99.0

    def test_appended_close_candidate_inherits_nearest_cluster(self) -> None:
        runtime: CSAClusteringState[int] = CSAClusteringState(
            policy=CSAClusteringPolicy(enabled=True),
            cluster_distance=3.0,
            cluster_labels=(1, 2),
        )

        next_runtime = runtime.register_admission(
            admitted_index=2,
            nearest_index=0,
            nearest_distance=1.0,
            appended=True,
        )

        assert next_runtime.cluster_labels == (1, 2, 1)

    def test_largest_cluster_mode_separates_comparison_and_removal_targets(self) -> None:
        runtime: CSAClusteringState[int] = CSAClusteringState(
            policy=CSAClusteringPolicy(
                enabled=True,
                update_mode="largest_cluster",
            ),
            cluster_distance=2.0,
            cluster_labels=(1, 1, 2, 2, 2),
        )

        decision = runtime.select_cluster_update(
            shaped_scores=(0.0, 10.0, 5.0, 20.0, 30.0),
            nearest_index=1,
        )

        assert decision is not None
        assert decision.comparison_index == 1
        assert decision.comparison_score == 10.0
        assert decision.remove_index == 4

    def test_cluster_update_largest_cluster_mode_replaces_largest_cluster_worst(self) -> None:
        batch_result = run_cluster_batch(
            bank=Bank(
                capacity=5,
                entries=(
                    BankEntry(candidate=0, value=0.0),
                    BankEntry(candidate=1, value=10.0),
                    BankEntry(candidate=10, value=5.0),
                    BankEntry(candidate=11, value=20.0),
                    BankEntry(candidate=12, value=30.0),
                ),
            ),
            observation=Observation(
                proposal=Proposal(candidate=2, proposal_id="p-1"),
                candidate=2,
                value=8.0,
                score=8.0,
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(
                    enabled=True,
                    update_mode="largest_cluster",
                ),
                cluster_distance=2.0,
                cluster_labels=(1, 1, 2, 2, 2),
            ),
            distance_cutoff=0.5,
        )

        assert batch_result.bank.entries[4].candidate == 2
        assert batch_result.bank.entries[4].value == 8.0
        assert batch_result.bank.entries[1].candidate == 1

    def test_cluster_update_current_cluster_mode_replaces_current_cluster_worst(self) -> None:
        batch_result = run_cluster_batch(
            bank=Bank(
                capacity=5,
                entries=(
                    BankEntry(candidate=0, value=0.0),
                    BankEntry(candidate=1, value=10.0),
                    BankEntry(candidate=10, value=5.0),
                    BankEntry(candidate=11, value=20.0),
                    BankEntry(candidate=12, value=30.0),
                ),
            ),
            observation=Observation(
                proposal=Proposal(candidate=2, proposal_id="p-1"),
                candidate=2,
                value=8.0,
                score=8.0,
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(
                    enabled=True,
                    update_mode="current_cluster",
                ),
                cluster_distance=2.0,
                cluster_labels=(1, 1, 2, 2, 2),
            ),
            distance_cutoff=0.5,
        )

        assert batch_result.bank.entries[1].candidate == 2
        assert batch_result.bank.entries[1].value == 8.0
        assert batch_result.bank.entries[4].candidate == 12

    def test_cluster_update_rejection_bumps_current_cluster_candidate(self) -> None:
        adaptive_potential = CSAAdaptivePotential(
            axes=(
                CSAAdaptivePotentialAxis(
                    reference_candidate=1,
                    minimum_distance=0.0,
                    maximum_distance=4.0,
                    bin_count=4,
                ),
            ),
            increment=2.0,
            overflow_energy=100.0,
        )
        score_model: CSAScoreModel[int] = CSAScoreModel(
            adaptive_potential=adaptive_potential,
        )
        initial_runtime = CSAScoreModelState(score_model=score_model)
        batch_result = run_cluster_batch(
            bank=Bank(
                capacity=5,
                entries=(
                    BankEntry(candidate=0, value=0.0),
                    BankEntry(candidate=1, value=10.0),
                    BankEntry(candidate=10, value=5.0),
                    BankEntry(candidate=11, value=20.0),
                    BankEntry(candidate=12, value=30.0),
                ),
            ),
            observation=Observation(
                proposal=Proposal(candidate=2, proposal_id="p-1"),
                candidate=2,
                value=50.0,
                score=50.0,
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(
                    enabled=True,
                    update_mode="largest_cluster",
                ),
                cluster_distance=2.0,
                cluster_labels=(1, 1, 2, 2, 2),
            ),
            distance_cutoff=0.5,
            score_model=score_model,
        )

        initial_trial = initial_runtime.score_trial(
            observation=Observation(
                proposal=Proposal(candidate=1, proposal_id="candidate-1"),
                candidate=1,
                value=10.0,
                score=10.0,
            ),
            bank_real_scores=(),
            entry_distances=(),
            diversity_metric=AbsoluteDistance(),
            distance_cutoff=0.5,
            minimum_distance_cutoff=0.5,
        )
        next_trial = batch_result.score_model_state.score_trial(
            observation=Observation(
                proposal=Proposal(candidate=1, proposal_id="candidate-1"),
                candidate=1,
                value=10.0,
                score=10.0,
            ),
            bank_real_scores=(),
            entry_distances=(),
            diversity_metric=AbsoluteDistance(),
            distance_cutoff=0.5,
            minimum_distance_cutoff=0.5,
        )

        assert batch_result.bank.entries[1].candidate == 1
        assert next_trial.shaped_score == initial_trial.shaped_score + 2.0

    def test_remove_top_uses_cluster_cutoff_when_clustering_is_enabled(self) -> None:
        batch_result = run_cluster_batch(
            bank=Bank(
                capacity=3,
                entries=(
                    BankEntry(candidate=0, value=0.0),
                    BankEntry(candidate=10, value=20.0),
                    BankEntry(candidate=30, value=100.0),
                ),
            ),
            observation=Observation(
                proposal=Proposal(candidate=20, proposal_id="p-1"),
                candidate=20,
                value=50.0,
                score=50.0,
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(enabled=True),
                cluster_distance=12.0,
                cluster_labels=(1, 1, 2),
            ),
            distance_cutoff=5.0,
        )

        assert tuple(entry.candidate for entry in batch_result.bank.entries) == (0, 10, 30)

    def test_crowded_worst_far_update_preserves_isolated_worst_entry(self) -> None:
        batch_result = run_cluster_batch(
            bank=Bank(
                capacity=3,
                entries=(
                    BankEntry(candidate=0, value=50.0),
                    BankEntry(candidate=1, value=40.0),
                    BankEntry(candidate=100, value=100.0),
                ),
            ),
            observation=Observation(
                proposal=Proposal(candidate=20, proposal_id="p-1"),
                candidate=20,
                value=60.0,
                score=60.0,
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(),
            ),
            distance_cutoff=2.0,
            update_policy=CSABankUpdatePolicy(far_update_mode="crowded_worst"),
        )

        assert tuple(entry.candidate for entry in batch_result.bank.entries) == (0, 1, 100)

    def test_crowding_aware_far_update_can_replace_isolated_worst_entry(self) -> None:
        batch_result = run_cluster_batch(
            bank=Bank(
                capacity=3,
                entries=(
                    BankEntry(candidate=0, value=50.0),
                    BankEntry(candidate=1, value=40.0),
                    BankEntry(candidate=100, value=100.0),
                ),
            ),
            observation=Observation(
                proposal=Proposal(candidate=20, proposal_id="p-1"),
                candidate=20,
                value=60.0,
                score=60.0,
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(),
            ),
            distance_cutoff=2.0,
            update_policy=CSABankUpdatePolicy(far_update_mode="crowding_aware"),
        )

        assert tuple(entry.candidate for entry in batch_result.bank.entries) == (0, 1, 20)

    def test_crowding_aware_far_update_can_penalize_poor_crowded_niche(self) -> None:
        batch_result = run_cluster_batch(
            bank=Bank(
                capacity=4,
                entries=(
                    BankEntry(candidate=0, value=10.0),
                    BankEntry(candidate=1, value=80.0),
                    BankEntry(candidate=100, value=55.0),
                    BankEntry(candidate=101, value=60.0),
                ),
            ),
            observation=Observation(
                proposal=Proposal(candidate=20, proposal_id="p-1"),
                candidate=20,
                value=58.0,
                score=58.0,
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(),
            ),
            distance_cutoff=2.0,
            update_policy=CSABankUpdatePolicy(
                far_update_mode="crowding_aware",
                crowding_penalty_ratio=1.0,
                niche_quality_policy=CSANicheQualityPolicy(
                    mode="mean",
                    ratio=1.0,
                ),
            ),
        )

        assert tuple(entry.candidate for entry in batch_result.bank.entries) == (0, 1, 100, 20)

    def test_crowding_aware_best_mean_mode_runs(self) -> None:
        best_mean_result = run_cluster_batch(
            bank=Bank(
                capacity=4,
                entries=(
                    BankEntry(candidate=0, value=10.0),
                    BankEntry(candidate=1, value=80.0),
                    BankEntry(candidate=100, value=55.0),
                    BankEntry(candidate=101, value=60.0),
                ),
            ),
            observation=Observation(
                proposal=Proposal(candidate=20, proposal_id="p-1"),
                candidate=20,
                value=58.0,
                score=58.0,
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(),
            ),
            distance_cutoff=2.0,
            update_policy=CSABankUpdatePolicy(
                far_update_mode="crowding_aware",
                crowding_penalty_ratio=1.0,
                niche_quality_policy=CSANicheQualityPolicy(
                    mode="best_mean",
                    ratio=1.0,
                ),
            ),
        )
        assert len(best_mean_result.bank.entries) == 4

    def test_bank_distance_workspace_is_shared_across_scoring_and_crowding(self) -> None:
        diversity_metric = CountingAbsoluteDistance()

        batch_result = run_cluster_batch(
            bank=Bank(
                capacity=3,
                entries=(
                    BankEntry(candidate=0, value=50.0),
                    BankEntry(candidate=1, value=40.0),
                    BankEntry(candidate=100, value=100.0),
                ),
            ),
            observation=Observation(
                proposal=Proposal(candidate=20, proposal_id="p-1"),
                candidate=20,
                value=60.0,
                score=60.0,
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(),
            ),
            distance_cutoff=2.0,
            score_model=CSAScoreModel(
                biased_potential=CSABiasedPotential(
                    maximum_bias=1.0,
                    sigma=1.0,
                    sigma_reference="constant",
                ),
            ),
            update_policy=CSABankUpdatePolicy(
                far_update_mode="crowding_aware",
                crowding_penalty_ratio=1.0,
                niche_quality_policy=CSANicheQualityPolicy(
                    mode="best_mean",
                    ratio=1.0,
                ),
            ),
            diversity_metric=diversity_metric,
        )

        assert len(batch_result.bank.entries) == 3
        expected_trial_bank_distances = 3
        expected_bank_pair_distances = 3
        assert (
            diversity_metric.call_count
            == expected_trial_bank_distances + expected_bank_pair_distances
        )


def run_cluster_batch(
    *,
    bank: Bank[int],
    observation: Observation[int],
    clustering_state: CSAClusteringState[int],
    distance_cutoff: float,
    score_model: CSAScoreModel[int] | None = None,
    update_policy: CSABankUpdatePolicy | None = None,
    diversity_metric: DiversityMetric[int] | None = None,
) -> BankUpdateResult[int]:
    resolved_score_model: CSAScoreModel[int]
    if score_model is None:
        resolved_score_model = CSAScoreModel()
    else:
        resolved_score_model = score_model
    resolved_update_policy = (
        CSABankUpdatePolicy()
        if update_policy is None
        else update_policy
    )
    growth_policy = CSABankGrowthPolicy()

    return apply_bank_update_batch(
        bank=bank,
        state=CSAProgressionState(
            cutoff_state=CSACutoffState(
                distance_cutoff=distance_cutoff,
                minimum_distance_cutoff=distance_cutoff,
                cutoff_recover_limit=distance_cutoff,
            ),
            stage_state=CSAStageState(
                base_capacity=bank.capacity,
                max_capacity=bank.capacity,
            ),
        ),
        observations=(observation,),
        diversity_metric=(
            AbsoluteDistance()
            if diversity_metric is None
            else diversity_metric
        ),
        infer_average_distance=lambda entries: infer_average_distance(entries),
        infer_score_gap=infer_constant_score_gap,
        cutoff_schedule=CSACutoffSchedule(
            initial_distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=distance_cutoff,
        ),
        update_policy=resolved_update_policy,
        acceptance_state=CSAAcceptanceState.from_policy(CSAAcceptancePolicy()),
        score_model_state=CSAScoreModelState(score_model=resolved_score_model),
        growth_state=CSABankGrowthState[int](
            policy=growth_policy,
            active_energy_gap_limit=growth_policy.initial_energy_gap_limit,
        ),
        clustering_state=clustering_state,
        base_bank_capacity=bank.capacity,
        masked_seed_indices=frozenset(),
        random_state=np.random.RandomState(0),
    )


def run_empty_cluster_batch(
    *,
    bank: Bank[int],
    clustering_state: CSAClusteringState[int],
    infer_average_distance: Callable[[Sequence[BankEntry[int]]], float],
) -> BankUpdateResult[int]:
    growth_policy = CSABankGrowthPolicy()
    return apply_bank_update_batch(
        bank=bank,
        state=CSAProgressionState(
            cutoff_state=CSACutoffState(
                distance_cutoff=2.0,
                minimum_distance_cutoff=1.0,
                cutoff_recover_limit=2.0,
            ),
            stage_state=CSAStageState(
                base_capacity=bank.capacity,
                max_capacity=bank.capacity,
            ),
        ),
        observations=(),
        diversity_metric=AbsoluteDistance(),
        infer_average_distance=infer_average_distance,
        infer_score_gap=infer_constant_score_gap,
        cutoff_schedule=CSACutoffSchedule(
            initial_distance_cutoff=2.0,
            minimum_distance_cutoff=1.0,
        ),
        update_policy=CSABankUpdatePolicy(),
        acceptance_state=CSAAcceptanceState.from_policy(CSAAcceptancePolicy()),
        score_model_state=CSAScoreModelState(score_model=CSAScoreModel()),
        growth_state=CSABankGrowthState[int](
            policy=growth_policy,
            active_energy_gap_limit=growth_policy.initial_energy_gap_limit,
        ),
        clustering_state=clustering_state,
        base_bank_capacity=bank.capacity,
        masked_seed_indices=frozenset(),
        random_state=np.random.RandomState(0),
    )


def reject_average_distance(entries: Sequence[BankEntry[int]]) -> float:
    _ = entries
    raise AssertionError("average distance should not be inferred")


def infer_average_distance(entries: Sequence[BankEntry[int]]) -> float:
    if len(entries) < 2:
        return 0.0

    total_distance = 0.0
    pair_count = 0
    for left_index, left_entry in enumerate(entries[:-1]):
        for right_entry in entries[left_index + 1 :]:
            total_distance += abs(left_entry.candidate - right_entry.candidate)
            pair_count += 1

    return total_distance / float(pair_count)


def infer_constant_score_gap(entries: Sequence[BankEntry[int]]) -> float | None:
    if not entries:
        return None

    return 10.0
