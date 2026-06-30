"""Tests for CSA banking, score-model, and admission semantics."""

from typing import Literal, cast

import pytest
from typing_extensions import override

from tests.csa_support import (
    AbsoluteDistance,
    Bank,
    BankEntry,
    CSAAcceptancePolicy,
    CSAAdaptivePotential,
    CSAAdaptivePotentialAxis,
    CSABankGrowthPolicy,
    CSABankUpdatePolicy,
    CSABiasedPotential,
    CSACutoffState,
    CSANicheQualityPolicy,
    CSAOptimizerTestCase,
    CSAScoreModel,
    DiversityMetric,
    IntegerSpace,
    NaNDistance,
    NegativeDistance,
    Observation,
    Problem,
    Proposal,
    ReferenceBank,
    RepeatParent,
    ScriptedIntegerSpace,
    SequentialEvaluator,
    SquareObjective,
    admit_observation,
    evaluate_observations,
    make_optimizer,
    significant_update_indices,
)
from variopt.algorithms.population.csa.banking.queries import crowding_aware_scores


class CountingDistance(DiversityMetric[int]):
    """Absolute distance metric that counts pairwise distance calls."""

    call_count: int

    def __init__(self) -> None:
        self.call_count = 0

    @override
    def distance(self, left: int, right: int) -> float:
        self.call_count += 1
        return float(abs(left - right))


class BankUpdatePolicyTests:
    """Unit tests for CSA bank admission semantics."""

    def test_policy_rejects_negative_crowding_penalty_ratio(self) -> None:
        with pytest.raises(ValueError, match="crowding_penalty_ratio must be non-negative"):
            _ = CSABankUpdatePolicy(crowding_penalty_ratio=-0.1)

    def test_crowding_aware_best_mean_reuses_pairwise_distances(self) -> None:
        entries = (
            BankEntry(candidate=0, value=5.0),
            BankEntry(candidate=1, value=3.0),
            BankEntry(candidate=2, value=7.0),
            BankEntry(candidate=10, value=1.0),
        )
        metric = CountingDistance()

        scores = crowding_aware_scores(
            base_scores=tuple(entry.value for entry in entries),
            entries=entries,
            diversity_metric=metric,
            distance_cutoff=3.0,
            penalty_ratio=1.0,
            niche_quality_policy=CSANicheQualityPolicy(
                mode="best_mean",
                ratio=1.0,
            ),
        )

        assert len(scores) == len(entries)
        assert metric.call_count == 6

    def test_admit_appends_until_full(self) -> None:
        bank = Bank[int](capacity=2)
        policy = CSABankUpdatePolicy()
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-1"),
            candidate=3,
            value=9.0,
            score=9.0,
        )

        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=1.0,
        )

        assert bank.entries == ()
        assert len(updated_bank.entries) == 1
        assert updated_bank.entries[0] == BankEntry(candidate=3, value=9.0, proposal_id="p-1")

    def test_admit_replaces_nearest_when_within_cutoff_and_better(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=4, value=16.0, proposal_id="p-1"),
                BankEntry(candidate=8, value=64.0, proposal_id="p-2"),
            ),
        )
        policy = CSABankUpdatePolicy()
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-3"),
            candidate=3,
            value=9.0,
            score=9.0,
        )

        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=2.0,
        )

        assert updated_bank.entries[0] == BankEntry(candidate=3, value=9.0, proposal_id="p-3")
        assert updated_bank.entries[1] == bank.entries[1]

    def test_admit_treats_cutoff_equality_as_far_case(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="p-1"),
                BankEntry(candidate=10, value=100.0, proposal_id="p-2"),
            ),
        )
        policy = CSABankUpdatePolicy()
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=5, proposal_id="p-3"),
            candidate=5,
            value=25.0,
            score=25.0,
        )

        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=4.0,
        )

        assert updated_bank.entries[0] == bank.entries[0]
        assert updated_bank.entries[1] == BankEntry(candidate=5, value=25.0, proposal_id="p-3")

    def test_admit_replaces_worst_when_far_and_better(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="p-1"),
                BankEntry(candidate=8, value=64.0, proposal_id="p-2"),
            ),
        )
        policy = CSABankUpdatePolicy()
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=5, proposal_id="p-3"),
            candidate=5,
            value=25.0,
            score=25.0,
        )

        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=1.0,
        )

        assert updated_bank.entries[0] == bank.entries[0]
        assert updated_bank.entries[1] == BankEntry(candidate=5, value=25.0, proposal_id="p-3")

    def test_admit_crowded_worst_mode_preserves_isolated_far_worst(self) -> None:
        bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=0, value=50.0, proposal_id="p-1"),
                BankEntry(candidate=1, value=40.0, proposal_id="p-2"),
                BankEntry(candidate=100, value=100.0, proposal_id="p-3"),
            ),
        )
        policy = CSABankUpdatePolicy(far_update_mode="crowded_worst")
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=20, proposal_id="p-4"),
            candidate=20,
            value=60.0,
            score=60.0,
        )

        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=2.0,
        )

        assert updated_bank == bank

    def test_admit_crowded_worst_mode_falls_back_when_no_crowded_entries_exist(self) -> None:
        bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=0, value=1.0, proposal_id="p-1"),
                BankEntry(candidate=10, value=100.0, proposal_id="p-2"),
                BankEntry(candidate=30, value=30.0, proposal_id="p-3"),
            ),
        )
        policy = CSABankUpdatePolicy(far_update_mode="crowded_worst")
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=20, proposal_id="p-4"),
            candidate=20,
            value=25.0,
            score=25.0,
        )

        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=2.0,
        )

        assert updated_bank.entries[0] == bank.entries[0]
        assert updated_bank.entries[2] == bank.entries[2]
        assert updated_bank.entries[1] == BankEntry(candidate=20, value=25.0, proposal_id="p-4")

    def test_admit_crowding_aware_mode_can_replace_isolated_far_worst(self) -> None:
        bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=0, value=50.0, proposal_id="p-1"),
                BankEntry(candidate=1, value=40.0, proposal_id="p-2"),
                BankEntry(candidate=100, value=100.0, proposal_id="p-3"),
            ),
        )
        policy = CSABankUpdatePolicy(far_update_mode="crowding_aware")
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=20, proposal_id="p-4"),
            candidate=20,
            value=60.0,
            score=60.0,
        )

        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=2.0,
        )

        assert updated_bank.entries[0] == bank.entries[0]
        assert updated_bank.entries[1] == bank.entries[1]
        assert updated_bank.entries[2] == BankEntry(candidate=20, value=60.0, proposal_id="p-4")

    def test_admit_crowding_aware_mode_biases_toward_crowded_entries(self) -> None:
        bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=0, value=50.0, proposal_id="p-1"),
                BankEntry(candidate=1, value=40.0, proposal_id="p-2"),
                BankEntry(candidate=100, value=55.0, proposal_id="p-3"),
            ),
        )
        policy = CSABankUpdatePolicy(
            far_update_mode="crowding_aware",
            crowding_penalty_ratio=1.0,
        )
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=20, proposal_id="p-4"),
            candidate=20,
            value=45.0,
            score=45.0,
        )

        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=2.0,
        )

        assert updated_bank.entries[0] == BankEntry(candidate=20, value=45.0, proposal_id="p-4")
        assert updated_bank.entries[1] == bank.entries[1]
        assert updated_bank.entries[2] == bank.entries[2]

    def test_admit_crowding_aware_mode_can_bias_against_poor_crowded_niche(self) -> None:
        bank = Bank(
            capacity=4,
            entries=(
                BankEntry(candidate=0, value=10.0, proposal_id="p-1"),
                BankEntry(candidate=1, value=80.0, proposal_id="p-2"),
                BankEntry(candidate=100, value=55.0, proposal_id="p-3"),
                BankEntry(candidate=101, value=60.0, proposal_id="p-4"),
            ),
        )
        policy = CSABankUpdatePolicy(
            far_update_mode="crowding_aware",
            crowding_penalty_ratio=1.0,
            niche_quality_policy=CSANicheQualityPolicy(mode="mean", ratio=1.0),
        )
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=20, proposal_id="p-5"),
            candidate=20,
            value=58.0,
            score=58.0,
        )

        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=2.0,
        )

        assert tuple(entry.candidate for entry in updated_bank.entries) == (0, 1, 100, 20)

    def test_admit_crowding_aware_best_mean_mode_runs(self) -> None:
        bank = Bank(
            capacity=4,
            entries=(
                BankEntry(candidate=0, value=10.0, proposal_id="p-1"),
                BankEntry(candidate=1, value=80.0, proposal_id="p-2"),
                BankEntry(candidate=100, value=55.0, proposal_id="p-3"),
                BankEntry(candidate=101, value=60.0, proposal_id="p-4"),
            ),
        )
        policy = CSABankUpdatePolicy(
            far_update_mode="crowding_aware",
            crowding_penalty_ratio=1.0,
            niche_quality_policy=CSANicheQualityPolicy(
                mode="best_mean",
                ratio=1.0,
            ),
        )
        metric = AbsoluteDistance()
        observation = Observation(
            proposal=Proposal(candidate=20, proposal_id="p-5"),
            candidate=20,
            value=58.0,
            score=58.0,
        )
        updated_bank = admit_observation(
            policy=policy,
            bank=bank,
            observation=observation,
            diversity_metric=metric,
            distance_cutoff=2.0,
        )
        assert len(updated_bank.entries) == 4

    def test_rejects_negative_niche_quality_ratio(self) -> None:
        with pytest.raises(ValueError, match="ratio must be non-negative"):
            _ = CSANicheQualityPolicy(mode="mean", ratio=-1.0)

    def test_rejects_unknown_niche_quality_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            _ = CSANicheQualityPolicy(
                mode=cast(
                    Literal["disabled", "mean", "best_mean"],
                    cast(object, "not-a-mode"),
                ),
                ratio=1.0,
            )

    def test_admit_rejects_nan_diversity_distance(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=4, value=16.0, proposal_id="p-1"),
                BankEntry(candidate=8, value=64.0, proposal_id="p-2"),
            ),
        )
        policy = CSABankUpdatePolicy()
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-3"),
            candidate=3,
            value=9.0,
            score=9.0,
        )

        with pytest.raises(ValueError):
            _ = admit_observation(
                policy=policy,
                bank=bank,
                observation=observation,
                diversity_metric=NaNDistance(),
                distance_cutoff=2.0,
            )

    def test_admit_rejects_negative_diversity_distance(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=4, value=16.0, proposal_id="p-1"),
                BankEntry(candidate=8, value=64.0, proposal_id="p-2"),
            ),
        )
        policy = CSABankUpdatePolicy()
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-3"),
            candidate=3,
            value=9.0,
            score=9.0,
        )

        with pytest.raises(ValueError):
            _ = admit_observation(
                policy=policy,
                bank=bank,
                observation=observation,
                diversity_metric=NegativeDistance(),
                distance_cutoff=2.0,
            )


class CSABankingTests(CSAOptimizerTestCase):
    """White-box tests for CSA banking and score-model state transitions."""

    def test_significant_update_threshold_ignores_small_score_changes(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            update_policy=CSABankUpdatePolicy(minimum_significant_score_gap=2.0),
            random_state=0,
        )
        previous_bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=10, value=10.0, proposal_id="p-0"),
                BankEntry(candidate=20, value=20.0, proposal_id="p-1"),
            ),
        )
        next_bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=9, value=9.0, proposal_id="p-2"),
                BankEntry(candidate=20, value=20.0, proposal_id="p-1"),
            ),
        )

        updated_indices = significant_update_indices(
            previous_bank=previous_bank,
            next_bank=next_bank,
            minimum_significant_score_gap=(
                optimizer.bank_update_policy.minimum_significant_score_gap
            ),
        )

        assert updated_indices == set()

    def test_significant_update_threshold_marks_large_score_changes_and_new_entries(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=3,
            update_policy=CSABankUpdatePolicy(minimum_significant_score_gap=2.0),
            random_state=0,
        )
        previous_bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=10, value=10.0, proposal_id="p-0"),
                BankEntry(candidate=20, value=20.0, proposal_id="p-1"),
            ),
        )
        next_bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=6, value=6.0, proposal_id="p-2"),
                BankEntry(candidate=20, value=20.0, proposal_id="p-1"),
                BankEntry(candidate=30, value=30.0, proposal_id="p-3"),
            ),
        )

        updated_indices = significant_update_indices(
            previous_bank=previous_bank,
            next_bank=next_bank,
            minimum_significant_score_gap=(
                optimizer.bank_update_policy.minimum_significant_score_gap
            ),
        )

        assert updated_indices == {0, 2}

    def test_bank_update_policy_treats_cutoff_equality_as_far_case(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            random_state=0,
        )
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="p-1"),
                BankEntry(candidate=10, value=100.0, proposal_id="p-2"),
            ),
        )
        observation = Observation(
            proposal=Proposal(candidate=5, proposal_id="p-3"),
            candidate=5,
            value=25.0,
            score=25.0,
        )

        updated_bank = admit_observation(
            policy=optimizer.bank_update_policy,
            bank=bank,
            observation=observation,
            diversity_metric=AbsoluteDistance(),
            distance_cutoff=4.0,
        )

        assert updated_bank.entries[0] == bank.entries[0]
        assert updated_bank.entries[1] == BankEntry(candidate=5, value=25.0, proposal_id="p-3")

    def test_bank_update_policy_can_disable_local_update(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            update_policy=CSABankUpdatePolicy(local_update_mode="disabled"),
            random_state=0,
        )
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=10, value=10.0, proposal_id="p-1"),
                BankEntry(candidate=50, value=100.0, proposal_id="p-2"),
            ),
        )
        observation = Observation(
            proposal=Proposal(candidate=12, proposal_id="p-3"),
            candidate=12,
            value=9.0,
            score=9.0,
        )

        updated_bank = admit_observation(
            policy=optimizer.bank_update_policy,
            bank=bank,
            observation=observation,
            diversity_metric=AbsoluteDistance(),
            distance_cutoff=5.0,
        )

        assert updated_bank == bank

    def test_temperature_policy_does_not_bypass_local_update_maxscore_gate(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            acceptance_policy=CSAAcceptancePolicy(
                initial_temperature=2.0,
                reduction_factor=1.0,
                minimum_temperature=0.0,
                boltzmann_constant=1.0,
            ),
            random_state=0,
        )
        self.prime_full_bank(
            optimizer=optimizer,
            entries=(
                BankEntry(candidate=0, value=0.0, proposal_id="b-0"),
                BankEntry(candidate=10, value=10.0, proposal_id="b-1"),
            ),
            distance_cutoff=2.0,
        )
        proposal = Proposal(candidate=10, proposal_id="p-1")
        optimizer.pending_by_id = {"p-1": proposal}

        optimizer.tell(
            (
                Observation(
                    proposal=proposal,
                    candidate=10,
                    value=11.0,
                    score=11.0,
                ),
            )
        )

        assert optimizer.bank.entries[1].value == 10.0

    def test_tell_keeps_random_state_when_acceptance_is_deterministic(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            random_state=0,
        )
        self.prime_full_bank(
            optimizer=optimizer,
            entries=(
                BankEntry(candidate=0, value=0.0, proposal_id="b-0"),
                BankEntry(candidate=10, value=10.0, proposal_id="b-1"),
            ),
            distance_cutoff=2.0,
        )
        proposal = Proposal(candidate=50, proposal_id="p-1")
        optimizer.pending_by_id = {"p-1": proposal}
        previous_random_state = optimizer.engine_state.random_state

        optimizer.tell(
            (
                Observation(
                    proposal=proposal,
                    candidate=50,
                    value=11.0,
                    score=11.0,
                ),
            )
        )

        assert optimizer.engine_state.random_state == previous_random_state

    def test_biased_potential_does_not_bypass_local_update_maxscore_gate(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=3,
            score_model=CSAScoreModel(
                biased_potential=CSABiasedPotential(
                    maximum_bias=5.0,
                    sigma=1.0,
                    sigma_reference="constant",
                ),
            ),
            random_state=0,
        )
        self.prime_full_bank(
            optimizer=optimizer,
            entries=(
                BankEntry(candidate=0, value=0.0, proposal_id="b-0"),
                BankEntry(candidate=1, value=1.0, proposal_id="b-1"),
                BankEntry(candidate=2, value=10.0, proposal_id="b-2"),
            ),
            distance_cutoff=2.0,
        )
        proposal = Proposal(candidate=3, proposal_id="p-1")
        optimizer.pending_by_id = {"p-1": proposal}

        optimizer.tell(
            (
                Observation(
                    proposal=proposal,
                    candidate=3,
                    value=11.0,
                    score=11.0,
                ),
            )
        )

        assert optimizer.bank.entries[2].value == 10.0

    def test_rejected_metadynamics_local_update_increments_bank_candidate_bin(self) -> None:
        axis: CSAAdaptivePotentialAxis[int] = CSAAdaptivePotentialAxis(
            reference_candidate=1,
            minimum_distance=0.0,
            maximum_distance=2.0,
            bin_count=2,
        )
        adaptive_potential: CSAAdaptivePotential[int] = CSAAdaptivePotential(
            axes=(axis,),
            increment=2.0,
            overflow_energy=100.0,
        )
        score_model: CSAScoreModel[int] = CSAScoreModel(
            adaptive_potential=adaptive_potential,
        )
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            score_model=score_model,
            random_state=0,
        )
        self.prime_full_bank(
            optimizer=optimizer,
            entries=(
                BankEntry(candidate=0, value=0.0, proposal_id="b-0"),
                BankEntry(candidate=1, value=10.0, proposal_id="b-1"),
            ),
            distance_cutoff=2.0,
        )
        proposal = Proposal(candidate=1, proposal_id="p-1")
        optimizer.pending_by_id = {"p-1": proposal}

        optimizer.tell(
            (
                Observation(
                    proposal=proposal,
                    candidate=1,
                    value=11.0,
                    score=11.0,
                ),
            )
        )

        assert optimizer.bank.entries[1].value == 10.0
        adaptive_state = optimizer.score_model_state.adaptive_potential_state
        assert adaptive_state is not None
        adaptive_energy, adaptive_bin_index = adaptive_state.score_candidate(
            candidate=1,
            diversity_metric=AbsoluteDistance(),
        )
        assert adaptive_bin_index == (0,)
        assert adaptive_energy == 2.0

    def test_growth_policy_can_append_far_candidate(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            growth_policy=CSABankGrowthPolicy(
                enabled=True,
                maximum_capacity=3,
                initial_energy_gap_limit=20.0,
            ),
            random_state=0,
        )
        self.prime_full_bank(
            optimizer=optimizer,
            entries=(
                BankEntry(candidate=0, value=0.0, proposal_id="b-0"),
                BankEntry(candidate=10, value=10.0, proposal_id="b-1"),
            ),
            distance_cutoff=2.0,
        )
        proposal = Proposal(candidate=20, proposal_id="p-1")
        optimizer.pending_by_id = {"p-1": proposal}

        optimizer.tell(
            (
                Observation(
                    proposal=proposal,
                    candidate=20,
                    value=5.0,
                    score=5.0,
                ),
            )
        )

        assert optimizer.bank.capacity == 3
        assert (
            tuple(entry.value for entry in optimizer.bank.entries) == (0.0, 5.0, 10.0)
        )

    def test_growth_policy_reduces_oversized_bank_after_batch(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            growth_policy=CSABankGrowthPolicy(
                enabled=True,
                maximum_capacity=4,
                initial_energy_gap_limit=1.0,
            ),
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=0, value=0.0, proposal_id="b-0"),
                BankEntry(candidate=1, value=0.5, proposal_id="b-1"),
                BankEntry(candidate=2, value=10.0, proposal_id="b-2"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=3,
            entries=optimizer.bank.entries,
        )
        optimizer.cutoff_state = CSACutoffState(
            distance_cutoff=2.0,
            minimum_distance_cutoff=2.0,
            cutoff_recover_limit=2.0,
        )
        proposal = Proposal(candidate=30, proposal_id="p-1")
        optimizer.pending_by_id = {"p-1": proposal}

        optimizer.tell(
            (
                Observation(
                    proposal=proposal,
                    candidate=30,
                    value=20.0,
                    score=20.0,
                ),
            )
        )

        assert optimizer.bank.capacity == 2
        assert (
            tuple(entry.value for entry in optimizer.bank.entries) == (0.0, 0.5)
        )

    def test_initial_fill_sorts_bank_and_reference_bank_by_score(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 1, 5)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=3,
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=3),
            )
        )

        assert (
            tuple(entry.candidate for entry in optimizer.bank.entries) == (1, 5, 9)
        )
        assert (
            tuple(entry.candidate for entry in optimizer.reference_bank.entries)
            == (1, 5, 9)
        )
