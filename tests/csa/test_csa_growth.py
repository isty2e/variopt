"""Tests for CSA bank-growth policy and runtime behavior."""

import pytest
from typing_extensions import override

from variopt.algorithms.population.csa import (
    Bank,
    CSABankGrowthPolicy,
    CSAScoreModel,
)
from variopt.algorithms.population.csa.banking.bank import BankEntry
from variopt.algorithms.population.csa.banking.growth import (
    CSABankGrowthState,
)
from variopt.algorithms.population.csa.banking.growth.logic import (
    advance_growth_state,
    reduce_bank_by_energy_cut,
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


class CSABankGrowthRuntimeTests:
    """Runtime-level regressions for CSA adaptive growth semantics."""

    def test_growth_policy_rejects_non_finite_initial_energy_gap_limit(self) -> None:
        with pytest.raises(ValueError, match="initial_energy_gap_limit must be finite"):
            _ = CSABankGrowthPolicy(initial_energy_gap_limit=float("inf"))

        with pytest.raises(ValueError, match="initial_energy_gap_limit must be finite"):
            _ = CSABankGrowthPolicy(initial_energy_gap_limit=float("nan"))

    def test_growth_policy_rejects_non_finite_energy_gap_update_factor(self) -> None:
        with pytest.raises(ValueError, match="energy_gap_update_factor must be finite"):
            _ = CSABankGrowthPolicy(energy_gap_update_factor=float("inf"))

        with pytest.raises(ValueError, match="energy_gap_update_factor must be finite"):
            _ = CSABankGrowthPolicy(energy_gap_update_factor=float("nan"))

    def test_growth_state_rejects_non_finite_active_energy_gap_limit(self) -> None:
        policy = CSABankGrowthPolicy()

        with pytest.raises(ValueError, match="active_energy_gap_limit must be finite"):
            _ = CSABankGrowthState[int](
                policy=policy,
                active_energy_gap_limit=float("inf"),
            )

    def test_growth_state_from_dict_rejects_bool_and_non_finite_numbers(self) -> None:
        policy = CSABankGrowthPolicy()

        with pytest.raises(
            TypeError, match="active_energy_gap_limit must be a JSON number"
        ):
            _ = CSABankGrowthState[int].from_dict(
                {
                    "active_energy_gap_limit": True,
                    "generation_growth_count": 0,
                },
                policy=policy,
            )

        with pytest.raises(ValueError, match="active_energy_gap_limit must be finite"):
            _ = CSABankGrowthState[int].from_dict(
                {
                    "active_energy_gap_limit": float("nan"),
                    "generation_growth_count": 0,
                },
                policy=policy,
            )

        with pytest.raises(ValueError, match="active_energy_gap_limit must be finite"):
            _ = CSABankGrowthState[int].from_dict(
                {
                    "active_energy_gap_limit": float("inf"),
                    "generation_growth_count": 0,
                },
                policy=policy,
            )

        with pytest.raises(
            TypeError, match="generation_growth_count must be a JSON integer"
        ):
            _ = CSABankGrowthState[int].from_dict(
                {
                    "active_energy_gap_limit": 1.0,
                    "generation_growth_count": True,
                },
                policy=policy,
            )

    def test_multiplicative_decay_updates_energy_gap_limit(self) -> None:
        score_model: CSAScoreModel[int] = CSAScoreModel()
        policy = CSABankGrowthPolicy(
            enabled=True,
            maximum_capacity=4,
            initial_energy_gap_limit=8.0,
            energy_gap_update_mode="multiplicative_decay",
            energy_gap_update_factor=0.5,
        )
        runtime = CSABankGrowthState[int](
            policy=policy,
            active_energy_gap_limit=policy.initial_energy_gap_limit,
        )
        bank: Bank[int] = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=0, value=0.0),
                BankEntry(candidate=4, value=4.0),
            ),
        )

        advanced_runtime = advance_growth_state(
            state=runtime,
            bank=bank,
            diversity_metric=AbsoluteDistance(),
            score_model_state=CSAScoreModelState(score_model=score_model),
            distance_cutoff=2.0,
            minimum_distance_cutoff=1.0,
        )

        assert advanced_runtime.active_energy_gap_limit == 4.0
        assert advanced_runtime.generation_growth_count == 0

    def test_max_score_ratio_updates_energy_gap_limit_from_bank_span(self) -> None:
        score_model: CSAScoreModel[int] = CSAScoreModel()
        policy = CSABankGrowthPolicy(
            enabled=True,
            maximum_capacity=4,
            initial_energy_gap_limit=8.0,
            energy_gap_update_mode="max_score_ratio",
            energy_gap_update_factor=0.25,
        )
        runtime = CSABankGrowthState[int](
            policy=policy,
            active_energy_gap_limit=policy.initial_energy_gap_limit,
        )
        bank: Bank[int] = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=0, value=2.0),
                BankEntry(candidate=4, value=10.0),
            ),
        )

        advanced_runtime = advance_growth_state(
            state=runtime,
            bank=bank,
            diversity_metric=AbsoluteDistance(),
            score_model_state=CSAScoreModelState(score_model=score_model),
            distance_cutoff=2.0,
            minimum_distance_cutoff=1.0,
        )

        assert advanced_runtime.active_energy_gap_limit == 2.0

    def test_reduce_bank_removes_oversized_high_energy_tail(self) -> None:
        score_model: CSAScoreModel[int] = CSAScoreModel()
        policy = CSABankGrowthPolicy(
            enabled=True,
            maximum_capacity=4,
            initial_energy_gap_limit=1.0,
        )
        runtime = CSABankGrowthState[int](
            policy=policy,
            active_energy_gap_limit=policy.initial_energy_gap_limit,
        )
        bank: Bank[int] = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=0, value=0.0),
                BankEntry(candidate=1, value=0.5),
                BankEntry(candidate=2, value=10.0),
            ),
        )

        reduced_bank, removed_indices, _ = reduce_bank_by_energy_cut(
            state=runtime,
            bank=bank,
            minimum_capacity=2,
            score_model_state=CSAScoreModelState(score_model=score_model),
            diversity_metric=AbsoluteDistance(),
            distance_cutoff=2.0,
            minimum_distance_cutoff=2.0,
        )

        assert removed_indices == frozenset({2})
        assert reduced_bank.capacity == 2
        assert tuple(entry.value for entry in reduced_bank.entries) == (0.0, 0.5)
