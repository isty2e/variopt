"""Tests for CSA acceptance and score-model runtime behavior."""

import numpy as np
import pytest
from typing_extensions import override

from variopt import DiversityMetric, Observation, Proposal
from variopt.algorithms.population.csa import (
    CSAAcceptancePolicy,
    CSAAdaptivePotential,
    CSAAdaptivePotentialAxis,
    CSABiasedPotential,
    CSAScoreModel,
)
from variopt.algorithms.population.csa.banking.bank import BankEntry
from variopt.algorithms.population.csa.scoring.acceptance_state import (
    CSAAcceptanceState,
)
from variopt.algorithms.population.csa.scoring.model_state import (
    CSAScoreModelState,
)


class AbsoluteDistance(DiversityMetric[int]):
    """Absolute-value distance for integer candidates."""

    @override
    def distance(self, left: int, right: int) -> float:
        return float(abs(left - right))


class CSAAcceptanceRuntimeTests:
    """Regression tests for CSA acceptance scheduling."""

    def test_zero_temperature_acceptance_is_deterministic(self) -> None:
        runtime = CSAAcceptanceState.from_policy(CSAAcceptancePolicy())

        assert not (runtime.requires_random_state)
        assert not (
            runtime.should_accept(
                trial_score=11.0,
                reference_score=10.0,
            )
        )

    def test_positive_temperature_can_accept_uphill_trial(self) -> None:
        runtime = CSAAcceptanceState.from_policy(
            CSAAcceptancePolicy(
                initial_temperature=2.0,
                reduction_factor=1.0,
                minimum_temperature=0.0,
                boltzmann_constant=1.0,
            )
        )

        accepted = runtime.should_accept(
            trial_score=11.0,
            reference_score=10.0,
            random_state=np.random.RandomState(0),
        )

        assert accepted

    def test_positive_temperature_requires_random_state(self) -> None:
        runtime = CSAAcceptanceState.from_policy(
            CSAAcceptancePolicy(
                initial_temperature=1.0,
                reduction_factor=1.0,
                minimum_temperature=0.0,
                boltzmann_constant=1.0,
            )
        )

        assert runtime.requires_random_state
        with pytest.raises(ValueError, match="random_state is required"):
            _ = runtime.should_accept(
                trial_score=11.0,
                reference_score=10.0,
            )

    def test_advance_respects_recovery_and_minimum_temperature(self) -> None:
        runtime = CSAAcceptanceState.from_policy(
            CSAAcceptancePolicy(
                initial_temperature=1.0,
                reduction_factor=0.5,
                minimum_temperature=3.0,
                recover=True,
            )
        )

        next_runtime = runtime.advance()

        assert next_runtime.temperature == 3.0


class CSAScoreModelRuntimeTests:
    """Regression tests for CSA score shaping."""

    def test_biased_potential_penalizes_worse_bank_entries(self) -> None:
        score_model: CSAScoreModel[int] = CSAScoreModel(
            biased_potential=CSABiasedPotential(
                maximum_bias=5.0,
                sigma=1.0,
                sigma_reference="constant",
            )
        )
        runtime = CSAScoreModelState(score_model=score_model)

        scored_bank, _ = runtime.score_bank(
            entries=(
                BankEntry(candidate=0, value=0.0),
                BankEntry(candidate=1, value=10.0),
            ),
            diversity_metric=AbsoluteDistance(),
            distance_cutoff=1.0,
            minimum_distance_cutoff=0.5,
            masked_entry_indices=frozenset(),
        )

        assert scored_bank.shaped_scores[0] == 0.0
        assert scored_bank.shaped_scores[1] > 10.0

    def test_adaptive_potential_increments_candidate_bin(self) -> None:
        axis: CSAAdaptivePotentialAxis[int] = CSAAdaptivePotentialAxis(
            reference_candidate=0,
            minimum_distance=0.0,
            maximum_distance=4.0,
            bin_count=4,
        )
        adaptive_potential: CSAAdaptivePotential[int] = CSAAdaptivePotential(
            axes=(axis,),
            increment=2.0,
            overflow_energy=100.0,
        )
        score_model: CSAScoreModel[int] = CSAScoreModel(
            adaptive_potential=adaptive_potential,
        )
        runtime = CSAScoreModelState(score_model=score_model)
        observation = Observation(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            value=5.0,
            score=5.0,
        )

        first_trial = runtime.score_trial(
            observation=observation,
            bank_real_scores=(),
            entry_distances=(),
            diversity_metric=AbsoluteDistance(),
            distance_cutoff=1.0,
            minimum_distance_cutoff=0.5,
        )
        next_runtime = runtime.bump_trial(first_trial)
        second_trial = next_runtime.score_trial(
            observation=observation,
            bank_real_scores=(),
            entry_distances=(),
            diversity_metric=AbsoluteDistance(),
            distance_cutoff=1.0,
            minimum_distance_cutoff=0.5,
        )

        assert first_trial.shaped_score == 5.0
        assert second_trial.shaped_score == 7.0
