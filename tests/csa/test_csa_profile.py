"""Tests for public CSA profile normalization."""

from collections.abc import Sequence

import numpy as np
import pytest
from typing_extensions import override

from tests.numeric_support import approx_equal
from variopt import VariationOperator
from variopt.algorithms.population.csa import (
    CSAAcceptancePolicy,
    CSABankGrowthPolicy,
    CSABankUpdatePolicy,
    CSABiasedPotential,
    CSAClusteringPolicy,
    CSACutoffSchedule,
    CSAPerturbationSchedule,
    CSAPerturbationSpec,
    CSAProfile,
    CSAProposalPolicy,
    CSARefreshPolicy,
    CSAScoreModel,
)


class CSAProfileTests:
    """Regression tests for CSA profile presets and normalization."""

    def test_variopt_preset_resolves_house_defaults(self) -> None:
        profile: CSAProfile[int] = CSAProfile(
            preset="variopt",
            perturbation_schedule=mutation_only_schedule(),
        )

        resolved = profile.resolve()

        assert resolved.seed_count == 5
        assert not (resolved.restart_lite)
        assert resolved.cutoff_schedule.stagnation_update_limit == 10
        assert approx_equal(resolved.cutoff_schedule.reduction_factor, 0.983912)
        assert resolved.update_policy.minimum_significant_score_gap == 0.001
        assert resolved.update_policy.local_update_mode == "normal"
        assert resolved.update_policy.far_update_mode == "crowding_aware"
        assert resolved.max_bank_capacity == 24
        assert resolved.proposal_policy == CSAProposalPolicy()
        expected_score_model: CSAScoreModel[int] = CSAScoreModel()
        assert resolved.score_model == expected_score_model

    def test_joung_2018_preset_resolves_literature_aligned_defaults(self) -> None:
        profile: CSAProfile[int] = CSAProfile(
            preset="joung_2018",
            perturbation_schedule=mutation_only_schedule(),
        )

        resolved = profile.resolve()

        assert resolved.seed_count == 6
        assert not (resolved.restart_lite)
        assert resolved.initial_new_bank_cut == 1
        assert resolved.cutoff_schedule.stagnation_update_limit == 10
        assert approx_equal(resolved.cutoff_schedule.reduction_factor, 0.983912)
        assert resolved.update_policy.minimum_significant_score_gap == 0.001
        assert resolved.update_policy.local_update_mode == "normal"
        assert resolved.update_policy.far_update_mode == "worst"
        assert resolved.proposal_policy == CSAProposalPolicy()
        assert resolved.refresh_policy == CSARefreshPolicy()
        expected_score_model: CSAScoreModel[int] = CSAScoreModel()
        assert resolved.score_model == expected_score_model

    def test_profile_can_override_variopt_max_bank_capacity(self) -> None:
        profile: CSAProfile[int] = CSAProfile(
            preset="variopt",
            perturbation_schedule=mutation_only_schedule(),
            max_bank_capacity=18,
        )

        resolved = profile.resolve()

        assert resolved.max_bank_capacity == 18

    def test_profile_can_override_named_default_update_policy(self) -> None:
        profile: CSAProfile[int] = CSAProfile(
            preset="variopt",
            perturbation_schedule=mutation_only_schedule(),
            cutoff_schedule=CSACutoffSchedule(initial_distance_cutoff=1.0),
            acceptance_policy=CSAAcceptancePolicy(initial_temperature=3.0),
            clustering_policy=CSAClusteringPolicy(
                enabled=True,
                cluster_cutoff_ratio=4.0,
                cluster_distance_ratio=2.0,
                update_mode="current_cluster",
            ),
            growth_policy=CSABankGrowthPolicy(
                enabled=True,
                maximum_capacity=9,
            ),
            refresh_policy=CSARefreshPolicy(
                mode="adaptive_refresh",
                preserve_fraction=0.5,
                newcomer_first_round=False,
            ),
            update_policy=CSABankUpdatePolicy(
                minimum_significant_score_gap=5.0,
                local_update_mode="disabled",
            ),
            score_model=CSAScoreModel(
                biased_potential=CSABiasedPotential(maximum_bias=7.0),
            ),
            proposal_policy=CSAProposalPolicy(enabled=True),
            cycle_limit=0,
        )

        resolved = profile.resolve()

        assert resolved.acceptance_policy.initial_temperature == 3.0
        assert resolved.clustering_policy.enabled
        assert resolved.clustering_policy.cluster_cutoff_ratio == 4.0
        assert resolved.clustering_policy.cluster_distance_ratio == 2.0
        assert resolved.clustering_policy.update_mode == "current_cluster"
        assert resolved.growth_policy.enabled
        assert resolved.growth_policy.maximum_capacity == 9
        assert resolved.refresh_policy == CSARefreshPolicy(
                mode="adaptive_refresh",
                preserve_fraction=0.5,
                newcomer_first_round=False,
            )
        assert resolved.update_policy.minimum_significant_score_gap == 5.0
        assert resolved.update_policy.local_update_mode == "disabled"
        assert resolved.proposal_policy.enabled
        assert resolved.score_model.biased_potential is not None
        assert resolved.score_model.biased_potential.maximum_bias == 7.0
        assert resolved.cycle_limit == 0
        assert resolved.cutoff_schedule.initial_distance_cutoff == 1.0

    def test_resolve_requires_perturbation_schedule(self) -> None:
        profile: CSAProfile[int] = CSAProfile()

        with pytest.raises(ValueError, match="perturbation_schedule must be provided"):
            _ = profile.resolve()

    def test_profile_rejects_removed_legacy_preset_name(self) -> None:
        with pytest.raises(ValueError, match="unsupported CSA preset: legacy_wrapper"):
            _ = CSAProfile(
                preset="legacy_wrapper",  # pyright: ignore[reportArgumentType]
                perturbation_schedule=mutation_only_schedule(),
            )

    def test_profile_no_longer_exposes_legacy_constructor_helpers(self) -> None:
        assert not (hasattr(CSAProfile, "legacy_engine"))
        assert not (hasattr(CSAProfile, "legacy_wrapper"))
        assert not (hasattr(CSAProfile, "legacy_benchmark"))


def mutation_only_schedule() -> CSAPerturbationSchedule[int]:
    return CSAPerturbationSchedule(
        mutation_family=(CSAPerturbationSpec(IdentityMutation()),),
    )


class IdentityMutation(VariationOperator[int]):
    """Test-only unary operator used to build valid perturbation schedules."""

    @property
    @override
    def arity(self) -> int:
        return 1

    @override
    def apply(
        self,
        parents: Sequence[int],
        random_state: np.random.RandomState,
    ) -> int:
        _ = random_state
        return parents[0]
