"""Tests for CSA cutoff observations and bounded reduction dispatch."""

from dataclasses import dataclass, replace
from math import isclose
from sys import float_info

import pytest
from typing_extensions import override

from tests.csa_support import (
    AbsoluteDistance,
    CSAOptimizerDriver,
    RepeatParent,
    make_optimizer,
)
from variopt import IntegerSpace, Observation, Proposal
from variopt.algorithms.population.csa.banking.bank import Bank, BankEntry
from variopt.algorithms.population.csa.progression.cutoff.logic import (
    advance_cutoff_state,
)
from variopt.algorithms.population.csa.progression.cutoff.observation import (
    CSACutoffObservation,
)
from variopt.algorithms.population.csa.progression.cutoff.policy import (
    CSACutoffSchedule,
)
from variopt.algorithms.population.csa.progression.cutoff.state import (
    CSACutoffState,
)
from variopt.algorithms.population.csa.progression.state import (
    CSAProgressionState,
)
from variopt.algorithms.population.csa.selection.state import SeedSelectionState


@dataclass(frozen=True, slots=True)
class CrowdedFractionSpeedSchedule(CSACutoffSchedule):
    """Test schedule driven by post-update bank crowding."""

    @property
    @override
    def requires_bank_crowding(self) -> bool:
        return True

    @override
    def resolve_reduction_speed(
        self,
        *,
        observation: CSACutoffObservation,
    ) -> float:
        fraction = observation.crowded_entry_fraction
        return 1.0 if fraction is None else 1.0 + fraction


@dataclass(frozen=True, slots=True)
class LocalRouteSpeedSchedule(CSACutoffSchedule):
    """Test schedule driven by same-batch local admission routes."""

    @override
    def resolve_reduction_speed(
        self,
        *,
        observation: CSACutoffObservation,
    ) -> float:
        fraction = observation.local_route_fraction
        return 1.0 if fraction is None else 1.0 + fraction


@dataclass(frozen=True, slots=True)
class UsedFractionSpeedSchedule(CSACutoffSchedule):
    """Test schedule that exposes mask-aware seed utilization."""

    @override
    def resolve_reduction_speed(
        self,
        *,
        observation: CSACutoffObservation,
    ) -> float:
        fraction = observation.used_entry_fraction
        return 1.0 if fraction is None else 1.0 + fraction


def cutoff_observation(
    *,
    score_gap: float | None = 2.0,
    eligible_entry_count: int = 4,
    unused_entry_count: int = 3,
    bank_entry_count: int = 5,
    crowded_entry_count: int | None = 2,
    cutoff_sensitive_transition_count: int = 4,
    local_transition_count: int = 1,
) -> CSACutoffObservation:
    """Build one valid observation with selected test overrides."""
    return CSACutoffObservation(
        score_gap=score_gap,
        eligible_entry_count=eligible_entry_count,
        unused_entry_count=unused_entry_count,
        bank_entry_count=bank_entry_count,
        crowded_entry_count=crowded_entry_count,
        cutoff_sensitive_transition_count=cutoff_sensitive_transition_count,
        local_transition_count=local_transition_count,
    )


def test_cutoff_observation_derives_normalized_fractions() -> None:
    observation = cutoff_observation()

    assert observation.score_gap == 2.0
    assert observation.crowded_entry_fraction == 0.4
    assert observation.local_route_fraction == 0.25
    assert observation.used_entry_fraction == 0.25


def test_cutoff_observation_distinguishes_unobserved_fractions() -> None:
    observation = cutoff_observation(
        eligible_entry_count=0,
        unused_entry_count=0,
        bank_entry_count=1,
        crowded_entry_count=0,
        cutoff_sensitive_transition_count=0,
        local_transition_count=0,
    )

    assert observation.crowded_entry_fraction is None
    assert observation.local_route_fraction is None
    assert observation.used_entry_fraction is None
    assert cutoff_observation(crowded_entry_count=None).crowded_entry_fraction is None


def test_cutoff_observation_rejects_boolean_counts() -> None:
    with pytest.raises(TypeError, match="eligible_entry_count must be an integer"):
        _ = cutoff_observation(eligible_entry_count=True)
    with pytest.raises(TypeError, match="unused_entry_count must be an integer"):
        _ = cutoff_observation(unused_entry_count=True)
    with pytest.raises(TypeError, match="bank_entry_count must be an integer"):
        _ = cutoff_observation(bank_entry_count=True)
    with pytest.raises(TypeError, match="crowded_entry_count must be an integer"):
        _ = cutoff_observation(crowded_entry_count=True)
    with pytest.raises(
        TypeError,
        match="cutoff_sensitive_transition_count must be an integer",
    ):
        _ = cutoff_observation(cutoff_sensitive_transition_count=True)
    with pytest.raises(TypeError, match="local_transition_count must be an integer"):
        _ = cutoff_observation(local_transition_count=True)


def test_cutoff_observation_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="eligible_entry_count must not exceed"):
        _ = cutoff_observation(bank_entry_count=3)
    with pytest.raises(ValueError, match="unused_entry_count must not exceed"):
        _ = cutoff_observation(unused_entry_count=5)
    with pytest.raises(ValueError, match="crowded_entry_count must not exceed"):
        _ = cutoff_observation(crowded_entry_count=6)
    with pytest.raises(
        ValueError,
        match="local_transition_count must not exceed",
    ):
        _ = cutoff_observation(cutoff_sensitive_transition_count=0)
    with pytest.raises(
        ValueError,
        match="fewer than two entries cannot be crowded",
    ):
        _ = cutoff_observation(
            bank_entry_count=1,
            eligible_entry_count=1,
            unused_entry_count=1,
            crowded_entry_count=1,
        )


@pytest.mark.parametrize("score_gap", [-1.0, float("nan"), float("inf")])
def test_cutoff_observation_rejects_invalid_score_gap(score_gap: float) -> None:
    with pytest.raises(
        ValueError,
        match="score_gap must be a finite non-negative float",
    ):
        _ = cutoff_observation(score_gap=score_gap)


def test_cutoff_observation_rejects_boolean_score_gap() -> None:
    with pytest.raises(TypeError, match="score_gap must be numeric"):
        _ = cutoff_observation(score_gap=True)


def test_cutoff_schedule_rejects_boolean_speed() -> None:
    with pytest.raises(TypeError, match="speed must be numeric"):
        _ = CSACutoffSchedule().reduce(
            distance_cutoff=4.0,
            minimum_distance_cutoff=1.0,
            speed=True,
        )


@pytest.mark.parametrize("speed", [0.0, -1.0, float("nan"), float("inf")])
def test_cutoff_schedule_rejects_invalid_speed(speed: float) -> None:
    with pytest.raises(ValueError, match="speed must be finite and positive"):
        _ = CSACutoffSchedule().reduce(
            distance_cutoff=4.0,
            minimum_distance_cutoff=1.0,
            speed=speed,
        )


@pytest.mark.parametrize(
    ("schedule", "distance_cutoff", "minimum_cutoff", "expected"),
    [
        (CSACutoffSchedule(reduction_factor=0.75), 4.0, 1.0, 3.0),
        (
            CSACutoffSchedule(
                reduction_method="linear",
                reduction_factor=0.75,
            ),
            4.0,
            1.0,
            3.25,
        ),
        (CSACutoffSchedule(reduction_factor=0.1), 4.0, 1.0, 1.0),
    ],
)
def test_fixed_cutoff_schedule_preserves_exact_reduction_algebra(
    schedule: CSACutoffSchedule,
    distance_cutoff: float,
    minimum_cutoff: float,
    expected: float,
) -> None:
    assert (
        schedule.reduce(
            distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=minimum_cutoff,
        )
        == expected
    )


def test_cutoff_schedule_scales_exponential_and_linear_reduction() -> None:
    assert (
        CSACutoffSchedule(reduction_factor=0.5).reduce(
            distance_cutoff=8.0,
            minimum_distance_cutoff=0.0,
            speed=2.0,
        )
        == 2.0
    )
    assert (
        CSACutoffSchedule(
            reduction_method="linear",
            reduction_factor=0.5,
        ).reduce(
            distance_cutoff=8.0,
            minimum_distance_cutoff=0.0,
            speed=2.0,
        )
        == 7.0
    )


def test_advance_cutoff_state_dispatches_observed_reduction_speed() -> None:
    state = CSAProgressionState().initialize_cutoff(
        distance_cutoff=8.0,
        minimum_distance_cutoff=0.0,
        previous_score_gap=2.0,
    )
    observation = cutoff_observation(
        cutoff_sensitive_transition_count=1,
        local_transition_count=1,
    )

    next_state, cycle_increment = advance_cutoff_state(
        state=state,
        schedule=LocalRouteSpeedSchedule(reduction_factor=0.5),
        observation=observation,
    )

    assert next_state.distance_cutoff == 2.0
    assert not cycle_increment


def test_optimizer_counts_crowded_entries_with_strict_cutoff() -> None:
    optimizer = make_optimizer(
        space=IntegerSpace(low=0, high=20),
        diversity_metric=AbsoluteDistance(),
        variation_operator=RepeatParent(),
        bank_capacity=4,
        random_state=0,
    ).optimizer
    entries = (
        BankEntry(candidate=0, value=0.0),
        BankEntry(candidate=4, value=1.0),
        BankEntry(candidate=8, value=2.0),
        BankEntry(candidate=8, value=3.0),
    )

    assert optimizer.infer_crowded_entry_count_for_entries((), 1.0) == 0
    assert optimizer.infer_crowded_entry_count_for_entries(entries[:1], 1.0) == 0
    assert optimizer.infer_crowded_entry_count_for_entries(entries, 4.0) == 2
    assert optimizer.infer_crowded_entry_count_for_entries(entries, 4.1) == 4


def test_optimizer_caps_overflowing_finite_score_gap() -> None:
    optimizer = make_optimizer(
        space=IntegerSpace(low=0, high=20),
        diversity_metric=AbsoluteDistance(),
        variation_operator=RepeatParent(),
        bank_capacity=2,
        random_state=0,
    ).optimizer
    entries = (
        BankEntry(candidate=0, value=-1e308),
        BankEntry(candidate=1, value=1e308),
    )

    assert optimizer.infer_score_gap_for_entries(entries) == float_info.max


def prime_bank_for_cutoff_test(
    *,
    cutoff_schedule: CSACutoffSchedule,
) -> tuple[CSAOptimizerDriver, Proposal[int]]:
    """Return a primed optimizer driver and one pending local proposal."""
    optimizer = make_optimizer(
        space=IntegerSpace(low=0, high=30),
        diversity_metric=AbsoluteDistance(),
        variation_operator=RepeatParent(),
        bank_capacity=3,
        cutoff_schedule=cutoff_schedule,
        random_state=0,
    )
    optimizer.bank = Bank(
        capacity=3,
        entries=(
            BankEntry(candidate=0, value=0.0, proposal_id="b0"),
            BankEntry(candidate=4, value=16.0, proposal_id="b1"),
            BankEntry(candidate=10, value=100.0, proposal_id="b2"),
        ),
    )
    optimizer.cutoff_state = CSACutoffState(
        distance_cutoff=5.0,
        minimum_distance_cutoff=0.0,
        cutoff_recover_limit=5.0,
    )
    proposal = Proposal(candidate=0, proposal_id="p0")
    optimizer.set_pending_proposals((proposal,))
    return optimizer, proposal


def tell_rejected_local_proposal(
    optimizer: CSAOptimizerDriver,
    proposal: Proposal[int],
) -> None:
    """Tell one inferior local proposal through a test driver."""
    optimizer.tell(
        (
            Observation(
                proposal=proposal,
                candidate=proposal.candidate,
                value=1000.0,
                score=1000.0,
            ),
        )
    )


def test_fixed_schedule_does_not_materialize_bank_crowding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer, proposal = prime_bank_for_cutoff_test(
        cutoff_schedule=CSACutoffSchedule(
            initial_distance_cutoff=5.0,
            minimum_distance_cutoff=0.0,
            reduction_factor=0.5,
        ),
    )

    def fail_if_called(*_arguments: object) -> int:
        raise AssertionError("fixed schedule must not request bank crowding")

    monkeypatch.setattr(
        type(optimizer.optimizer),
        "infer_crowded_entry_count_for_entries",
        fail_if_called,
    )

    tell_rejected_local_proposal(optimizer, proposal)

    assert optimizer.state.distance_cutoff == 2.5


def test_optimizer_supplies_post_update_bank_crowding() -> None:
    schedule = CrowdedFractionSpeedSchedule(
        initial_distance_cutoff=5.0,
        minimum_distance_cutoff=0.0,
        reduction_factor=0.5,
    )
    optimizer, proposal = prime_bank_for_cutoff_test(cutoff_schedule=schedule)

    tell_rejected_local_proposal(optimizer, proposal)

    expected_speed = 1.0 + 2.0 / 3.0
    distance_cutoff = optimizer.state.distance_cutoff
    assert distance_cutoff is not None
    assert isclose(
        distance_cutoff,
        5.0 * (0.5**expected_speed),
    )


def test_optimizer_supplies_same_batch_local_route_fraction() -> None:
    schedule = LocalRouteSpeedSchedule(
        initial_distance_cutoff=5.0,
        minimum_distance_cutoff=0.0,
        reduction_factor=0.5,
    )
    optimizer, proposal = prime_bank_for_cutoff_test(cutoff_schedule=schedule)

    tell_rejected_local_proposal(optimizer, proposal)

    assert optimizer.state.distance_cutoff == 1.25


def test_optimizer_excludes_valid_seed_masks_from_utilization() -> None:
    schedule = UsedFractionSpeedSchedule(
        initial_distance_cutoff=5.0,
        minimum_distance_cutoff=0.0,
        reduction_factor=0.5,
    )
    optimizer, proposal = prime_bank_for_cutoff_test(cutoff_schedule=schedule)
    optimizer.engine_state = replace(
        optimizer.engine_state,
        progression_state=replace(
            optimizer.engine_state.progression_state,
            stage_state=(
                optimizer.engine_state.progression_state.stage_state.with_masks(
                    seed_mask=frozenset({1, 99}),
                    partner_mask=frozenset(),
                )
            ),
        ),
        selection_state=SeedSelectionState(
            used_entry_indices=frozenset({0}),
            bank_status=(True, False, False),
        ),
    )

    tell_rejected_local_proposal(optimizer, proposal)

    distance_cutoff = optimizer.state.distance_cutoff
    assert distance_cutoff is not None
    assert isclose(distance_cutoff, 5.0 * (0.5**1.5))
