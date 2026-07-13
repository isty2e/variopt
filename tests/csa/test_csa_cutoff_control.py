"""Tests for CSA cutoff observations and bounded local-route control."""

from dataclasses import replace
from math import exp, isclose, log
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
from variopt.algorithms.population.csa.banking.queries import infer_score_gap
from variopt.algorithms.population.csa.progression.cutoff.logic import (
    advance_cutoff_state,
)
from variopt.algorithms.population.csa.progression.cutoff.observation import (
    CSACutoffObservation,
)
from variopt.algorithms.population.csa.progression.cutoff.policy import (
    CSACutoffSchedule,
    CSALocalRouteCutoffSchedule,
)
from variopt.algorithms.population.csa.progression.cutoff.state import (
    CSACutoffState,
)
from variopt.algorithms.population.csa.progression.state import (
    CSAProgressionState,
)


class DoubleSpeedCutoffSchedule(CSACutoffSchedule):
    """Custom schedule that requests observations by overriding its resolver."""

    @override
    def resolve_reduction_speed(
        self,
        *,
        observation: CSACutoffObservation,
    ) -> float:
        """Return a constant double-speed adaptive step."""
        _ = observation
        return 2.0


def cutoff_observation(
    *,
    score_gap: float | None = 2.0,
    unused_entry_count: int = 3,
    full_bank_transition_count: int = 4,
    local_transition_count: int = 1,
) -> CSACutoffObservation:
    """Build one valid cutoff observation with selected overrides."""
    return CSACutoffObservation(
        score_gap=score_gap,
        unused_entry_count=unused_entry_count,
        full_bank_transition_count=full_bank_transition_count,
        local_transition_count=local_transition_count,
    )


def test_cutoff_observation_derives_local_route_fraction() -> None:
    observation = cutoff_observation()

    assert observation.score_gap == 2.0
    assert observation.local_route_fraction == 0.25


@pytest.mark.parametrize(
    ("score_gap", "expected_score_gap"),
    [(None, None), (0.0, 0.0), (2, 2.0)],
)
def test_cutoff_observation_accepts_available_and_missing_score_gaps(
    score_gap: float | None,
    expected_score_gap: float | None,
) -> None:
    observation = cutoff_observation(score_gap=score_gap)

    assert observation.score_gap == expected_score_gap
    if expected_score_gap is not None:
        assert type(observation.score_gap) is float


def test_cutoff_observation_distinguishes_missing_route_evidence() -> None:
    observation = cutoff_observation(
        full_bank_transition_count=0,
        local_transition_count=0,
    )

    assert observation.local_route_fraction is None


def test_cutoff_observation_rejects_boolean_counts() -> None:
    with pytest.raises(TypeError, match="unused_entry_count must be an integer"):
        _ = cutoff_observation(unused_entry_count=True)
    with pytest.raises(
        TypeError,
        match="full_bank_transition_count must be an integer",
    ):
        _ = cutoff_observation(full_bank_transition_count=True)
    with pytest.raises(TypeError, match="local_transition_count must be an integer"):
        _ = cutoff_observation(local_transition_count=True)


def test_cutoff_observation_rejects_inconsistent_route_counts() -> None:
    with pytest.raises(
        ValueError,
        match="local_transition_count must not exceed full_bank_transition_count",
    ):
        _ = cutoff_observation(full_bank_transition_count=0)


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cutoff_schedule_rejects_non_finite_configuration(value: float) -> None:
    with pytest.raises(ValueError, match="initial_distance_cutoff must be finite"):
        _ = CSACutoffSchedule(initial_distance_cutoff=value)
    with pytest.raises(ValueError, match="minimum_distance_cutoff must be finite"):
        _ = CSACutoffSchedule(minimum_distance_cutoff=value)
    with pytest.raises(ValueError, match="initial_distance_divisor must be finite"):
        _ = CSACutoffSchedule(initial_distance_divisor=value)
    with pytest.raises(ValueError, match="minimum_distance_divisor must be finite"):
        _ = CSACutoffSchedule(minimum_distance_divisor=value)
    with pytest.raises(ValueError, match="reduction_factor must be finite"):
        _ = CSACutoffSchedule(reduction_factor=value)


def test_cutoff_schedule_rejects_boolean_configuration() -> None:
    with pytest.raises(TypeError, match="initial_distance_cutoff must be numeric"):
        _ = CSACutoffSchedule(initial_distance_cutoff=True)
    with pytest.raises(TypeError, match="minimum_distance_cutoff must be numeric"):
        _ = CSACutoffSchedule(minimum_distance_cutoff=True)
    with pytest.raises(TypeError, match="initial_distance_divisor must be numeric"):
        _ = CSACutoffSchedule(initial_distance_divisor=True)
    with pytest.raises(TypeError, match="minimum_distance_divisor must be numeric"):
        _ = CSACutoffSchedule(minimum_distance_divisor=True)
    with pytest.raises(TypeError, match="reduction_factor must be numeric"):
        _ = CSACutoffSchedule(reduction_factor=True)
    with pytest.raises(TypeError, match="stagnation_update_limit must be an integer"):
        _ = CSACutoffSchedule(stagnation_update_limit=True)
    with pytest.raises(TypeError, match="recover_steps must be an integer"):
        _ = CSACutoffSchedule(recover_steps=True)


def test_cutoff_schedule_rejects_non_boolean_cycle_gate() -> None:
    schedule = CSACutoffSchedule()
    object.__setattr__(
        schedule,
        "cycle_increment_requires_minimum_cutoff",
        1,
    )

    with pytest.raises(
        TypeError,
        match="cycle_increment_requires_minimum_cutoff must be a bool",
    ):
        schedule.__post_init__()


def test_cutoff_schedule_normalizes_numeric_configuration() -> None:
    schedule = CSACutoffSchedule(
        initial_distance_cutoff=4,
        minimum_distance_cutoff=1,
        initial_distance_divisor=2,
        minimum_distance_divisor=5,
        reduction_factor=1,
    )

    assert type(schedule.initial_distance_cutoff) is float
    assert type(schedule.minimum_distance_cutoff) is float
    assert type(schedule.initial_distance_divisor) is float
    assert type(schedule.minimum_distance_divisor) is float
    assert type(schedule.reduction_factor) is float


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


def test_cutoff_schedule_rejects_invalid_runtime_cutoffs() -> None:
    schedule = CSACutoffSchedule()
    with pytest.raises(TypeError, match="distance_cutoff must be numeric"):
        _ = schedule.reduce(
            distance_cutoff=True,
            minimum_distance_cutoff=0.0,
        )
    with pytest.raises(ValueError, match="distance_cutoff must be finite"):
        _ = schedule.reduce(
            distance_cutoff=float("nan"),
            minimum_distance_cutoff=0.0,
        )
    with pytest.raises(ValueError, match="minimum_distance_cutoff must be finite"):
        _ = schedule.reduce(
            distance_cutoff=1.0,
            minimum_distance_cutoff=float("inf"),
        )
    with pytest.raises(ValueError, match="distance_cutoff must be non-negative"):
        _ = schedule.reduce(
            distance_cutoff=-1.0,
            minimum_distance_cutoff=0.0,
        )
    with pytest.raises(
        ValueError,
        match="minimum_distance_cutoff must be non-negative",
    ):
        _ = schedule.reduce(
            distance_cutoff=1.0,
            minimum_distance_cutoff=-1.0,
        )
    with pytest.raises(
        ValueError,
        match="minimum_distance_cutoff must not exceed distance_cutoff",
    ):
        _ = schedule.reduce(
            distance_cutoff=1.0,
            minimum_distance_cutoff=2.0,
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
    assert (
        CSACutoffSchedule(reduction_factor=0.5).reduce(
            distance_cutoff=8.0,
            minimum_distance_cutoff=1.0,
            speed=1,
        )
        == 4.0
    )
    assert (
        CSACutoffSchedule(reduction_factor=0.5).reduce(
            distance_cutoff=8.0,
            minimum_distance_cutoff=1.0,
            speed=1e10,
        )
        == 1.0
    )


def test_local_route_schedule_uses_evidence_backed_defaults() -> None:
    schedule = CSALocalRouteCutoffSchedule()

    assert schedule.reduction_factor == 0.983912
    assert schedule.stagnation_update_limit == 10
    assert not schedule.cycle_increment_requires_minimum_cutoff
    assert schedule.target_local_route_fraction == 0.25
    assert schedule.response == 2.0
    assert schedule.requires_reduction_observation
    assert not CSACutoffSchedule().requires_reduction_observation


def test_replacing_local_route_schedule_preserves_adaptive_subtype() -> None:
    schedule = replace(CSALocalRouteCutoffSchedule(), response=3.0)

    assert type(schedule) is CSALocalRouteCutoffSchedule
    assert schedule.response == 3.0
    assert schedule.requires_reduction_observation


def test_local_route_schedule_is_neutral_without_evidence_or_at_target() -> None:
    schedule = CSALocalRouteCutoffSchedule()

    assert (
        schedule.resolve_reduction_speed(
            observation=cutoff_observation(
                full_bank_transition_count=0,
                local_transition_count=0,
            )
        )
        == 1.0
    )
    assert schedule.resolve_reduction_speed(observation=cutoff_observation()) == 1.0


def test_local_route_schedule_clamps_extreme_response_without_overflow() -> None:
    slow_schedule = CSALocalRouteCutoffSchedule(
        target_local_route_fraction=0.99,
        response=float_info.max,
    )
    fast_schedule = CSALocalRouteCutoffSchedule(
        target_local_route_fraction=0.01,
        response=float_info.max,
    )

    assert (
        slow_schedule.resolve_reduction_speed(
            observation=cutoff_observation(
                full_bank_transition_count=1,
                local_transition_count=0,
            )
        )
        == 0.25
    )
    assert (
        fast_schedule.resolve_reduction_speed(
            observation=cutoff_observation(
                full_bank_transition_count=1,
                local_transition_count=1,
            )
        )
        == 4.0
    )


def test_local_route_schedule_matches_registered_control_law() -> None:
    schedule = CSALocalRouteCutoffSchedule()
    for denominator in range(1, 25):
        for numerator in range(denominator + 1):
            observed_fraction = numerator / denominator
            expected_speed = min(
                4.0,
                max(0.25, exp(2.0 * (observed_fraction - 0.25))),
            )

            assert (
                schedule.resolve_reduction_speed(
                    observation=cutoff_observation(
                        full_bank_transition_count=denominator,
                        local_transition_count=numerator,
                    )
                )
                == expected_speed
            )


@pytest.mark.parametrize(
    "target_fraction",
    [0.0, 1.0, float("nan"), float("inf")],
)
def test_local_route_schedule_rejects_invalid_target(
    target_fraction: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="target_local_route_fraction must be finite and in",
    ):
        _ = CSALocalRouteCutoffSchedule(
            target_local_route_fraction=target_fraction,
        )


@pytest.mark.parametrize("response", [0.0, -1.0, float("nan"), float("inf")])
def test_local_route_schedule_rejects_invalid_response(response: float) -> None:
    with pytest.raises(ValueError, match="response must be finite and positive"):
        _ = CSALocalRouteCutoffSchedule(response=response)


def test_local_route_schedule_rejects_boolean_parameters() -> None:
    with pytest.raises(TypeError, match="target_local_route_fraction must be numeric"):
        _ = CSALocalRouteCutoffSchedule(target_local_route_fraction=True)
    with pytest.raises(TypeError, match="response must be numeric"):
        _ = CSALocalRouteCutoffSchedule(response=True)


def test_local_route_schedule_requires_exponential_reduction() -> None:
    with pytest.raises(
        ValueError,
        match="local-route cutoff control requires exponential reduction",
    ):
        _ = CSALocalRouteCutoffSchedule(reduction_method="linear")


def test_advance_cutoff_state_dispatches_local_route_speed() -> None:
    state = CSAProgressionState().initialize_cutoff(
        distance_cutoff=8.0,
        minimum_distance_cutoff=0.0,
        previous_score_gap=2.0,
    )
    response_for_double_speed = log(2.0) / 0.75
    schedule = CSALocalRouteCutoffSchedule(
        reduction_factor=0.5,
        stagnation_update_limit=0,
        target_local_route_fraction=0.25,
        response=response_for_double_speed,
    )
    observation = cutoff_observation(
        full_bank_transition_count=1,
        local_transition_count=1,
    )

    next_state, cycle_increment = advance_cutoff_state(
        state=state,
        schedule=schedule,
        score_gap=observation.score_gap,
        unused_entry_count=observation.unused_entry_count,
        reduction_speed=schedule.resolve_reduction_speed(
            observation=observation,
        ),
    )

    assert next_state.distance_cutoff == 2.0
    assert not cycle_increment


def test_adaptive_reduction_cannot_override_explicit_recovery() -> None:
    state = CSAProgressionState().initialize_cutoff(
        distance_cutoff=4.0,
        minimum_distance_cutoff=1.0,
        previous_score_gap=1.0,
    )
    schedule = CSALocalRouteCutoffSchedule(
        reduction_factor=0.5,
        recover_steps=2,
        recover_mode="score_gap_increase",
    )
    observation = cutoff_observation(
        score_gap=2.0,
        full_bank_transition_count=1,
        local_transition_count=1,
    )

    next_state, _ = advance_cutoff_state(
        state=state,
        schedule=schedule,
        score_gap=observation.score_gap,
        unused_entry_count=observation.unused_entry_count,
        reduction_speed=schedule.resolve_reduction_speed(
            observation=observation,
        ),
    )

    assert next_state.distance_cutoff == 16.0
    assert next_state.cutoff_recover_limit == 4.0


def test_score_gap_query_distinguishes_missing_and_finite_evidence() -> None:
    assert infer_score_gap(()) is None
    assert infer_score_gap((BankEntry(candidate=0, value=3.0),)) == 0.0
    assert (
        infer_score_gap(
            (
                BankEntry(candidate=0, value=-1.0),
                BankEntry(candidate=1, value=3.0),
            )
        )
        == 4.0
    )


def test_overflowing_score_gap_is_unavailable_and_cannot_trigger_recovery() -> None:
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

    overflow_gap = infer_score_gap(entries)
    assert overflow_gap is None
    assert optimizer.infer_score_gap_for_entries(entries) is None

    schedule = CSACutoffSchedule(
        reduction_factor=0.5,
        recover_steps=2,
        recover_mode="score_gap_decrease",
    )
    state = CSAProgressionState().initialize_cutoff(
        distance_cutoff=4.0,
        minimum_distance_cutoff=1.0,
        previous_score_gap=10.0,
    )
    state, _ = advance_cutoff_state(
        state=state,
        schedule=schedule,
        score_gap=overflow_gap,
        unused_entry_count=1,
    )
    state, _ = advance_cutoff_state(
        state=state,
        schedule=schedule,
        score_gap=1.0,
        unused_entry_count=1,
    )

    assert state.distance_cutoff == 1.0


def prime_bank_for_cutoff_test(
    *,
    cutoff_schedule: CSACutoffSchedule,
    proposal_candidate: int,
    score_scale: float = 1.0,
    score_offset: float = 0.0,
) -> tuple[CSAOptimizerDriver, Proposal[int]]:
    """Return a primed optimizer driver and one pending proposal."""
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
            BankEntry(
                candidate=0,
                value=score_offset,
                proposal_id="b0",
            ),
            BankEntry(
                candidate=4,
                value=score_scale * 16.0 + score_offset,
                proposal_id="b1",
            ),
            BankEntry(
                candidate=10,
                value=score_scale * 100.0 + score_offset,
                proposal_id="b2",
            ),
        ),
    )
    optimizer.cutoff_state = CSACutoffState(
        distance_cutoff=5.0,
        minimum_distance_cutoff=0.0,
        cutoff_recover_limit=5.0,
    )
    proposal = Proposal(candidate=proposal_candidate, proposal_id="p0")
    optimizer.set_pending_proposals((proposal,))
    return optimizer, proposal


def tell_inferior_proposal(
    optimizer: CSAOptimizerDriver,
    proposal: Proposal[int],
    *,
    score_scale: float = 1.0,
    score_offset: float = 0.0,
) -> None:
    """Tell one inferior proposal through a test driver."""
    optimizer.tell(
        (
            Observation(
                proposal=proposal,
                candidate=proposal.candidate,
                value=score_scale * 1000.0 + score_offset,
                score=score_scale * 1000.0 + score_offset,
            ),
        )
    )


def test_fixed_schedule_preserves_end_to_end_cutoff_step() -> None:
    optimizer, proposal = prime_bank_for_cutoff_test(
        cutoff_schedule=CSACutoffSchedule(
            initial_distance_cutoff=5.0,
            minimum_distance_cutoff=0.0,
            reduction_factor=0.5,
        ),
        proposal_candidate=0,
    )

    tell_inferior_proposal(optimizer, proposal)

    assert optimizer.state.distance_cutoff == 2.5


def test_custom_resolver_automatically_enters_adaptive_path() -> None:
    schedule = DoubleSpeedCutoffSchedule(
        initial_distance_cutoff=5.0,
        minimum_distance_cutoff=0.0,
        reduction_factor=0.5,
    )
    assert schedule.requires_reduction_observation

    optimizer, proposal = prime_bank_for_cutoff_test(
        cutoff_schedule=schedule,
        proposal_candidate=0,
    )

    tell_inferior_proposal(optimizer, proposal)

    assert optimizer.state.distance_cutoff == 1.25


def test_optimizer_supplies_same_batch_local_route_fraction() -> None:
    response_for_double_speed = log(2.0) / 0.75
    optimizer, proposal = prime_bank_for_cutoff_test(
        cutoff_schedule=CSALocalRouteCutoffSchedule(
            initial_distance_cutoff=5.0,
            minimum_distance_cutoff=0.0,
            reduction_factor=0.5,
            target_local_route_fraction=0.25,
            response=response_for_double_speed,
        ),
        proposal_candidate=0,
    )

    tell_inferior_proposal(optimizer, proposal)

    assert optimizer.state.distance_cutoff == 1.25


def test_optimizer_counts_far_decisions_in_local_route_denominator() -> None:
    response_for_half_speed = log(2.0) / 0.25
    optimizer, proposal = prime_bank_for_cutoff_test(
        cutoff_schedule=CSALocalRouteCutoffSchedule(
            initial_distance_cutoff=5.0,
            minimum_distance_cutoff=0.0,
            reduction_factor=0.5,
            target_local_route_fraction=0.25,
            response=response_for_half_speed,
        ),
        proposal_candidate=30,
    )

    tell_inferior_proposal(optimizer, proposal)

    distance_cutoff = optimizer.state.distance_cutoff
    assert distance_cutoff is not None
    assert isclose(distance_cutoff, 5.0 * (0.5**0.5))


@pytest.mark.parametrize("proposal_candidates", [(0, 30), (30, 0)])
def test_optimizer_aggregates_mixed_routes_independently_of_batch_order(
    proposal_candidates: tuple[int, int],
) -> None:
    response_for_double_speed = log(2.0) / 0.25
    optimizer, _ = prime_bank_for_cutoff_test(
        cutoff_schedule=CSALocalRouteCutoffSchedule(
            initial_distance_cutoff=5.0,
            minimum_distance_cutoff=0.0,
            reduction_factor=0.5,
            target_local_route_fraction=0.25,
            response=response_for_double_speed,
        ),
        proposal_candidate=0,
    )
    proposals = tuple(
        Proposal(candidate=candidate, proposal_id=f"p{index}")
        for index, candidate in enumerate(proposal_candidates)
    )
    optimizer.set_pending_proposals(proposals)

    optimizer.tell(
        tuple(
            Observation(
                proposal=proposal,
                candidate=proposal.candidate,
                value=1000.0,
                score=1000.0,
            )
            for proposal in proposals
        )
    )

    assert optimizer.state.distance_cutoff == 1.25


def test_local_route_cutoff_remains_monotone_and_bounded_across_iterations() -> None:
    schedule = CSALocalRouteCutoffSchedule(
        reduction_factor=0.5,
        stagnation_update_limit=0,
    )
    state = CSAProgressionState().initialize_cutoff(
        distance_cutoff=8.0,
        minimum_distance_cutoff=1.0,
        previous_score_gap=2.0,
    )

    for local_transition_count in (0, 4, 1, 3, 0, 4):
        previous_distance_cutoff = state.distance_cutoff
        assert previous_distance_cutoff is not None
        observation = cutoff_observation(
            full_bank_transition_count=4,
            local_transition_count=local_transition_count,
        )
        state, cycle_increment = advance_cutoff_state(
            state=state,
            schedule=schedule,
            score_gap=observation.score_gap,
            unused_entry_count=1,
            reduction_speed=schedule.resolve_reduction_speed(
                observation=observation,
            ),
        )

        assert state.distance_cutoff is not None
        assert 1.0 <= state.distance_cutoff <= previous_distance_cutoff
        assert not cycle_increment

    assert state.distance_cutoff == 1.0


def test_local_route_cutoff_trajectory_is_positive_affine_invariant() -> None:
    schedule = CSALocalRouteCutoffSchedule(
        initial_distance_cutoff=5.0,
        minimum_distance_cutoff=0.0,
        reduction_factor=0.5,
    )
    baseline_optimizer, baseline_proposal = prime_bank_for_cutoff_test(
        cutoff_schedule=schedule,
        proposal_candidate=0,
    )
    transformed_optimizer, transformed_proposal = prime_bank_for_cutoff_test(
        cutoff_schedule=schedule,
        proposal_candidate=0,
        score_scale=100.0,
        score_offset=7.0,
    )

    tell_inferior_proposal(baseline_optimizer, baseline_proposal)
    tell_inferior_proposal(
        transformed_optimizer,
        transformed_proposal,
        score_scale=100.0,
        score_offset=7.0,
    )

    assert transformed_optimizer.state.distance_cutoff == (
        baseline_optimizer.state.distance_cutoff
    )
    assert transformed_optimizer.state.iteration_count == (
        baseline_optimizer.state.iteration_count
    )
    assert transformed_optimizer.state.cycle_count == (
        baseline_optimizer.state.cycle_count
    )
    assert tuple(
        entry.candidate for entry in transformed_optimizer.bank.entries
    ) == tuple(entry.candidate for entry in baseline_optimizer.bank.entries)
