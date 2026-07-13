"""Tests for CSA cutoff-control observations and schedule dispatch."""

from dataclasses import dataclass

import pytest
from typing_extensions import override

from tests.csa_support import AbsoluteDistance, RepeatParent, make_optimizer
from variopt import IntegerSpace, Observation, Proposal
from variopt.algorithms.population.csa.banking.bank import Bank, BankEntry
from variopt.algorithms.population.csa.progression.cutoff.observation import (
    CSACutoffObservation,
)
from variopt.algorithms.population.csa.progression.cutoff.policy import (
    CSACutoffSchedule,
)
from variopt.algorithms.population.csa.progression.cutoff.state import (
    CSACutoffState,
)


@dataclass(frozen=True, slots=True)
class PairwiseMedianCutoffSchedule(CSACutoffSchedule):
    """Test schedule that selects the current median pair distance."""

    @property
    @override
    def requires_pairwise_distances(self) -> bool:
        return True

    @override
    def resolve_next_distance_cutoff(
        self,
        *,
        state: CSACutoffState,
        observation: CSACutoffObservation,
    ) -> float:
        _ = state
        pairwise_distances = observation.pairwise_distances
        if pairwise_distances is None:
            msg = "pairwise distances were not materialized"
            raise ValueError(msg)
        ordered_distances = sorted(pairwise_distances)
        return ordered_distances[len(ordered_distances) // 2]


def test_cutoff_observation_reports_used_entry_fraction() -> None:
    observation = CSACutoffObservation(
        score_gap=2.0,
        eligible_entry_count=8,
        unused_entry_count=3,
        pairwise_distances=(2, 1.5, 0.0),
    )

    assert observation.score_gap == 2.0
    assert observation.pairwise_distances == (2.0, 1.5, 0.0)
    assert observation.used_entry_fraction == 0.625


def test_cutoff_observation_has_no_fraction_without_eligible_entries() -> None:
    observation = CSACutoffObservation(
        score_gap=None,
        eligible_entry_count=0,
        unused_entry_count=0,
    )

    assert observation.used_entry_fraction is None


@pytest.mark.parametrize(
    ("eligible_entry_count", "unused_entry_count", "message"),
    [
        (-1, 0, "eligible_entry_count must be non-negative"),
        (2, -1, "unused_entry_count must not exceed eligible_entry_count"),
        (2, 3, "unused_entry_count must not exceed eligible_entry_count"),
    ],
)
def test_cutoff_observation_rejects_invalid_entry_counts(
    eligible_entry_count: int,
    unused_entry_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = CSACutoffObservation(
            score_gap=None,
            eligible_entry_count=eligible_entry_count,
            unused_entry_count=unused_entry_count,
        )


@pytest.mark.parametrize("distance", [-1.0, float("nan"), float("inf")])
def test_cutoff_observation_rejects_invalid_pairwise_distance(
    distance: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="pairwise_distances must contain finite non-negative floats",
    ):
        _ = CSACutoffObservation(
            score_gap=None,
            eligible_entry_count=1,
            unused_entry_count=1,
            pairwise_distances=(distance,),
        )


def test_cutoff_schedule_requests_pairwise_distances_lazily() -> None:
    fixed_schedule = CSACutoffSchedule()
    pairwise_schedule = PairwiseMedianCutoffSchedule(
        initial_distance_cutoff=10.0,
        minimum_distance_cutoff=0.0,
    )

    assert not fixed_schedule.requires_pairwise_distances
    assert pairwise_schedule.requires_pairwise_distances


def test_optimizer_supplies_current_bank_pairwise_distances() -> None:
    cutoff_schedule = PairwiseMedianCutoffSchedule(
        initial_distance_cutoff=10.0,
        minimum_distance_cutoff=0.0,
        reduction_factor=1.0,
    )
    optimizer = make_optimizer(
        space=IntegerSpace(low=0, high=20),
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
        distance_cutoff=10.0,
        minimum_distance_cutoff=0.0,
    )
    proposal = Proposal(candidate=0, proposal_id="p0")
    optimizer.set_pending_proposals((proposal,))

    optimizer.tell(
        (
            Observation(
                proposal=proposal,
                candidate=0,
                value=0.0,
                score=0.0,
            ),
        )
    )

    assert optimizer.state.distance_cutoff == 6.0
