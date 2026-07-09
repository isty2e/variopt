"""Observation-aligned CSA bank-transition contracts."""

from collections.abc import Sequence

import numpy as np
import pytest

from tests.csa_support import AbsoluteDistance
from variopt import Observation, Proposal
from variopt.algorithms.population.csa.banking.bank import Bank, BankEntry
from variopt.algorithms.population.csa.banking.clustering import (
    CSAClusteringPolicy,
    CSAClusteringState,
)
from variopt.algorithms.population.csa.banking.growth import (
    CSABankGrowthPolicy,
    CSABankGrowthState,
)
from variopt.algorithms.population.csa.banking.reference import ReferenceBank
from variopt.algorithms.population.csa.banking.update.logic import (
    apply_bank_update_batch,
)
from variopt.algorithms.population.csa.banking.update.policy import (
    CSABankUpdatePolicy,
)
from variopt.algorithms.population.csa.banking.update.result import (
    BankUpdateResult,
    CSABankTransition,
)
from variopt.algorithms.population.csa.progression.cutoff.policy import (
    CSACutoffSchedule,
)
from variopt.algorithms.population.csa.progression.cutoff.state import (
    CSACutoffState,
)
from variopt.algorithms.population.csa.progression.stage import CSAStageState
from variopt.algorithms.population.csa.progression.state import CSAProgressionState
from variopt.algorithms.population.csa.scoring.acceptance import CSAAcceptancePolicy
from variopt.algorithms.population.csa.scoring.acceptance_state import (
    CSAAcceptanceState,
)
from variopt.algorithms.population.csa.scoring.model import CSAScoreModel
from variopt.algorithms.population.csa.scoring.model_state import CSAScoreModelState
from variopt.algorithms.population.csa.trace.events.state import CSAEventTraceState


def make_observation(
    *,
    candidate: int,
    score: float,
    proposal_id: str | None,
) -> Observation[int]:
    """Return one deterministic scalar observation for transition tests."""
    return Observation(
        proposal=Proposal(candidate=candidate, proposal_id=proposal_id),
        candidate=candidate,
        value=score,
        score=score,
    )


def run_bank_update(
    *,
    bank: Bank[int],
    observations: Sequence[Observation[int]],
    distance_cutoff: float | None = 2.0,
    base_bank_capacity: int | None = None,
    update_policy: CSABankUpdatePolicy | None = None,
    growth_state: CSABankGrowthState[int] | None = None,
    clustering_state: CSAClusteringState[int] | None = None,
    trace_state: CSAEventTraceState[int] | None = None,
) -> BankUpdateResult[int]:
    """Apply a deterministic bank update with explicit runtime defaults."""
    resolved_base_capacity = (
        bank.capacity if base_bank_capacity is None else base_bank_capacity
    )
    if growth_state is None:
        growth_policy = CSABankGrowthPolicy()
        resolved_growth_state = CSABankGrowthState[int](
            policy=growth_policy,
            active_energy_gap_limit=growth_policy.initial_energy_gap_limit,
        )
    else:
        resolved_growth_state = growth_state

    maximum_capacity = resolved_growth_state.policy.maximum_capacity
    stage_maximum_capacity = (
        bank.capacity if maximum_capacity is None else maximum_capacity
    )
    cutoff_state = CSACutoffState()
    if distance_cutoff is not None:
        cutoff_state = CSACutoffState(
            distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=distance_cutoff,
            cutoff_recover_limit=distance_cutoff,
        )

    return apply_bank_update_batch(
        bank=bank,
        state=CSAProgressionState(
            cutoff_state=cutoff_state,
            stage_state=CSAStageState(
                base_capacity=resolved_base_capacity,
                max_capacity=stage_maximum_capacity,
            ),
        ),
        observations=observations,
        diversity_metric=AbsoluteDistance(),
        infer_average_distance=lambda entries: 2.0,
        infer_score_gap=lambda entries: (
            None
            if len(entries) < 2
            else max(entry.value for entry in entries)
            - min(entry.value for entry in entries)
        ),
        cutoff_schedule=CSACutoffSchedule(
            initial_distance_cutoff=2.0,
            minimum_distance_cutoff=2.0,
        ),
        update_policy=(
            CSABankUpdatePolicy() if update_policy is None else update_policy
        ),
        acceptance_state=CSAAcceptanceState.from_policy(CSAAcceptancePolicy()),
        score_model_state=CSAScoreModelState(score_model=CSAScoreModel()),
        growth_state=resolved_growth_state,
        clustering_state=(
            CSAClusteringState(policy=CSAClusteringPolicy(enabled=False))
            if clustering_state is None
            else clustering_state
        ),
        base_bank_capacity=resolved_base_capacity,
        masked_seed_indices=frozenset(),
        random_state=np.random.RandomState(0),
        trace_state=trace_state,
    )


def start_trace(bank: Bank[int]) -> CSAEventTraceState[int]:
    """Return a trace state with one active generation."""
    return CSAEventTraceState[int]().start_generation(
        stage_index=0,
        stage_round=0,
        cycle_count=0,
        bank=bank,
        reference_bank=ReferenceBank(
            capacity=bank.capacity,
            entries=bank.entries,
        ),
        bank_status_before=tuple(True for _ in bank.entries),
        seed_mask=frozenset(),
        partner_mask=frozenset(),
        seed_batch=(),
        proposal_families_before=(),
    )


class CSABankTransitionModelTests:
    """Validate canonical transition field relationships."""

    def test_rejects_incoherent_disposition_states(self) -> None:
        with pytest.raises(ValueError, match="must not declare a target_index"):
            CSABankTransition(
                proposal_id="p-1",
                route="local",
                disposition="rejected",
                target_index=0,
                survived_batch=False,
            )

        with pytest.raises(ValueError, match="cannot survive"):
            CSABankTransition(
                proposal_id="p-1",
                route="local",
                disposition="rejected",
                target_index=None,
                survived_batch=True,
            )

        with pytest.raises(TypeError, match="integer target_index"):
            CSABankTransition(
                proposal_id="p-1",
                route="initial",
                disposition="appended",
                target_index=None,
                survived_batch=False,
            )

        with pytest.raises(ValueError, match="must be non-negative"):
            CSABankTransition(
                proposal_id="p-1",
                route="far",
                disposition="replaced",
                target_index=-1,
                survived_batch=False,
            )

        with pytest.raises(ValueError, match="initial and growth routes must append"):
            CSABankTransition(
                proposal_id="p-1",
                route="growth",
                disposition="rejected",
                target_index=None,
                survived_batch=False,
            )

        with pytest.raises(ValueError, match="cannot append"):
            CSABankTransition(
                proposal_id="p-1",
                route="cluster",
                disposition="appended",
                target_index=0,
                survived_batch=False,
            )


class CSABankTransitionRouteTests:
    """Validate route and disposition facts emitted by bank reduction."""

    def test_initial_admissions_are_aligned_in_observation_order(self) -> None:
        result = run_bank_update(
            bank=Bank(
                capacity=3,
                entries=(BankEntry(candidate=0, value=0.0, proposal_id="b-0"),),
            ),
            observations=(
                make_observation(candidate=1, score=1.0, proposal_id="p-1"),
                make_observation(candidate=2, score=2.0, proposal_id="p-2"),
            ),
            distance_cutoff=None,
        )

        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="initial",
                disposition="appended",
                target_index=1,
                survived_batch=True,
            ),
            CSABankTransition(
                proposal_id="p-2",
                route="initial",
                disposition="appended",
                target_index=2,
                survived_batch=True,
            ),
        )

    def test_initial_admission_keeps_clustering_aligned_for_next_observation(
        self,
    ) -> None:
        result = run_bank_update(
            bank=Bank(
                capacity=2,
                entries=(BankEntry(candidate=0, value=10.0, proposal_id="b-0"),),
            ),
            observations=(
                make_observation(candidate=10, score=0.0, proposal_id="p-1"),
                make_observation(candidate=3, score=5.0, proposal_id="p-2"),
            ),
            clustering_state=CSAClusteringState(
                policy=CSAClusteringPolicy(enabled=True),
                cluster_distance=10.0,
                cluster_labels=(1,),
            ),
        )

        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="initial",
                disposition="appended",
                target_index=1,
                survived_batch=True,
            ),
            CSABankTransition(
                proposal_id="p-2",
                route="cluster",
                disposition="replaced",
                target_index=0,
                survived_batch=True,
            ),
        )

    def test_local_route_distinguishes_replacement_from_rejection(self) -> None:
        result = run_bank_update(
            bank=Bank(
                capacity=2,
                entries=(
                    BankEntry(candidate=0, value=10.0, proposal_id="b-0"),
                    BankEntry(candidate=10, value=0.0, proposal_id="b-1"),
                ),
            ),
            observations=(
                make_observation(candidate=1, score=5.0, proposal_id="p-1"),
                make_observation(candidate=2, score=20.0, proposal_id="p-2"),
            ),
        )

        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="local",
                disposition="replaced",
                target_index=0,
                survived_batch=True,
            ),
            CSABankTransition(
                proposal_id="p-2",
                route="local",
                disposition="rejected",
                target_index=None,
                survived_batch=False,
            ),
        )

    def test_later_replacement_uses_proposal_identity_for_survival(self) -> None:
        result = run_bank_update(
            bank=Bank(
                capacity=1,
                entries=(BankEntry(candidate=0, value=10.0, proposal_id="b-0"),),
            ),
            observations=(
                make_observation(candidate=0, score=5.0, proposal_id="p-1"),
                make_observation(candidate=0, score=1.0, proposal_id="p-2"),
            ),
            distance_cutoff=1.0,
        )

        assert result.transitions[0] == CSABankTransition(
            proposal_id="p-1",
            route="local",
            disposition="replaced",
            target_index=0,
            survived_batch=False,
        )
        assert result.transitions[1] == CSABankTransition(
            proposal_id="p-2",
            route="local",
            disposition="replaced",
            target_index=0,
            survived_batch=True,
        )

    def test_growth_survival_is_reconciled_after_energy_cut(self) -> None:
        growth_policy = CSABankGrowthPolicy(
            enabled=True,
            maximum_capacity=4,
            initial_energy_gap_limit=10.0,
        )
        result = run_bank_update(
            bank=Bank(
                capacity=2,
                entries=(
                    BankEntry(candidate=0, value=0.0, proposal_id="b-0"),
                    BankEntry(candidate=10, value=10.0, proposal_id="b-1"),
                ),
            ),
            observations=(
                make_observation(candidate=20, score=9.0, proposal_id="p-1"),
                make_observation(candidate=30, score=-100.0, proposal_id="p-2"),
            ),
            base_bank_capacity=2,
            growth_state=CSABankGrowthState[int](
                policy=growth_policy,
                active_energy_gap_limit=growth_policy.initial_energy_gap_limit,
            ),
        )

        assert result.removed_indices == frozenset({1, 2})
        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="growth",
                disposition="appended",
                target_index=2,
                survived_batch=False,
            ),
            CSABankTransition(
                proposal_id="p-2",
                route="growth",
                disposition="appended",
                target_index=3,
                survived_batch=True,
            ),
        )

    def test_cluster_route_reports_replacement(self) -> None:
        result = run_bank_update(
            bank=cluster_test_bank(),
            observations=(
                make_observation(candidate=3, score=70.0, proposal_id="p-1"),
            ),
            clustering_state=cluster_test_state(),
        )

        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="cluster",
                disposition="replaced",
                target_index=1,
                survived_batch=True,
            ),
        )

    def test_cluster_route_reports_rejection(self) -> None:
        result = run_bank_update(
            bank=cluster_test_bank(),
            observations=(
                make_observation(candidate=3, score=90.0, proposal_id="p-1"),
            ),
            clustering_state=cluster_test_state(),
        )

        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="cluster",
                disposition="rejected",
                target_index=None,
                survived_batch=False,
            ),
        )

    def test_far_route_distinguishes_replacement_from_rejection(self) -> None:
        result = run_bank_update(
            bank=Bank(
                capacity=2,
                entries=(
                    BankEntry(candidate=0, value=10.0, proposal_id="b-0"),
                    BankEntry(candidate=10, value=0.0, proposal_id="b-1"),
                ),
            ),
            observations=(
                make_observation(candidate=100, score=5.0, proposal_id="p-1"),
                make_observation(candidate=200, score=20.0, proposal_id="p-2"),
            ),
        )

        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="far",
                disposition="replaced",
                target_index=0,
                survived_batch=True,
            ),
            CSABankTransition(
                proposal_id="p-2",
                route="far",
                disposition="rejected",
                target_index=None,
                survived_batch=False,
            ),
        )

    def test_disabled_local_route_is_an_explicit_rejection(self) -> None:
        result = run_bank_update(
            bank=Bank(
                capacity=2,
                entries=(
                    BankEntry(candidate=0, value=10.0, proposal_id="b-0"),
                    BankEntry(candidate=10, value=0.0, proposal_id="b-1"),
                ),
            ),
            observations=(make_observation(candidate=1, score=1.0, proposal_id="p-1"),),
            update_policy=CSABankUpdatePolicy(local_update_mode="disabled"),
        )

        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="local",
                disposition="rejected",
                target_index=None,
                survived_batch=False,
            ),
        )

    def test_growth_admission_supersedes_an_earlier_disabled_local_route(
        self,
    ) -> None:
        growth_policy = CSABankGrowthPolicy(
            enabled=True,
            maximum_capacity=3,
            initial_energy_gap_limit=10.0,
            require_distance_cutoff=False,
        )
        result = run_bank_update(
            bank=Bank(
                capacity=2,
                entries=(
                    BankEntry(candidate=0, value=0.0, proposal_id="b-0"),
                    BankEntry(candidate=10, value=10.0, proposal_id="b-1"),
                ),
            ),
            observations=(make_observation(candidate=1, score=5.0, proposal_id="p-1"),),
            update_policy=CSABankUpdatePolicy(local_update_mode="disabled"),
            growth_state=CSABankGrowthState[int](
                policy=growth_policy,
                active_energy_gap_limit=growth_policy.initial_energy_gap_limit,
            ),
        )

        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="growth",
                disposition="appended",
                target_index=2,
                survived_batch=True,
            ),
        )

    def test_far_admission_supersedes_an_earlier_cluster_rejection(self) -> None:
        growth_policy = CSABankGrowthPolicy(
            enabled=True,
            maximum_capacity=4,
            initial_energy_gap_limit=300.0,
        )
        result = run_bank_update(
            bank=cluster_test_bank(),
            observations=(
                make_observation(candidate=103, score=70.0, proposal_id="p-1"),
            ),
            base_bank_capacity=2,
            growth_state=CSABankGrowthState[int](
                policy=growth_policy,
                active_energy_gap_limit=growth_policy.initial_energy_gap_limit,
            ),
            clustering_state=cluster_test_state(),
        )

        assert result.transitions == (
            CSABankTransition(
                proposal_id="p-1",
                route="far",
                disposition="replaced",
                target_index=1,
                survived_batch=True,
            ),
        )


class CSABankTransitionBoundaryTests:
    """Validate identity and diagnostics boundaries around transition emission."""

    def test_transition_identity_requires_new_distinct_proposal_ids(self) -> None:
        bank = Bank(
            capacity=1,
            entries=(BankEntry(candidate=0, value=0.0, proposal_id="b-0"),),
        )

        with pytest.raises(ValueError, match="must reference proposal ids"):
            run_bank_update(
                bank=bank,
                observations=(
                    make_observation(candidate=1, score=1.0, proposal_id=None),
                ),
            )

        with pytest.raises(ValueError, match="must have distinct proposal ids"):
            run_bank_update(
                bank=bank,
                observations=(
                    make_observation(candidate=1, score=1.0, proposal_id="p-1"),
                    make_observation(candidate=2, score=2.0, proposal_id="p-1"),
                ),
            )

        with pytest.raises(ValueError, match="must not already exist"):
            run_bank_update(
                bank=bank,
                observations=(
                    make_observation(candidate=1, score=1.0, proposal_id="b-0"),
                ),
            )

    def test_empty_observation_batch_has_no_transitions(self) -> None:
        result = run_bank_update(
            bank=Bank(
                capacity=1,
                entries=(BankEntry(candidate=0, value=0.0, proposal_id="b-0"),),
            ),
            observations=(),
        )

        assert result.transitions == ()

    def test_trace_recording_does_not_change_transition_facts(self) -> None:
        bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=0, value=10.0, proposal_id="b-0"),
                BankEntry(candidate=10, value=0.0, proposal_id="b-1"),
            ),
        )
        observations = (
            make_observation(candidate=100, score=5.0, proposal_id="p-1"),
            make_observation(candidate=200, score=20.0, proposal_id="p-2"),
        )

        untraced_result = run_bank_update(bank=bank, observations=observations)
        traced_result = run_bank_update(
            bank=bank,
            observations=observations,
            trace_state=start_trace(bank),
        )

        assert traced_result.trace_state is not None
        assert traced_result.transitions == untraced_result.transitions


def cluster_test_bank() -> Bank[int]:
    """Return a bank with two deterministic two-member clusters."""
    return Bank(
        capacity=4,
        entries=(
            BankEntry(candidate=0, value=10.0, proposal_id="b-0"),
            BankEntry(candidate=1, value=80.0, proposal_id="b-1"),
            BankEntry(candidate=100, value=55.0, proposal_id="b-2"),
            BankEntry(candidate=101, value=60.0, proposal_id="b-3"),
        ),
    )


def cluster_test_state() -> CSAClusteringState[int]:
    """Return clustering metadata aligned with :func:`cluster_test_bank`."""
    return CSAClusteringState(
        policy=CSAClusteringPolicy(enabled=True),
        cluster_distance=10.0,
        cluster_labels=(1, 1, 2, 2),
    )
