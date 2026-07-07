"""Tests for adaptive CSA refresh behavior."""

from dataclasses import replace

from typing_extensions import override

from variopt.algorithms.population.csa import (
    CSACutoffSchedule,
    CSARefreshPolicy,
)
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
from variopt.algorithms.population.csa.engine.boundary import (
    begin_refresh,
    begin_stage_transition,
    complete_refresh,
)
from variopt.algorithms.population.csa.generation.proposal import (
    CSAProposalPolicy,
    CSAProposalState,
)
from variopt.algorithms.population.csa.generation.state import (
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
)
from variopt.algorithms.population.csa.scoring.acceptance import (
    CSAAcceptancePolicy,
)
from variopt.algorithms.population.csa.scoring.acceptance_state import (
    CSAAcceptanceState,
)
from variopt.algorithms.population.csa.scoring.model import CSAScoreModel
from variopt.algorithms.population.csa.scoring.model_state import (
    CSAScoreModelState,
)
from variopt.algorithms.population.csa.selection.state import (
    SeedSelectionState,
)
from variopt.diversity import DiversityMetric
from variopt.randomness import RandomStateSnapshot


class CSARefreshPolicyTests:
    """Regression tests for adaptive refresh payload construction and masking."""

    def test_begin_refresh_legacy_mode_preserves_no_entries(self) -> None:
        engine_state = build_full_engine_state(
            capacity=4,
            max_capacity=4,
            entries=(
                BankEntry(candidate=10, value=4.0),
                BankEntry(candidate=11, value=1.0),
                BankEntry(candidate=12, value=3.0),
                BankEntry(candidate=13, value=2.0),
            ),
        )

        next_state = begin_refresh(engine_state, refresh_policy=CSARefreshPolicy())

        assert next_state.banking_state.refresh_state is not None
        assert next_state.banking_state.refresh_state.preserved_bank_entries == ()
        assert next_state.banking_state.refresh_state.preserved_reference_entries == ()

    def test_adaptive_refresh_preserves_elite_entries_and_masks_them_for_one_cycle(
        self,
    ) -> None:
        refresh_policy = CSARefreshPolicy(
            mode="adaptive_refresh",
            preserve_fraction=0.25,
        )
        engine_state = build_full_engine_state(
            capacity=4,
            max_capacity=4,
            entries=(
                BankEntry(candidate=10, value=4.0),
                BankEntry(candidate=11, value=1.0),
                BankEntry(candidate=12, value=3.0),
                BankEntry(candidate=13, value=2.0),
            ),
        )

        refreshing_state = begin_refresh(engine_state, refresh_policy=refresh_policy)
        refresh_state = refreshing_state.banking_state.refresh_state
        assert refresh_state is not None
        assert refresh_state.preserved_bank_entries == (
            BankEntry(candidate=11, value=1.0),
        )
        assert refresh_state.preserved_reference_entries == (
            BankEntry(candidate=11, value=1.0),
        )

        refreshed_state = complete_refresh(
            replace(
                refreshing_state,
                banking_state=replace(
                    refreshing_state.banking_state,
                    refresh_state=ReferenceRefreshState(
                        target_capacity=4,
                        preserved_bank_entries=refresh_state.preserved_bank_entries,
                        preserved_reference_entries=refresh_state.preserved_reference_entries,
                        pool_entries=(
                            BankEntry(candidate=20, value=5.0),
                            BankEntry(candidate=21, value=0.5),
                            BankEntry(candidate=22, value=6.0),
                        ),
                    ),
                ),
            ),
            replace(
                refresh_state,
                pool_entries=(
                    BankEntry(candidate=20, value=5.0),
                    BankEntry(candidate=21, value=0.5),
                    BankEntry(candidate=22, value=6.0),
                ),
            ),
            refresh_policy=refresh_policy,
            diversity_metric=AbsoluteDistance(),
            cutoff_schedule=CSACutoffSchedule(),
            infer_average_distance=lambda entries: 1.0,
            infer_score_gap=lambda entries: None,
        )

        assert refreshed_state.banking_state.bank.entries == (
            BankEntry(candidate=11, value=1.0),
            BankEntry(candidate=21, value=0.5),
            BankEntry(candidate=20, value=5.0),
            BankEntry(candidate=22, value=6.0),
        )
        assert refreshed_state.progression_state.refresh_mask == frozenset({0})
        assert refreshed_state.progression_state.seed_mask == frozenset({0})
        assert refreshed_state.progression_state.partner_mask == frozenset({0})

    def test_adaptive_stage_growth_refresh_retargets_newcomer_masks_to_preserved_entries(
        self,
    ) -> None:
        refresh_policy = CSARefreshPolicy(
            mode="adaptive_refresh",
            preserve_fraction=0.5,
        )
        stage_state = CSAStageState(base_capacity=2, max_capacity=4)
        transition = stage_state.next_transition()
        assert transition is not None
        engine_state = build_full_engine_state(
            capacity=2,
            max_capacity=4,
            entries=(
                BankEntry(candidate=30, value=4.0),
                BankEntry(candidate=31, value=1.0),
            ),
        )

        transitioned_state = begin_stage_transition(
            engine_state,
            transition,
            refresh_policy=refresh_policy,
        )
        refresh_state = transitioned_state.banking_state.refresh_state
        assert refresh_state is not None
        assert len(refresh_state.preserved_bank_entries) == 1
        assert refresh_state.preserved_bank_entries == (
            BankEntry(candidate=31, value=1.0),
        )

        refreshed_state = complete_refresh(
            replace(
                transitioned_state,
                banking_state=replace(
                    transitioned_state.banking_state,
                    refresh_state=ReferenceRefreshState(
                        target_capacity=4,
                        preserved_bank_entries=refresh_state.preserved_bank_entries,
                        preserved_reference_entries=refresh_state.preserved_reference_entries,
                        pool_entries=(
                            BankEntry(candidate=40, value=2.0),
                            BankEntry(candidate=41, value=3.0),
                            BankEntry(candidate=42, value=4.0),
                        ),
                    ),
                ),
            ),
            replace(
                refresh_state,
                pool_entries=(
                    BankEntry(candidate=40, value=2.0),
                    BankEntry(candidate=41, value=3.0),
                    BankEntry(candidate=42, value=4.0),
                ),
            ),
            refresh_policy=refresh_policy,
            diversity_metric=AbsoluteDistance(),
            cutoff_schedule=CSACutoffSchedule(),
            infer_average_distance=lambda entries: 1.0,
            infer_score_gap=lambda entries: None,
        )

        assert refreshed_state.progression_state.refresh_mask == frozenset()
        assert refreshed_state.progression_state.stage_state.seed_mask == frozenset({0})
        assert refreshed_state.progression_state.stage_state.partner_mask == frozenset(
            {0}
        )


def build_full_engine_state(
    *,
    capacity: int,
    max_capacity: int,
    entries: tuple[BankEntry[int], ...],
) -> CSAEngineState[int]:
    engine_state = build_engine_state()
    return replace(
        engine_state,
        banking_state=replace(
            engine_state.banking_state,
            bank=Bank[int](capacity=capacity, entries=entries),
            reference_bank=ReferenceBank[int](capacity=capacity, entries=entries),
        ),
        progression_state=replace(
            engine_state.progression_state,
            stage_state=CSAStageState(
                base_capacity=capacity,
                max_capacity=max_capacity,
            ),
        ),
    )


def build_engine_state() -> CSAEngineState[int]:
    return CSAEngineState(
        random_state=RandomStateSnapshot.from_seed(0),
        banking_state=CSABankingState(
            bank=Bank[int](capacity=4),
            reference_bank=ReferenceBank[int](capacity=4),
            refresh_state=None,
            growth_state=CSABankGrowthState[int].from_policy(
                CSABankGrowthPolicy(),
            ),
            clustering_state=CSAClusteringState[int](
                policy=CSAClusteringPolicy(),
            ),
        ),
        progression_state=CSAProgressionState(
            cutoff_state=CSACutoffState(),
            stage_state=CSAStageState(base_capacity=4, max_capacity=4),
        ),
        selection_state=SeedSelectionState(),
        generation_state=GenerationRuntimeState[int](),
        proposal_state=CSAProposalState.from_policy(CSAProposalPolicy()),
        scoring_state=CSAScoringState(
            acceptance_state=CSAAcceptanceState.from_policy(
                CSAAcceptancePolicy(),
            ),
            model_state=CSAScoreModelState(score_model=CSAScoreModel()),
        ),
        pending_proposals=CSAPendingProposals[int](),
    )


class AbsoluteDistance(DiversityMetric[int]):
    """Simple absolute-difference metric for refresh regression tests."""

    @override
    def distance(self, left: int, right: int) -> float:
        return float(abs(left - right))
