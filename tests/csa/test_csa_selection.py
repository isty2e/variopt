"""Tests for CSA seed selection and initial-routing semantics."""

import numpy as np
from typing_extensions import final, override

from tests.csa_support import (
    AbsoluteDistance,
    Bank,
    BankEntry,
    CSABankUpdatePolicy,
    CSACutoffState,
    CSAOptimizerTestCase,
    DiversityMetric,
    EncodeBinaryParents,
    IntegerSpace,
    Observation,
    Problem,
    Proposal,
    RecordingTernaryParents,
    ReferenceBank,
    RepeatParent,
    ScriptedIntegerSpace,
    SeedSelectionState,
    SequentialEvaluator,
    SquareObjective,
    evaluate_observations,
    make_optimizer,
    perturbation_schedule,
    prepare_seed_batch,
    schedule,
    select_partner_indices,
    should_use_reference_primary,
)
from variopt.algorithms.population.csa.selection.policy import (
    pick_diverse_low_value_seed,
)
from variopt.randomness import (
    random_state_choice_indices_without_replacement,
    random_state_randints,
)


@final
class CountingDistance(DiversityMetric[int]):
    """Distance metric that counts concrete distance evaluations."""

    def __init__(self) -> None:
        self.call_count: int = 0

    @override
    def distance(self, left: int, right: int) -> float:
        self.call_count += 1
        return float(abs(left - right))


def select_partner_indices_materialized_reference(
    *,
    entries: tuple[BankEntry[int], ...],
    seed_index: int,
    partner_count: int,
    random_state: np.random.RandomState,
) -> tuple[int, ...]:
    """Select partners through the pre-fast-path materialized-index contract."""
    available_indices = tuple(
        index for index in range(len(entries)) if index != seed_index
    )
    selected_positions = random_state_choice_indices_without_replacement(
        random_state,
        len(available_indices),
        partner_count,
    )
    return tuple(available_indices[position] for position in selected_positions)


class CSASelectionTests(CSAOptimizerTestCase):
    """White-box tests for CSA seed selection and routing state."""

    def test_seed_selection_uses_distinct_unused_seeds_before_reset(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((8, 3)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            seed_count=1,
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=2),
            )
        )

        first_batch = optimizer.ask(batch_size=1)
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                first_batch,
            )
        )

        second_batch = optimizer.ask(batch_size=1)
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                second_batch,
            )
        )

        first_candidate = first_batch[0].candidate
        second_candidate = second_batch[0].candidate
        assert first_candidate != second_candidate
        assert {first_candidate, second_candidate} == {3, 8}

    def test_seed_batch_contains_distinct_seeds(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((11, 7, 3)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=3,
            seed_count=2,
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
        proposal_batch = optimizer.ask(batch_size=2)

        assert len({proposal.candidate for proposal in proposal_batch}) == 2

    def test_generation_seed_selection_reuses_pair_distances(self) -> None:
        distance = CountingDistance()
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=distance,
            variation_operator=RepeatParent(),
            bank_capacity=4,
            seed_count=3,
            random_seed_mode=0,
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=4,
            entries=(
                BankEntry(candidate=0, value=0.0),
                BankEntry(candidate=10, value=10.0),
                BankEntry(candidate=20, value=20.0),
                BankEntry(candidate=30, value=30.0),
            ),
        )

        proposals = optimizer.ask(batch_size=1)

        assert len(proposals) == 1
        assert distance.call_count <= 5

    def test_diverse_seed_selection_excludes_ineligible_first_candidate(
        self,
    ) -> None:
        selected_index = pick_diverse_low_value_seed(
            entries=(
                BankEntry(candidate=0, value=100.0),
                BankEntry(candidate=1, value=0.0),
                BankEntry(candidate=2, value=1.0),
            ),
            selected_indices=(0,),
            remaining_indices=(1, 2),
            distance_between_indices=lambda _left, right: (0.0, 0.0, 10.0)[right],
        )

        assert selected_index == 2

    def test_diverse_seed_selection_breaks_eligible_score_ties_by_order(
        self,
    ) -> None:
        selected_index = pick_diverse_low_value_seed(
            entries=(
                BankEntry(candidate=0, value=100.0),
                BankEntry(candidate=1, value=1.0),
                BankEntry(candidate=2, value=1.0),
            ),
            selected_indices=(0,),
            remaining_indices=(1, 2),
            distance_between_indices=lambda _left, _right: 10.0,
        )

        assert selected_index == 1

    def test_diverse_seed_selection_handles_large_finite_total(self) -> None:
        selected_index = pick_diverse_low_value_seed(
            entries=(
                BankEntry(candidate=0, value=100.0),
                BankEntry(candidate=1, value=0.0),
                BankEntry(candidate=2, value=1.0),
            ),
            selected_indices=(0,),
            remaining_indices=(1, 2),
            distance_between_indices=lambda _left, _right: 1e308,
        )

        assert selected_index == 1

    def test_diverse_seed_selection_handles_large_finite_candidate_sums(
        self,
    ) -> None:
        selected_index = pick_diverse_low_value_seed(
            entries=(
                BankEntry(candidate=0, value=100.0),
                BankEntry(candidate=1, value=100.0),
                BankEntry(candidate=2, value=1.0),
                BankEntry(candidate=3, value=0.0),
            ),
            selected_indices=(0, 1),
            remaining_indices=(2, 3),
            distance_between_indices=lambda _left, _right: 1e308,
        )

        assert selected_index == 3

    def test_random_seed_mode_one_is_reproducible_and_prefers_unused_seed_first(
        self,
    ) -> None:
        first_state = prepare_seed_batch(
            current_state=SeedSelectionState(bank_status=(True, False, True)),
            entries=(
                BankEntry(candidate=0, value=100.0),
                BankEntry(candidate=1, value=1.0),
                BankEntry(candidate=2, value=2.0),
            ),
            seed_count=2,
            random_seed_mode=1,
            distance_between_indices=lambda left, right: float(abs(left - right)),
            random_state=np.random.RandomState(1),
        )
        second_state = prepare_seed_batch(
            current_state=SeedSelectionState(bank_status=(True, False, True)),
            entries=(
                BankEntry(candidate=0, value=100.0),
                BankEntry(candidate=1, value=1.0),
                BankEntry(candidate=2, value=2.0),
            ),
            seed_count=2,
            random_seed_mode=1,
            distance_between_indices=lambda left, right: float(abs(left - right)),
            random_state=np.random.RandomState(1),
        )

        assert first_state.active_seed_indices == second_state.active_seed_indices
        assert len(first_state.active_seed_indices) == 2
        assert first_state.active_seed_indices[0] == 1
        assert first_state.active_seed_indices[1] in {0, 2}

    def test_random_seed_mode_two_ignores_bank_status(self) -> None:
        next_state = prepare_seed_batch(
            current_state=SeedSelectionState(
                used_entry_indices=frozenset({0, 1, 2}),
                bank_status=(True, True, True),
            ),
            entries=(
                BankEntry(candidate=10, value=10.0),
                BankEntry(candidate=20, value=20.0),
                BankEntry(candidate=30, value=30.0),
            ),
            seed_count=1,
            random_seed_mode=2,
            masked_seed_indices=frozenset({1, 2}),
            distance_between_indices=lambda left, right: float(abs(left - right)),
            random_state=np.random.RandomState(1),
        )

        assert next_state.active_seed_indices == (0,)
        assert next_state.bank_status[0]

    def test_random_seed_mode_three_uses_lowest_value_seed_first(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=3,
            seed_count=1,
            random_seed_mode=3,
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=20, value=20.0, proposal_id="b0"),
                BankEntry(candidate=2, value=2.0, proposal_id="b1"),
                BankEntry(candidate=5, value=5.0, proposal_id="b2"),
            ),
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 2

    def test_weighted_partner_selection_prefers_nearest_partner(self) -> None:
        partner_indices = select_partner_indices(
            entries=(
                BankEntry(candidate=10, value=10.0),
                BankEntry(candidate=11, value=11.0),
                BankEntry(candidate=110, value=110.0),
            ),
            seed_index=0,
            partner_count=1,
            distance_between_indices=lambda left, right: float(
                abs((10, 11, 110)[left] - (10, 11, 110)[right])
            ),
            weighted_partner_selection=True,
            random_state=np.random.RandomState(0),
        )

        assert partner_indices == (1,)

    def test_weighted_partner_selection_prioritizes_zero_distance_candidates(
        self,
    ) -> None:
        candidates = (10, 10, 10, 110)
        partner_indices = select_partner_indices(
            entries=tuple(
                BankEntry(candidate=candidate, value=float(index))
                for index, candidate in enumerate(candidates)
            ),
            seed_index=0,
            partner_count=2,
            distance_between_indices=lambda left, right: float(
                abs(candidates[left] - candidates[right])
            ),
            weighted_partner_selection=True,
            random_state=np.random.RandomState(0),
        )

        assert set(partner_indices) == {1, 2}

    def test_unmasked_partner_selection_preserves_rng_trajectory(self) -> None:
        entries = tuple(
            BankEntry(candidate=index, value=float(index)) for index in range(32)
        )
        fast_random_state = np.random.RandomState(17)
        fallback_random_state = np.random.RandomState(17)

        fast_indices = select_partner_indices(
            entries=entries,
            seed_index=11,
            partner_count=4,
            partner_mask=frozenset(),
            distance_between_indices=lambda left, right: float(abs(left - right)),
            weighted_partner_selection=False,
            random_state=fast_random_state,
        )
        fallback_indices = select_partner_indices_materialized_reference(
            entries=entries,
            seed_index=11,
            partner_count=4,
            random_state=fallback_random_state,
        )

        assert fast_indices == fallback_indices
        fast_followup = random_state_randints(fast_random_state, 0, 1_000_000, 16)
        fallback_followup = random_state_randints(
            fallback_random_state,
            0,
            1_000_000,
            16,
        )
        assert fast_followup.shape == fallback_followup.shape
        assert fast_followup.dtype == fallback_followup.dtype
        assert fast_followup.tobytes() == fallback_followup.tobytes()

    def test_initial_variation_uses_bank_primary_after_first_cycle(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            initial_variation_operator=EncodeBinaryParents(),
            bank_capacity=2,
            seed_count=1,
            initial_new_bank_cut=1,
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=2,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="r0"),
                BankEntry(candidate=2, value=4.0, proposal_id="r1"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=1,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )
        optimizer.selection_state = type(optimizer.selection_state)(
            used_entry_indices=frozenset({0}),
            bank_status=(True, False),
            active_seed_indices=(0,),
            next_seed_offset=0,
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 1002

    def test_initial_variation_falls_back_when_reference_bank_is_not_full(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            initial_variation_operator=EncodeBinaryParents(),
            bank_capacity=2,
            seed_count=1,
            initial_new_bank_cut=1,
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=2,
            entries=(BankEntry(candidate=1, value=1.0, proposal_id="r0"),),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )
        optimizer.selection_state = type(optimizer.selection_state)(
            used_entry_indices=frozenset({0}),
            active_seed_indices=(0,),
            next_seed_offset=0,
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 10

    def test_initial_new_bank_cut_zero_uses_bank_primary_immediately(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            initial_variation_operator=EncodeBinaryParents(),
            bank_capacity=2,
            seed_count=1,
            initial_new_bank_cut=0,
            perturbation_schedule=perturbation_schedule(
                regular_children_per_seed=0,
                initial_children_per_seed=1,
                shuffle_children=False,
            ),
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=2,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="r0"),
                BankEntry(candidate=2, value=4.0, proposal_id="r1"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )
        optimizer.selection_state = type(optimizer.selection_state)(
            used_entry_indices=frozenset({0}),
            bank_status=(True, False),
            active_seed_indices=(0,),
            next_seed_offset=0,
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 1002

    def test_initial_variation_uses_reference_partners_for_higher_arity(self) -> None:
        initial_operator = RecordingTernaryParents()
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            initial_variation_operator=initial_operator,
            bank_capacity=3,
            seed_count=1,
            initial_new_bank_cut=1,
            perturbation_schedule=perturbation_schedule(
                regular_children_per_seed=0,
                initial_children_per_seed=1,
                shuffle_children=False,
            ),
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
                BankEntry(candidate=30, value=900.0, proposal_id="b2"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=3,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="r0"),
                BankEntry(candidate=2, value=4.0, proposal_id="r1"),
                BankEntry(candidate=3, value=9.0, proposal_id="r2"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )
        optimizer.selection_state = type(optimizer.selection_state)(
            used_entry_indices=frozenset({0}),
            bank_status=(True, False, False),
            active_seed_indices=(0,),
            next_seed_offset=0,
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 1
        assert initial_operator.last_parents is not None
        assert initial_operator.last_parents[0] == 1
        assert set(initial_operator.last_parents[1:]) == {2, 3}

    def test_initial_variation_switches_only_primary_after_new_bank_cut_for_higher_arity(
        self,
    ) -> None:
        initial_operator = RecordingTernaryParents()
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            initial_variation_operator=initial_operator,
            bank_capacity=3,
            seed_count=1,
            initial_new_bank_cut=1,
            perturbation_schedule=perturbation_schedule(
                regular_children_per_seed=0,
                initial_children_per_seed=1,
                shuffle_children=False,
            ),
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=3,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
                BankEntry(candidate=30, value=900.0, proposal_id="b2"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=3,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="r0"),
                BankEntry(candidate=2, value=4.0, proposal_id="r1"),
                BankEntry(candidate=3, value=9.0, proposal_id="r2"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )
        optimizer.selection_state = type(optimizer.selection_state)(
            used_entry_indices=frozenset({0, 1}),
            bank_status=(True, True, False),
            active_seed_indices=(0,),
            next_seed_offset=0,
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 10
        assert initial_operator.last_parents is not None
        assert initial_operator.last_parents[0] == 10
        assert set(initial_operator.last_parents[1:]) == {2, 3}

    def test_initial_new_bank_cut_larger_than_bank_size_remains_reference_primary(
        self,
    ) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            initial_variation_operator=EncodeBinaryParents(),
            bank_capacity=2,
            seed_count=1,
            initial_new_bank_cut=10,
            perturbation_schedule=perturbation_schedule(
                regular_children_per_seed=0,
                initial_children_per_seed=1,
                shuffle_children=False,
            ),
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=2,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="r0"),
                BankEntry(candidate=2, value=4.0, proposal_id="r1"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )
        optimizer.selection_state = type(optimizer.selection_state)(
            used_entry_indices=frozenset({0, 1}),
            bank_status=(True, True),
            active_seed_indices=(0,),
            next_seed_offset=0,
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 102

    def test_initial_variation_uses_reference_primary_before_new_bank_cut(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            initial_variation_operator=EncodeBinaryParents(),
            bank_capacity=2,
            seed_count=1,
            initial_new_bank_cut=1,
            perturbation_schedule=perturbation_schedule(
                regular_children_per_seed=0,
                initial_children_per_seed=1,
                shuffle_children=False,
            ),
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=2,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="r0"),
                BankEntry(candidate=2, value=4.0, proposal_id="r1"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )
        optimizer.selection_state = type(optimizer.selection_state)(
            used_entry_indices=frozenset({0}),
            bank_status=(True, False),
            active_seed_indices=(0,),
            next_seed_offset=0,
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 102

    def test_initial_variation_switches_to_bank_primary_after_new_bank_cut(
        self,
    ) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            initial_variation_operator=EncodeBinaryParents(),
            bank_capacity=2,
            seed_count=1,
            initial_new_bank_cut=1,
            perturbation_schedule=perturbation_schedule(
                regular_children_per_seed=0,
                initial_children_per_seed=1,
                shuffle_children=False,
            ),
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=2,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="r0"),
                BankEntry(candidate=2, value=4.0, proposal_id="r1"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )
        optimizer.selection_state = type(optimizer.selection_state)(
            used_entry_indices=frozenset({0, 1}),
            bank_status=(True, True),
            active_seed_indices=(0,),
            next_seed_offset=0,
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 1002

    def test_initial_routing_matches_legacy_new_bank_cut_rule(self) -> None:
        assert should_use_reference_primary(
            cycle_count=0,
            entry_count=5,
            active_seed_count=2,
            unused_entry_count=3,
            new_bank_cut=1,
        )
        assert not (
            should_use_reference_primary(
                cycle_count=0,
                entry_count=5,
                active_seed_count=2,
                unused_entry_count=2,
                new_bank_cut=1,
            )
        )
        assert not (
            should_use_reference_primary(
                cycle_count=1,
                entry_count=5,
                active_seed_count=2,
                unused_entry_count=3,
                new_bank_cut=1,
            )
        )

    def test_batch_tell_invalidates_seed_state_from_original_bank_delta(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cutoff_schedule=schedule(
                initial_distance_cutoff=5.0,
                minimum_distance_cutoff=5.0,
            ),
            update_policy=CSABankUpdatePolicy(
                minimum_significant_score_gap_ratio=0.02,
            ),
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=5.0,
            minimum_distance_cutoff=5.0,
        )
        optimizer.selection_state = SeedSelectionState(
            used_entry_indices=frozenset({0, 1}),
            bank_status=(True, True),
        )

        proposals = (
            Proposal(candidate=11, proposal_id="p0"),
            Proposal(candidate=12, proposal_id="p1"),
        )
        optimizer.set_pending_proposals(proposals)

        optimizer.tell(
            (
                Observation(
                    proposal=proposals[0], candidate=11, value=97.0, score=97.0
                ),
                Observation(
                    proposal=proposals[1], candidate=12, value=93.0, score=93.0
                ),
            )
        )

        assert tuple(entry.candidate for entry in optimizer.bank.entries) == (12, 20)
        assert optimizer.selection_state.used_entry_indices == frozenset({1})

    def test_batch_tell_seed_invalidation_is_positive_affine_invariant(self) -> None:
        def run_transformed_batch(
            *,
            scale: float,
            offset: float,
        ) -> tuple[
            tuple[int, ...],
            SeedSelectionState,
            tuple[int, int, int, int],
            tuple[int, ...],
        ]:
            def transform(score: float) -> float:
                return scale * score + offset

            optimizer = make_optimizer(
                space=IntegerSpace(low=0, high=2000),
                diversity_metric=AbsoluteDistance(),
                variation_operator=RepeatParent(),
                bank_capacity=2,
                cutoff_schedule=schedule(
                    initial_distance_cutoff=5.0,
                    minimum_distance_cutoff=5.0,
                ),
                update_policy=CSABankUpdatePolicy(
                    minimum_significant_score_gap_ratio=0.01,
                ),
                random_state=0,
            )
            optimizer.bank = Bank(
                capacity=2,
                entries=(
                    BankEntry(
                        candidate=10,
                        value=transform(100.0),
                        proposal_id="b0",
                    ),
                    BankEntry(
                        candidate=20,
                        value=transform(400.0),
                        proposal_id="b1",
                    ),
                ),
            )
            optimizer.cutoff_state = CSACutoffState(
                iteration_count=1,
                cycle_count=0,
                distance_cutoff=5.0,
                minimum_distance_cutoff=5.0,
            )
            optimizer.selection_state = SeedSelectionState(
                used_entry_indices=frozenset({0, 1}),
                bank_status=(True, True),
            )
            proposals = (
                Proposal(candidate=11, proposal_id="p0"),
                Proposal(candidate=12, proposal_id="p1"),
            )
            optimizer.set_pending_proposals(proposals)
            optimizer.tell(
                (
                    Observation(
                        proposal=proposals[0],
                        candidate=11,
                        value=transform(97.0),
                        score=transform(97.0),
                    ),
                    Observation(
                        proposal=proposals[1],
                        candidate=12,
                        value=transform(93.0),
                        score=transform(93.0),
                    ),
                ),
            )
            next_proposals = optimizer.ask(batch_size=1)
            cutoff_state = optimizer.progression_state.cutoff_state
            stage_state = optimizer.progression_state.stage_state
            return (
                tuple(entry.candidate for entry in optimizer.bank.entries),
                optimizer.selection_state,
                (
                    cutoff_state.iteration_count,
                    cutoff_state.cycle_count,
                    stage_state.stage_index,
                    stage_state.stage_round,
                ),
                tuple(proposal.candidate for proposal in next_proposals),
            )

        assert run_transformed_batch(scale=1.0, offset=0.0) == run_transformed_batch(
            scale=100.0,
            offset=7.0,
        )

    def test_seed_count_larger_than_bank_capacity_still_uses_distinct_seeds(
        self,
    ) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((11, 7)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            seed_count=5,
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=2),
            )
        )
        proposal_batch = optimizer.ask(batch_size=2)

        assert len({proposal.candidate for proposal in proposal_batch}) == 2
