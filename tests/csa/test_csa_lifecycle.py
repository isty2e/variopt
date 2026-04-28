"""Tests for CSA lifecycle, cutoff, refresh, and staged-growth semantics."""

import pytest

from tests.csa_support import (
    AbsoluteDistance,
    Bank,
    BankEntry,
    CSABankGrowthPolicy,
    CSACutoffState,
    CSAOptimizerTestCase,
    CSAStageState,
    EncodeBinaryParents,
    IntegerSpace,
    Problem,
    ReferenceBank,
    RepeatParent,
    ScriptedIntegerSpace,
    SeedSelectionState,
    SequentialEvaluator,
    SquareObjective,
    evaluate_observations,
    make_optimizer,
    perturbation_schedule,
    schedule,
)


class CSALifecycleTests(CSAOptimizerTestCase):
    """White-box tests for CSA cutoff, refresh, and staged lifecycle transitions."""

    def test_state_initializes_from_explicit_cutoff_without_advancing_on_first_bank_fill(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cutoff_schedule=schedule(
                initial_distance_cutoff=3.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=0.5,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()
        problem = Problem(
            space=IntegerSpace(low=0, high=100),
            objective=SquareObjective(),
        )

        first_batch = optimizer.ask(batch_size=2)
        observations = evaluate_observations(
            problem,
            evaluator,
            first_batch,
        )
        optimizer.tell(observations)

        assert optimizer.state.cutoff_state == CSACutoffState(
                iteration_count=0,
                cycle_count=0,
                distance_cutoff=3.0,
                minimum_distance_cutoff=1.0,
            )

    def test_state_derives_cutoff_from_full_bank(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=3,
            cutoff_schedule=schedule(reduction_factor=1.0),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()
        problem = Problem(
            space=IntegerSpace(low=0, high=100),
            objective=SquareObjective(),
        )

        first_batch = optimizer.ask(batch_size=3)
        observations = evaluate_observations(
            problem,
            evaluator,
            first_batch,
        )
        optimizer.tell(observations)

        assert optimizer.state.distance_cutoff is not None
        assert optimizer.state.minimum_distance_cutoff is not None
        assert optimizer.state.distance_cutoff is not None
        assert optimizer.state.minimum_distance_cutoff is not None
        assert optimizer.state.distance_cutoff > 0.0
        assert optimizer.state.minimum_distance_cutoff <= optimizer.state.distance_cutoff
        assert optimizer.state.iteration_count == 0

    def test_state_advances_cutoff_once_bank_is_already_full(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cutoff_schedule=schedule(
                initial_distance_cutoff=4.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=0.5,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()
        problem = Problem(
            space=IntegerSpace(low=0, high=100),
            objective=SquareObjective(),
        )

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=2),
            )
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )

        assert optimizer.state.iteration_count == 1
        assert optimizer.state.cycle_count == 0
        assert optimizer.state.distance_cutoff == 2.0

    def test_state_recovers_cutoff_when_score_gap_increases(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cutoff_schedule=schedule(
                initial_distance_cutoff=4.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=0.5,
                recover_steps=2,
                recover_mode="score_gap_increase",
            ),
            random_state=0,
        )
        optimizer.bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="b0"),
                BankEntry(candidate=3, value=3.0, proposal_id="b1"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=0,
            cycle_count=0,
            distance_cutoff=4.0,
            minimum_distance_cutoff=1.0,
            previous_score_gap=1.0,
        )

        cycle_increment = optimizer.advance_state(
            unused_entry_count=1,
        )

        assert not (cycle_increment)
        assert optimizer.state.distance_cutoff == 16.0
        assert optimizer.state.cutoff_recover_limit == 4.0
        assert optimizer.state.previous_score_gap == 2.0

    def test_cycle_count_increments_only_after_unused_bank_entries_are_exhausted(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cycle_limit=0,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()
        problem = Problem(
            space=IntegerSpace(low=0, high=100),
            objective=SquareObjective(),
        )

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=2),
            )
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )

        assert optimizer.state.iteration_count == 1
        assert optimizer.state.cycle_count == 0
        assert not (optimizer.state.refresh_in_progress)

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )

        assert optimizer.state.iteration_count == 2
        assert optimizer.state.cycle_count == 1
        assert optimizer.state.refresh_in_progress

    def test_cycle_limit_delays_refresh_until_run_boundary(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cycle_limit=3,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()
        problem = Problem(
            space=IntegerSpace(low=0, high=100),
            objective=SquareObjective(),
        )

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=2),
            )
        )
        for _ in range(8):
            optimizer.tell(
                evaluate_observations(
                    problem,
                    evaluator,
                    optimizer.ask(batch_size=1),
                )
            )
            if optimizer.state.refresh_in_progress:
                break

            assert optimizer.state.cycle_count <= 3

        assert optimizer.state.refresh_in_progress
        assert optimizer.state.cycle_count == 4

    def test_non_terminal_cycle_increment_resets_bank_status(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 4)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cycle_limit=3,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
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
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )

        assert sum(
                optimizer.selection_state.bank_status,
            ) == 1

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )

        assert optimizer.state.cycle_count == 1
        assert not (optimizer.state.refresh_in_progress)
        assert (
            optimizer.selection_state.bank_status == (False, False)
        )

    def test_final_stage_exhausts_when_restart_lite_is_disabled(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=100),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            restart_lite=False,
            cycle_limit=0,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()
        problem = Problem(
            space=IntegerSpace(low=0, high=100),
            objective=SquareObjective(),
        )

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=2),
            )
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )

        assert not (optimizer.is_exhausted)

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )

        assert optimizer.is_exhausted
        assert not (optimizer.state.refresh_in_progress)
        with pytest.raises(RuntimeError, match="exhausted"):
            _ = optimizer.ask(batch_size=1)

    def test_optimizer_rejects_stage_and_adaptive_growth_combination(self) -> None:
        with pytest.raises(ValueError, match="adaptive bank growth and staged bank growth must not both be enabled"):
            _ = make_optimizer(
                space=IntegerSpace(low=0, high=100),
                diversity_metric=AbsoluteDistance(),
                variation_operator=RepeatParent(),
                bank_capacity=2,
                max_bank_capacity=4,
                growth_policy=CSABankGrowthPolicy(
                    enabled=True,
                    maximum_capacity=5,
                ),
            )

    def test_stage_growth_masks_old_prefix_for_seed_selection(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((0, 10, 5, 6)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            max_bank_capacity=4,
            cycle_limit=0,
            seed_count=1,
            random_seed_mode=3,
            cutoff_schedule=schedule(
                initial_distance_cutoff=100.0,
                minimum_distance_cutoff=100.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        growth_batch = self.enter_stage_growth(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                growth_batch,
            )
        )

        assert not (optimizer.state.refresh_in_progress)
        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 5

    def test_stage_second_round_unmasks_old_prefix(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((0, 10, 5, 6)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            max_bank_capacity=4,
            cycle_limit=0,
            seed_count=1,
            random_seed_mode=3,
            cutoff_schedule=schedule(
                initial_distance_cutoff=100.0,
                minimum_distance_cutoff=100.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        growth_batch = self.enter_stage_growth(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                growth_batch,
            )
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )

        assert not (optimizer.state.refresh_in_progress)
        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 0

    def test_stage_growth_masks_old_prefix_for_partner_selection(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((0, 10, 5, 6)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=EncodeBinaryParents(),
            bank_capacity=2,
            max_bank_capacity=4,
            cycle_limit=0,
            seed_count=1,
            random_seed_mode=3,
            cutoff_schedule=schedule(
                initial_distance_cutoff=100.0,
                minimum_distance_cutoff=100.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        growth_batch = self.enter_stage_growth(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                growth_batch,
            )
        )
        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 506

    def test_restart_lite_refresh_rebuilds_bank_from_new_samples(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8, 1, 2)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cycle_limit=0,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        self.enter_refresh(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )

        assert optimizer.state.refresh_in_progress
        refresh_batch = optimizer.ask(batch_size=2)
        assert tuple(proposal.candidate for proposal in refresh_batch) == (1, 2)

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                refresh_batch,
            )
        )

        assert not (optimizer.state.refresh_in_progress)
        assert optimizer.state.cycle_count == 0
        refreshed_candidate = optimizer.ask(batch_size=1)[0].candidate
        assert refreshed_candidate in {1, 2}
        assert refreshed_candidate not in {8, 9}

    def test_refresh_waits_for_outstanding_proposals_before_starting(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8, 1, 2)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cycle_limit=0,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            perturbation_schedule=perturbation_schedule(
                regular_children_per_seed=2,
                initial_children_per_seed=0,
                shuffle_children=False,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        self.fill_bank(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=2),
            )
        )
        stale_batch = optimizer.ask(batch_size=2)

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                stale_batch[:1],
            )
        )

        assert not (optimizer.state.refresh_in_progress)
        with pytest.raises(RuntimeError):
            _ = optimizer.ask(batch_size=1)

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                stale_batch[1:],
            )
        )

        assert optimizer.state.refresh_in_progress
        refresh_batch = optimizer.ask(batch_size=2)
        assert tuple(proposal.candidate for proposal in refresh_batch) == (1, 2)

    def test_refresh_completion_waits_for_outstanding_refresh_proposals(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8, 1, 2, 0)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cycle_limit=0,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        self.enter_refresh(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )

        refresh_batch = optimizer.ask(batch_size=3)
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                refresh_batch[:2],
            )
        )

        assert optimizer.state.refresh_in_progress
        with pytest.raises(RuntimeError):
            _ = optimizer.ask(batch_size=1)

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                refresh_batch[2:],
            )
        )

        assert not (optimizer.state.refresh_in_progress)
        assert (
            {entry.candidate for entry in optimizer.bank.entries} == {0, 1}
        )
        assert (
            {entry.candidate for entry in optimizer.reference_bank.entries} == {0, 1}
        )

    def test_refresh_start_pending_empty_tell_is_a_noop(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8, 1, 2, 0)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cycle_limit=0,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            perturbation_schedule=perturbation_schedule(
                regular_children_per_seed=2,
                initial_children_per_seed=0,
                shuffle_children=False,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        self.fill_bank(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=2),
            )
        )
        stale_batch = optimizer.ask(batch_size=2)
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                stale_batch[:1],
            )
        )
        state_before_empty_tell = optimizer.state

        optimizer.tell(())

        assert optimizer.state == state_before_empty_tell
        assert not (optimizer.state.refresh_in_progress)

    def test_refresh_completion_pending_empty_tell_is_a_noop(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8, 1, 2, 0)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cycle_limit=0,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        self.enter_refresh(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )
        refresh_batch = optimizer.ask(batch_size=3)
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                refresh_batch[:2],
            )
        )
        state_before_empty_tell = optimizer.state

        optimizer.tell(())

        assert optimizer.state == state_before_empty_tell
        assert optimizer.state.refresh_in_progress
        with pytest.raises(RuntimeError):
            _ = optimizer.ask(batch_size=1)

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                refresh_batch[2:],
            )
        )

        assert not (optimizer.state.refresh_in_progress)

    def test_refresh_overfill_keeps_late_improvement_in_refreshed_bank(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8, 1, 2, 0)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            cycle_limit=0,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        self.enter_refresh(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )
        refresh_batch = optimizer.ask(batch_size=3)

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                refresh_batch,
            )
        )

        assert not (optimizer.state.refresh_in_progress)
        assert (
            {entry.candidate for entry in optimizer.bank.entries} == {0, 1}
        )
        assert (
            {entry.candidate for entry in optimizer.reference_bank.entries} == {0, 1}
        )

    def test_stage_growth_appends_new_entries_to_reference_bank_without_overwriting_prefix(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((1, 2)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            max_bank_capacity=4,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
                reduction_factor=1.0,
                stagnation_update_limit=0,
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()
        optimizer.bank = Bank(
            capacity=2,
            entries=(
                BankEntry(candidate=7, value=49.0, proposal_id="b0"),
                BankEntry(candidate=8, value=64.0, proposal_id="b1"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=2,
            entries=(
                BankEntry(candidate=9, value=81.0, proposal_id="r0"),
                BankEntry(candidate=8, value=64.0, proposal_id="r1"),
            ),
        )
        transition = optimizer.lifecycle_state.stage_state.next_transition()
        assert transition is not None

        optimizer.begin_stage_transition(transition)
        refresh_batch = optimizer.ask(batch_size=2)
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                refresh_batch,
            )
        )

        assert (
            tuple(entry.candidate for entry in optimizer.bank.entries) == (7, 8, 1, 2)
        )
        assert (
            tuple(entry.candidate for entry in optimizer.reference_bank.entries)
            == (9, 8, 1, 2)
        )

    def test_stage_mask_does_not_reduce_initial_new_bank_cut_entry_count(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=5000),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            initial_variation_operator=EncodeBinaryParents(),
            bank_capacity=4,
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
            capacity=4,
            entries=(
                BankEntry(candidate=10, value=100.0, proposal_id="b0"),
                BankEntry(candidate=20, value=400.0, proposal_id="b1"),
                BankEntry(candidate=30, value=900.0, proposal_id="b2"),
                BankEntry(candidate=40, value=1600.0, proposal_id="b3"),
            ),
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=4,
            entries=(
                BankEntry(candidate=1, value=1.0, proposal_id="r0"),
                BankEntry(candidate=2, value=4.0, proposal_id="r1"),
                BankEntry(candidate=3, value=9.0, proposal_id="r2"),
                BankEntry(candidate=4, value=16.0, proposal_id="r3"),
            ),
        )
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )
        optimizer.lifecycle_state = type(optimizer.lifecycle_state)(
            stage_state=CSAStageState(
                base_capacity=2,
                max_capacity=4,
                stage_index=1,
                stage_round=0,
                seed_mask=frozenset({0, 1}),
                partner_mask=frozenset({0, 1}),
            ),
            base_cycle_limit=optimizer.lifecycle_state.base_cycle_limit,
            restart_lite=optimizer.lifecycle_state.restart_lite,
        )
        optimizer.selection_state = SeedSelectionState(
            used_entry_indices=frozenset({2}),
            bank_status=(True, True, True, False),
            active_seed_indices=(2,),
            next_seed_offset=0,
        )

        proposal = optimizer.ask(batch_size=1)[0]

        assert proposal.candidate == 3004
