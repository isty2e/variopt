"""Tests for CSA optimizer validation and orchestration boundaries."""

import pytest

from tests.csa_support import (
    AbsoluteDistance,
    Bank,
    BankEntry,
    CollapseToZero,
    CSABankUpdatePolicy,
    CSABiasedPotential,
    CSACutoffSchedule,
    CSACutoffState,
    CSAOptimizerTestCase,
    CSAScoreModel,
    EncodeBinaryParents,
    IntegerSpace,
    NaNDistance,
    NegativeDistance,
    Observation,
    Problem,
    Proposal,
    RepeatParent,
    ScriptedIntegerSpace,
    SequentialEvaluator,
    SquareObjective,
    Study,
    evaluate_observations,
    make_optimizer,
    perturbation_schedule,
    schedule,
)


class CSAOptimizerOrchestrationTests(CSAOptimizerTestCase):
    """White-box tests for CSA orchestration and public run boundaries."""

    def test_rejects_unary_initial_variation_operator(self) -> None:
        with pytest.raises(ValueError):
            _ = make_optimizer(
                space=IntegerSpace(low=0, high=10),
                diversity_metric=AbsoluteDistance(),
                variation_operator=RepeatParent(),
                initial_variation_operator=RepeatParent(),
                bank_capacity=2,
                random_state=0,
            )

    def test_rejects_bank_capacity_smaller_than_initial_variation_arity(self) -> None:
        with pytest.raises(ValueError):
            _ = make_optimizer(
                space=IntegerSpace(low=0, high=10),
                diversity_metric=AbsoluteDistance(),
                variation_operator=RepeatParent(),
                initial_variation_operator=EncodeBinaryParents(),
                bank_capacity=1,
                random_state=0,
            )

    def test_rejects_negative_initial_new_bank_cut(self) -> None:
        with pytest.raises(ValueError):
            _ = make_optimizer(
                space=IntegerSpace(low=0, high=10),
                diversity_metric=AbsoluteDistance(),
                variation_operator=RepeatParent(),
                initial_new_bank_cut=-1,
                bank_capacity=2,
                random_state=0,
            )

    def test_rejects_negative_cycle_limit(self) -> None:
        with pytest.raises(ValueError):
            _ = make_optimizer(
                space=IntegerSpace(low=0, high=10),
                diversity_metric=AbsoluteDistance(),
                variation_operator=RepeatParent(),
                bank_capacity=2,
                cycle_limit=-1,
                random_state=0,
            )

    def test_rejects_zero_batch_size_in_ask(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=10),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            random_state=0,
        )

        with pytest.raises(ValueError):
            _ = optimizer.ask(batch_size=0)

    def test_rejects_zero_exponential_cutoff_reduction_factor(self) -> None:
        with pytest.raises(ValueError):
            _ = make_optimizer(
                space=IntegerSpace(low=0, high=10),
                diversity_metric=AbsoluteDistance(),
                variation_operator=RepeatParent(),
                bank_capacity=2,
                cutoff_schedule=schedule(reduction_factor=0.0),
                random_state=0,
            )

    def test_rejects_negative_linear_cutoff_reduction_factor(self) -> None:
        with pytest.raises(ValueError):
            _ = CSACutoffSchedule(
                reduction_method="linear",
                reduction_factor=-0.1,
            )

    def test_rejects_invalid_random_seed_mode(self) -> None:
        with pytest.raises(ValueError):
            _ = make_optimizer(
                space=IntegerSpace(low=0, high=10),
                diversity_metric=AbsoluteDistance(),
                variation_operator=RepeatParent(),
                bank_capacity=2,
                random_seed_mode=4,
                random_state=0,
            )

    def test_empty_tell_without_pending_is_a_noop(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=10),
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            random_state=0,
        )
        state_before = optimizer.state

        optimizer.tell(())

        assert optimizer.state == state_before

    def test_ask_returns_only_one_child_pool_even_when_more_is_requested(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            seed_count=1,
            cutoff_schedule=schedule(initial_distance_cutoff=1.0),
            perturbation_schedule=perturbation_schedule(
                regular_children_per_seed=3,
                initial_children_per_seed=0,
                shuffle_children=False,
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
        proposals = optimizer.ask(batch_size=10)

        assert len(proposals) == 3

    def test_generation_pool_commits_only_after_all_children_are_observed(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=CollapseToZero(),
            bank_capacity=2,
            seed_count=1,
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

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=2),
            )
        )
        original_bank = optimizer.bank
        proposals = optimizer.ask(batch_size=10)

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                proposals[:1],
            )
        )

        assert optimizer.bank == original_bank
        assert optimizer.state.iteration_count == 0

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                proposals[1:],
            )
        )

        assert optimizer.bank != original_bank
        assert optimizer.state.iteration_count == 1

    def test_biased_potential_degrades_when_runtime_distance_scale_collapses(
        self,
    ) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((0, 0, 0, 0)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=4,
            seed_count=1,
            cutoff_schedule=schedule(
                initial_distance_cutoff=0.0,
                minimum_distance_cutoff=0.0,
            ),
            update_policy=CSABankUpdatePolicy(far_update_mode="crowding_aware"),
            score_model=CSAScoreModel[int](
                biased_potential=CSABiasedPotential(),
            ),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=4),
            )
        )
        proposals = optimizer.ask(batch_size=1)
        optimizer.tell(evaluate_observations(problem, evaluator, proposals))

        assert len(optimizer.bank.entries) == 4

    def test_optimizer_rejects_nan_cutoff_inference_distance(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8)),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=NaNDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=2,
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()

        with pytest.raises(ValueError):
            optimizer.tell(
                evaluate_observations(
                    problem,
                    evaluator,
                    optimizer.ask(batch_size=2),
                )
            )

    def test_optimizer_rejects_negative_seed_selection_distance(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=2000),
            diversity_metric=NegativeDistance(),
            variation_operator=RepeatParent(),
            bank_capacity=3,
            seed_count=2,
            cutoff_schedule=schedule(
                initial_distance_cutoff=1.0,
                minimum_distance_cutoff=1.0,
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
        optimizer.cutoff_state = CSACutoffState(
            iteration_count=1,
            cycle_count=0,
            distance_cutoff=1.0,
            minimum_distance_cutoff=1.0,
        )

        with pytest.raises(ValueError):
            _ = optimizer.ask(batch_size=1)

    def test_optimizer_runs_end_to_end_through_study(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=6),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=CollapseToZero(),
            bank_capacity=3,
            cutoff_schedule=schedule(initial_distance_cutoff=1.0),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer.optimizer,
            evaluator=evaluator,
        )

        result, _ = study.optimize(
            max_evaluations=8,
            initial_state=optimizer.engine_state,
        )

        assert result.best_observation is not None
        assert result.best_observation is not None
        assert result.best_observation.candidate == 0
        assert result.best_observation.value == 0.0
        assert len(result.observations) == 8

    def test_study_optimize_clips_final_batch_without_leaking_pending(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=6),
            objective=SquareObjective(),
        )
        optimizer = make_optimizer(
            space=problem.space,
            diversity_metric=AbsoluteDistance(),
            variation_operator=CollapseToZero(),
            bank_capacity=3,
            cutoff_schedule=schedule(initial_distance_cutoff=1.0),
            random_state=0,
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer.optimizer,
            evaluator=evaluator,
        )

        result, final_state = study.optimize(
            max_evaluations=5,
            batch_size=3,
            initial_state=optimizer.engine_state,
        )
        optimizer.engine_state = final_state

        assert len(result.observations) == 5
        assert len(result.trace.events) == 3
        assert len(optimizer.pending_by_id) == 0

    def test_study_optimize_crosses_refresh_without_leaking_pending(self) -> None:
        problem = Problem(
            space=ScriptedIntegerSpace((9, 8, 7, 1, 2, 0)),
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
        study = Study(
            problem=problem,
            run_method=optimizer.optimizer,
            evaluator=evaluator,
        )

        result, final_state = study.optimize(
            max_evaluations=6,
            batch_size=1,
            initial_state=optimizer.engine_state,
        )
        optimizer.engine_state = final_state

        assert len(result.observations) == 6
        assert not (optimizer.state.refresh_in_progress)
        assert len(optimizer.pending_by_id) == 0
        assert result.best_observation is not None
        assert result.best_observation is not None
        assert result.best_observation.candidate == 1

    def test_tell_rejects_unknown_proposal(self) -> None:
        optimizer = make_optimizer(
            space=IntegerSpace(low=0, high=6),
            diversity_metric=AbsoluteDistance(),
            variation_operator=CollapseToZero(),
            bank_capacity=3,
            cutoff_schedule=schedule(initial_distance_cutoff=1.0),
            random_state=0,
        )
        foreign_observation = Observation(
            proposal=Proposal(candidate=2, proposal_id="foreign"),
            candidate=2,
            value=4.0,
            score=4.0,
        )

        with pytest.raises(ValueError):
            optimizer.tell((foreign_observation,))
