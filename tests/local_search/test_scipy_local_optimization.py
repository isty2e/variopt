"""Tests for SciPy-backed local-search kernels."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import log10
from typing import TypeVar

import pytest
from typing_extensions import override

from tests.numeric_support import approx_equal
from variopt import (
    EvaluationOutcome,
    Objective,
    Observation,
    OptimizationDirection,
    Problem,
    Proposal,
)
from variopt.algorithms.local_search import ScipyMinimizeKernel
from variopt.algorithms.local_search.scipy import kernel as scipy_kernel_module
from variopt.artifacts import ProposalEvaluationSpec
from variopt.execution import (
    ExecutionResources,
    NestedParallelismPolicy,
)
from variopt.kernel import (
    KernelStatus,
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from variopt.spaces import (
    IntegerSpace,
    RealSpace,
    RecordCandidate,
    RecordSpace,
    SearchSpace,
    SpaceBoundaryValue,
)
from variopt.spaces.projections import ContinuousStructuredSpaceCodec

BoundaryRunnerT = TypeVar("BoundaryRunnerT")
CandidateRunnerT = TypeVar("CandidateRunnerT")


def evaluate_query_directly(
    query: ProposalBatchQuery[BoundaryRunnerT, CandidateRunnerT],
) -> tuple[EvaluationOutcome[CandidateRunnerT], ...]:
    """Evaluate one proposal batch directly through the problem objective."""
    return tuple(
        EvaluationOutcome(
            observation=Observation.from_objective_value(
                proposal=proposal,
                candidate=proposal.candidate,
                value=query.problem.objective.evaluate(proposal.candidate),
                direction=query.problem.direction,
            ),
            evaluation_count=1,
        )
        for proposal in query.proposals
    )


class ShiftedSquareObjective(Objective[float]):
    """One-dimensional continuous objective with a known optimum."""

    @override
    def evaluate(self, candidate: float) -> float:
        return (candidate - 1.5) ** 2


class ShiftedPeakObjective(Objective[float]):
    """One-dimensional maximization objective with a known peak."""

    @override
    def evaluate(self, candidate: float) -> float:
        return 10.0 - ((candidate - 1.5) ** 2)


class MixedRecordObjective(Objective[RecordCandidate]):
    """Continuous structured objective with one log-scaled coordinate."""

    @override
    def evaluate(self, candidate: RecordCandidate) -> float:
        x = record_real(candidate, "x")
        y = record_real(candidate, "y")
        return (log10(x) + 1.0) ** 2 + ((y - 2.0) ** 2)


class IntegerObjective(Objective[int]):
    """Discrete objective used to reject non-continuous spaces."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


@dataclass(frozen=True, slots=True)
class FakeScipyOptimizeResult:
    """Typed stand-in for one SciPy optimize result in tests."""

    x: tuple[float, ...]
    fun: float
    nfev: int
    success: bool
    message: str | None


@dataclass(frozen=True, slots=True)
class DummyProposalKernelHint(ProposalKernelHint):
    """Unexpected generic kernel hint used to test local-search validation."""


class ScipyMinimizeKernelTests:
    """Regression tests for SciPy-backed local-search kernels."""

    def make_query(
        self,
        *,
        problem: Problem[float | int, float],
        candidate: float,
    ) -> ProposalBatchQuery[float | int, float]:
        """Return one canonical single-proposal query."""
        return ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=candidate, proposal_id="p-1"),),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
        )

    def test_lbfgsb_improves_one_dimensional_real_problem(self) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")

        outcomes = kernel.run(
            self.make_query(problem=problem, candidate=4.0),
            evaluate_query_directly,
        )

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert approx_equal(
            outcome.observation.candidate,
            1.5,
            rel=0.0,
            abs=10 ** (-(5)),
        )
        assert outcome.observation.value < 1e-10
        assert outcome.evaluation_count > 0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "L-BFGS-B"
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED
        assert outcome.refinement is not None
        assert outcome.refinement.source_candidate == 4.0
        assert approx_equal(
            outcome.refinement.refined_candidate,
            1.5,
            rel=0.0,
            abs=10 ** (-(5)),
        )
        assert outcome.refinement.changed_leaf_paths == ((),)

    def test_lbfgsb_respects_maximize_direction(self) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedPeakObjective(),
            direction=OptimizationDirection.MAXIMIZE,
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        initial_value = problem.objective.evaluate(4.0)

        outcomes = kernel.run(
            self.make_query(problem=problem, candidate=4.0),
            evaluate_query_directly,
        )

        outcome = outcomes[0]
        assert approx_equal(
            outcome.observation.candidate,
            1.5,
            rel=0.0,
            abs=10 ** (-(5)),
        )
        assert outcome.observation.value > initial_value
        assert outcome.observation.score < -initial_value
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED

    def test_scipy_fun_is_treated_as_score_not_raw_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedPeakObjective(),
            direction=OptimizationDirection.MAXIMIZE,
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        objective_values_seen: list[float] = []

        def fake_run_scipy_minimize(
            *,
            objective_in_coordinate_space: Callable[[Sequence[float]], float],
            initial_coordinates: tuple[float, ...],
            method: str,
            coordinate_bounds: tuple[tuple[float, float], ...],
            tolerance: float | None,
            options: dict[str, int],
        ) -> FakeScipyOptimizeResult:
            _ = (
                initial_coordinates,
                method,
                coordinate_bounds,
                tolerance,
                options,
            )
            objective_score = objective_in_coordinate_space((1.5,))
            objective_values_seen.append(objective_score)
            return FakeScipyOptimizeResult(
                x=(1.5,),
                fun=objective_score,
                nfev=1,
                success=True,
                message="ok",
            )

        monkeypatch.setattr(
            scipy_kernel_module,
            "run_scipy_minimize",
            fake_run_scipy_minimize,
        )

        outcomes = kernel.run(
            self.make_query(problem=problem, candidate=4.0),
            evaluate_query_directly,
        )

        outcome = outcomes[0]
        assert objective_values_seen == [-10.0]
        assert outcome.observation.value == 10.0
        assert outcome.observation.score == -10.0
        assert outcome.evaluation_count == 1

    def test_scipy_unevaluated_final_coordinates_are_evaluated_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedPeakObjective(),
            direction=OptimizationDirection.MAXIMIZE,
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")

        def fake_run_scipy_minimize(
            *,
            objective_in_coordinate_space: Callable[[Sequence[float]], float],
            initial_coordinates: tuple[float, ...],
            method: str,
            coordinate_bounds: tuple[tuple[float, float], ...],
            tolerance: float | None,
            options: dict[str, int],
        ) -> FakeScipyOptimizeResult:
            _ = (
                method,
                coordinate_bounds,
                tolerance,
                options,
            )
            objective_score = objective_in_coordinate_space(initial_coordinates)
            return FakeScipyOptimizeResult(
                x=(1.5,),
                fun=objective_score,
                nfev=1,
                success=True,
                message="ok",
            )

        monkeypatch.setattr(
            scipy_kernel_module,
            "run_scipy_minimize",
            fake_run_scipy_minimize,
        )

        outcomes = kernel.run(
            self.make_query(problem=problem, candidate=4.0),
            evaluate_query_directly,
        )

        outcome = outcomes[0]
        assert outcome.observation.proposal.proposal_id == "p-1"
        assert outcome.observation.candidate == 1.5
        assert outcome.observation.value == 10.0
        assert outcome.evaluation_count == 2

    def test_sequence_coordinate_cache_preserves_elapsed_seconds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedPeakObjective(),
            direction=OptimizationDirection.MAXIMIZE,
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        runner_call_count = 0

        def fake_run_scipy_minimize(
            *,
            objective_in_coordinate_space: Callable[[Sequence[float]], float],
            initial_coordinates: tuple[float, ...],
            method: str,
            coordinate_bounds: tuple[tuple[float, float], ...],
            tolerance: float | None,
            options: dict[str, int],
        ) -> FakeScipyOptimizeResult:
            _ = (
                initial_coordinates,
                method,
                coordinate_bounds,
                tolerance,
                options,
            )
            objective_score = objective_in_coordinate_space([1.5])
            return FakeScipyOptimizeResult(
                x=(1.5,),
                fun=objective_score,
                nfev=1,
                success=True,
                message="ok",
            )

        def elapsed_runner(
            local_query: ProposalBatchQuery[float | int, float],
        ) -> tuple[EvaluationOutcome[float], ...]:
            nonlocal runner_call_count
            runner_call_count += 1
            proposal = local_query.proposals[0]
            return (
                EvaluationOutcome(
                    observation=Observation.from_objective_value(
                        proposal=proposal,
                        candidate=proposal.candidate,
                        value=local_query.problem.objective.evaluate(proposal.candidate),
                        direction=local_query.problem.direction,
                        elapsed_seconds=0.25,
                    ),
                    evaluation_count=1,
                ),
            )

        monkeypatch.setattr(
            scipy_kernel_module,
            "run_scipy_minimize",
            fake_run_scipy_minimize,
        )

        outcomes = kernel.run(
            self.make_query(problem=problem, candidate=4.0),
            elapsed_runner,
        )

        outcome = outcomes[0]
        assert runner_call_count == 1
        assert outcome.observation.elapsed_seconds == 0.25
        assert outcome.evaluation_count == 1

    def test_stopped_scipy_result_uses_final_evaluation_record(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedPeakObjective(),
            direction=OptimizationDirection.MAXIMIZE,
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")

        def fake_run_scipy_minimize(
            *,
            objective_in_coordinate_space: Callable[[Sequence[float]], float],
            initial_coordinates: tuple[float, ...],
            method: str,
            coordinate_bounds: tuple[tuple[float, float], ...],
            tolerance: float | None,
            options: dict[str, int],
        ) -> FakeScipyOptimizeResult:
            _ = (
                initial_coordinates,
                method,
                coordinate_bounds,
                tolerance,
                options,
            )
            objective_score = objective_in_coordinate_space((1.5,))
            return FakeScipyOptimizeResult(
                x=(1.5,),
                fun=objective_score,
                nfev=1,
                success=False,
                message="iteration limit",
            )

        monkeypatch.setattr(
            scipy_kernel_module,
            "run_scipy_minimize",
            fake_run_scipy_minimize,
        )

        outcomes = kernel.run(
            self.make_query(problem=problem, candidate=4.0),
            evaluate_query_directly,
        )

        outcome = outcomes[0]
        assert outcome.observation.value == 10.0
        assert outcome.observation.score == -10.0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == "iteration limit"

    def test_powell_improves_log_scaled_record_problem(self) -> None:
        space = RecordSpace(
            x=RealSpace(1e-4, 10.0, scale="log"),
            y=RealSpace(-5.0, 5.0),
        )
        problem = Problem(
            space=space,
            objective=MixedRecordObjective(),
        )
        kernel = ScipyMinimizeKernel[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ](method="Powell")
        initial_candidate = space.normalize({"x": 1.0, "y": -3.0})
        initial_value = problem.objective.evaluate(initial_candidate)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=initial_candidate, proposal_id="p-1"),),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        space.validate(outcome.observation.candidate)
        optimized_value = problem.objective.evaluate(outcome.observation.candidate)
        assert optimized_value < initial_value
        assert approx_equal(
            record_real(outcome.observation.candidate, "x"),
            0.1,
            rel=0.0,
            abs=10 ** (-(3)),
        )
        assert approx_equal(
            record_real(outcome.observation.candidate, "y"),
            2.0,
            rel=0.0,
            abs=10 ** (-(3)),
        )
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "Powell"
        assert outcome.refinement is not None
        assert outcome.refinement.source_candidate == initial_candidate
        assert outcome.refinement.refined_candidate == outcome.observation.candidate
        assert outcome.refinement.changed_leaf_paths == (("x",), ("y",))

    def test_rejects_non_continuous_integer_problem(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 10),
            objective=IntegerObjective(),
        )
        kernel = ScipyMinimizeKernel[int, int](method="Powell")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=4, proposal_id="p-1"),),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
        )

        with pytest.raises(TypeError, match="RealSpace"):
            _ = kernel.run(query, evaluate_query_directly)

    def test_context_can_disable_local_search(self) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=4.0, proposal_id="p-1"),),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(enabled=False),
            ),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == 4.0
        assert outcome.observation.value == 6.25
        assert outcome.evaluation_count == 1
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == "local search disabled by run-method context"
        assert outcome.refinement is None

    def test_disabled_local_search_forwards_proposal_evaluation_spec(self) -> None:
        class LocalSpec(ProposalEvaluationSpec):
            """Test-local proposal metadata marker."""

        spec = LocalSpec()
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=4.0, proposal_id="p-1"),),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
            proposal_evaluation_specs=(spec,),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(enabled=False),
            ),
        )

        def assert_spec_forwarded(
            local_query: ProposalBatchQuery[float | int, float],
        ) -> tuple[EvaluationOutcome[float], ...]:
            assert local_query.proposal_evaluation_specs == (spec,)
            proposal = local_query.proposals[0]
            return (
                EvaluationOutcome(
                    observation=Observation.from_objective_value(
                        proposal=proposal,
                        proposal_evaluation_spec=spec,
                        candidate=proposal.candidate,
                        value=local_query.problem.objective.evaluate(proposal.candidate),
                        direction=local_query.problem.direction,
                    ),
                    evaluation_count=1,
                ),
            )

        outcomes = kernel.run(query, assert_spec_forwarded)

        assert outcomes[0].observation.proposal_evaluation_spec is spec

    def test_enabled_local_search_forwards_spec_to_final_evaluation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class LocalSpec(ProposalEvaluationSpec):
            """Test-local proposal metadata marker."""

        spec = LocalSpec()
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedPeakObjective(),
            direction=OptimizationDirection.MAXIMIZE,
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=4.0, proposal_id="p-1"),),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
            proposal_evaluation_specs=(spec,),
        )

        def fake_run_scipy_minimize(
            *,
            objective_in_coordinate_space: Callable[[Sequence[float]], float],
            initial_coordinates: tuple[float, ...],
            method: str,
            coordinate_bounds: tuple[tuple[float, float], ...],
            tolerance: float | None,
            options: dict[str, int],
        ) -> FakeScipyOptimizeResult:
            _ = (
                objective_in_coordinate_space,
                initial_coordinates,
                method,
                coordinate_bounds,
                tolerance,
                options,
            )
            return FakeScipyOptimizeResult(
                x=(1.5,),
                fun=-10.0,
                nfev=0,
                success=True,
                message="ok",
            )

        def assert_spec_forwarded(
            local_query: ProposalBatchQuery[float | int, float],
        ) -> tuple[EvaluationOutcome[float], ...]:
            assert local_query.proposal_evaluation_specs == (spec,)
            proposal = local_query.proposals[0]
            return (
                EvaluationOutcome(
                    observation=Observation.from_objective_value(
                        proposal=proposal,
                        proposal_evaluation_spec=spec,
                        candidate=proposal.candidate,
                        value=local_query.problem.objective.evaluate(proposal.candidate),
                        direction=local_query.problem.direction,
                    ),
                    evaluation_count=1,
                ),
            )

        monkeypatch.setattr(
            scipy_kernel_module,
            "run_scipy_minimize",
            fake_run_scipy_minimize,
        )

        outcomes = kernel.run(query, assert_spec_forwarded)

        assert outcomes[0].observation.proposal_evaluation_spec is spec
        assert outcomes[0].observation.proposal.proposal_id == "p-1"
        assert outcomes[0].observation.value == 10.0

    def test_context_can_override_scipy_iteration_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](
            method="L-BFGS-B",
            max_iterations=20,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=4.0, proposal_id="p-1"),),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(local_budget=3),
            ),
        )
        captured_options: list[dict[str, int]] = []

        def fake_run_scipy_minimize(
            *,
            objective_in_coordinate_space: object,
            initial_coordinates: tuple[float, ...],
            method: str,
            coordinate_bounds: tuple[tuple[float, float], ...],
            tolerance: float | None,
            options: dict[str, int],
        ) -> FakeScipyOptimizeResult:
            _ = (
                objective_in_coordinate_space,
                initial_coordinates,
                method,
                coordinate_bounds,
                tolerance,
            )
            captured_options.append(options)
            return FakeScipyOptimizeResult(
                x=(1.5,),
                fun=0.0,
                nfev=4,
                success=True,
                message="ok",
            )

        monkeypatch.setattr(
            scipy_kernel_module,
            "run_scipy_minimize",
            fake_run_scipy_minimize,
        )
        _ = kernel.run(query, evaluate_query_directly)

        assert captured_options == [{"maxiter": 3}]

    def test_structured_codec_is_prepared_once_per_query(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(
                Proposal(candidate=4.0, proposal_id="p-1"),
                Proposal(candidate=-2.0, proposal_id="p-2"),
            ),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
        )
        codec_call_count = 0

        def count_from_space(
            cls: type[ContinuousStructuredSpaceCodec[float | int, float]],
            space: SearchSpace[float | int, float],
        ) -> ContinuousStructuredSpaceCodec[float | int, float]:
            nonlocal codec_call_count
            _ = cls
            codec_call_count += 1
            if not isinstance(space, RealSpace):
                msg = "test codec only supports RealSpace"
                raise TypeError(msg)
            return ContinuousStructuredSpaceCodec(
                space=space,
                leaf_paths=space.leaf_paths(),
                leaf_spaces=(space,),
            )

        def fake_run_scipy_minimize(
            *,
            objective_in_coordinate_space: object,
            initial_coordinates: tuple[float, ...],
            method: str,
            coordinate_bounds: tuple[tuple[float, float], ...],
            tolerance: float | None,
            options: dict[str, int],
        ) -> FakeScipyOptimizeResult:
            _ = (
                objective_in_coordinate_space,
                initial_coordinates,
                method,
                coordinate_bounds,
                tolerance,
                options,
            )
            return FakeScipyOptimizeResult(
                x=(1.5,),
                fun=0.0,
                nfev=1,
                success=True,
                message="ok",
            )

        monkeypatch.setattr(
            ContinuousStructuredSpaceCodec,
            "from_space",
            classmethod(count_from_space),
        )
        monkeypatch.setattr(
            scipy_kernel_module,
            "run_scipy_minimize",
            fake_run_scipy_minimize,
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        assert len(outcomes) == 2
        assert codec_call_count == 1

    def test_disabled_local_search_does_not_prepare_structured_codec(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=4.0, proposal_id="p-1"),),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
            proposal_kernel_hints=(ProposalLocalSearchContext(enabled=False),),
        )

        def reject_from_space(
            cls: type[ContinuousStructuredSpaceCodec[float | int, float]],
            space: SearchSpace[float | int, float],
        ) -> ContinuousStructuredSpaceCodec[float | int, float]:
            _ = (cls, space)
            raise AssertionError("disabled local search should not prepare codec")

        monkeypatch.setattr(
            ContinuousStructuredSpaceCodec,
            "from_space",
            classmethod(reject_from_space),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        assert outcomes[0].observation.candidate == 4.0

    def test_all_disabled_local_search_does_not_prepare_structured_codec(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(
                Proposal(candidate=4.0, proposal_id="p-1"),
                Proposal(candidate=-2.0, proposal_id="p-2"),
            ),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(enabled=False),
                ProposalLocalSearchContext(enabled=False),
            ),
        )

        def reject_from_space(
            cls: type[ContinuousStructuredSpaceCodec[float | int, float]],
            space: SearchSpace[float | int, float],
        ) -> ContinuousStructuredSpaceCodec[float | int, float]:
            _ = (cls, space)
            raise AssertionError("disabled local search should not prepare codec")

        monkeypatch.setattr(
            ContinuousStructuredSpaceCodec,
            "from_space",
            classmethod(reject_from_space),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        assert tuple(outcome.observation.candidate for outcome in outcomes) == (4.0, -2.0)

    def test_empty_local_search_batch_does_not_prepare_structured_codec(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
        )

        def reject_from_space(
            cls: type[ContinuousStructuredSpaceCodec[float | int, float]],
            space: SearchSpace[float | int, float],
        ) -> ContinuousStructuredSpaceCodec[float | int, float]:
            _ = (cls, space)
            raise AssertionError("empty local-search batch should not prepare codec")

        def reject_runner(
            empty_query: ProposalBatchQuery[float | int, float],
        ) -> tuple[EvaluationOutcome[float], ...]:
            _ = empty_query
            raise AssertionError("empty local-search batch should not call runner")

        monkeypatch.setattr(
            ContinuousStructuredSpaceCodec,
            "from_space",
            classmethod(reject_from_space),
        )

        outcomes = kernel.run(query, reject_runner)

        assert outcomes == ()

    def test_empty_local_search_batch_allows_incompatible_space_without_codec(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 10),
            objective=IntegerObjective(),
        )
        kernel = ScipyMinimizeKernel[int, int](method="Powell")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        assert outcomes == ()

    def test_mixed_disabled_and_enabled_query_prepares_structured_codec_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(
                Proposal(candidate=4.0, proposal_id="p-1"),
                Proposal(candidate=-2.0, proposal_id="p-2"),
            ),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(enabled=False),
                None,
            ),
        )
        codec_call_count = 0

        def count_from_space(
            cls: type[ContinuousStructuredSpaceCodec[float | int, float]],
            space: SearchSpace[float | int, float],
        ) -> ContinuousStructuredSpaceCodec[float | int, float]:
            nonlocal codec_call_count
            _ = cls
            codec_call_count += 1
            if not isinstance(space, RealSpace):
                msg = "test codec only supports RealSpace"
                raise TypeError(msg)
            return ContinuousStructuredSpaceCodec(
                space=space,
                leaf_paths=space.leaf_paths(),
                leaf_spaces=(space,),
            )

        def fake_run_scipy_minimize(
            *,
            objective_in_coordinate_space: object,
            initial_coordinates: tuple[float, ...],
            method: str,
            coordinate_bounds: tuple[tuple[float, float], ...],
            tolerance: float | None,
            options: dict[str, int],
        ) -> FakeScipyOptimizeResult:
            _ = (
                objective_in_coordinate_space,
                initial_coordinates,
                method,
                coordinate_bounds,
                tolerance,
                options,
            )
            return FakeScipyOptimizeResult(
                x=(1.5,),
                fun=0.0,
                nfev=1,
                success=True,
                message="ok",
            )

        monkeypatch.setattr(
            ContinuousStructuredSpaceCodec,
            "from_space",
            classmethod(count_from_space),
        )
        monkeypatch.setattr(
            scipy_kernel_module,
            "run_scipy_minimize",
            fake_run_scipy_minimize,
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        assert tuple(outcome.observation.candidate for outcome in outcomes) == (4.0, 1.5)
        assert codec_call_count == 1

    def test_structured_codec_cache_is_query_local(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first_space = RealSpace(-5.0, 5.0)
        second_space = RealSpace(10.0, 20.0)
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        execution_resources = ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )
        first_query = ProposalBatchQuery(
            problem=Problem(space=first_space, objective=ShiftedSquareObjective()),
            proposals=(Proposal(candidate=4.0, proposal_id="p-1"),),
            execution_resources=execution_resources,
        )
        second_query = ProposalBatchQuery(
            problem=Problem(space=second_space, objective=ShiftedSquareObjective()),
            proposals=(Proposal(candidate=15.0, proposal_id="p-2"),),
            execution_resources=execution_resources,
        )
        spaces_seen: list[RealSpace] = []

        def count_from_space(
            cls: type[ContinuousStructuredSpaceCodec[float | int, float]],
            space: SearchSpace[float | int, float],
        ) -> ContinuousStructuredSpaceCodec[float | int, float]:
            _ = cls
            if not isinstance(space, RealSpace):
                msg = "test codec only supports RealSpace"
                raise TypeError(msg)
            spaces_seen.append(space)
            return ContinuousStructuredSpaceCodec(
                space=space,
                leaf_paths=space.leaf_paths(),
                leaf_spaces=(space,),
            )

        def fake_run_scipy_minimize(
            *,
            objective_in_coordinate_space: object,
            initial_coordinates: tuple[float, ...],
            method: str,
            coordinate_bounds: tuple[tuple[float, float], ...],
            tolerance: float | None,
            options: dict[str, int],
        ) -> FakeScipyOptimizeResult:
            _ = (
                objective_in_coordinate_space,
                method,
                coordinate_bounds,
                tolerance,
                options,
            )
            return FakeScipyOptimizeResult(
                x=initial_coordinates,
                fun=0.0,
                nfev=1,
                success=True,
                message="ok",
            )

        monkeypatch.setattr(
            ContinuousStructuredSpaceCodec,
            "from_space",
            classmethod(count_from_space),
        )
        monkeypatch.setattr(
            scipy_kernel_module,
            "run_scipy_minimize",
            fake_run_scipy_minimize,
        )

        first_outcomes = kernel.run(first_query, evaluate_query_directly)
        second_outcomes = kernel.run(second_query, evaluate_query_directly)

        assert tuple(spaces_seen) == (first_space, second_space)
        assert first_outcomes[0].observation.candidate == 4.0
        assert second_outcomes[0].observation.candidate == 15.0

    def test_rejects_non_local_search_kernel_hint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=RealSpace(-5.0, 5.0),
            objective=ShiftedSquareObjective(),
        )
        kernel = ScipyMinimizeKernel[float | int, float](method="L-BFGS-B")
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=4.0, proposal_id="p-1"),),
            execution_resources=ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=1,
                owner_backend="sequential",
            ),
            proposal_kernel_hints=(DummyProposalKernelHint(),),
        )

        def reject_from_space(
            cls: type[ContinuousStructuredSpaceCodec[float | int, float]],
            space: SearchSpace[float | int, float],
        ) -> ContinuousStructuredSpaceCodec[float | int, float]:
            _ = (cls, space)
            raise AssertionError("invalid kernel hint should fail before codec setup")

        monkeypatch.setattr(
            ContinuousStructuredSpaceCodec,
            "from_space",
            classmethod(reject_from_space),
        )

        with pytest.raises(TypeError, match="ProposalLocalSearchContext hints"):
            _ = kernel.run(query, evaluate_query_directly)


def record_real(candidate: RecordCandidate, field_name: str) -> float:
    """Return one canonical real-valued record field."""
    value = candidate[field_name]
    if type(value) is not float:
        msg = f"record field {field_name!r} must be a canonical float"
        raise TypeError(msg)
    return value
