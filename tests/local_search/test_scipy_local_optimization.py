"""Tests for SciPy-backed local-search kernels."""

from collections.abc import Mapping
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
    Problem,
    Proposal,
)
from variopt.algorithms.local_search import ScipyMinimizeKernel
from variopt.algorithms.local_search.scipy import kernel as scipy_kernel_module
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
    SpaceBoundaryValue,
)

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

    def test_rejects_non_local_search_kernel_hint(self) -> None:
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

        with pytest.raises(TypeError, match="ProposalLocalSearchContext hints"):
            _ = kernel.run(query, evaluate_query_directly)


def record_real(candidate: RecordCandidate, field_name: str) -> float:
    """Return one canonical real-valued record field."""
    value = candidate[field_name]
    if type(value) is not float:
        msg = f"record field {field_name!r} must be a canonical float"
        raise TypeError(msg)
    return value
