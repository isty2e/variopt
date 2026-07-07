"""Tests for structured discrete local-search kernels."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar

import numpy as np
import pytest
from typing_extensions import override

from tests.study_support import BatchQueueOptimizer
from variopt import (
    ArraySpace,
    CategoricalSpace,
    EvaluationAttemptBatch,
    EvaluationBudget,
    EvaluationExceptionSnapshot,
    EvaluationFailure,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Observation,
    Problem,
    Proposal,
    RealSpace,
    Study,
)
from variopt.algorithms.local_search import (
    StructuredHillClimbKernel,
    StructuredIteratedLocalSearchKernel,
    StructuredKickPolicy,
    StructuredScheduledLocalSearchKernel,
    StructuredStochasticNeighborhoodKernel,
    StructuredVariableNeighborhoodKernel,
    StructuredVariableNeighborhoodStage,
)
from variopt.artifacts import EvaluationSuccess, KernelStatus, ObservationPayload
from variopt.evaluators import SequentialEvaluator
from variopt.execution import (
    ExecutionResources,
    NestedParallelismPolicy,
)
from variopt.kernel import (
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from variopt.randomness import RandomStateSnapshot
from variopt.spaces import (
    LeafPath,
    RecordCandidate,
    RecordSpace,
    SpaceBoundaryValue,
    StructuredLeafSpace,
    StructuredSearchSpace,
)
from variopt.spaces.types import SpaceCandidateValue

Color: TypeAlias = Literal["red", "green", "blue"]
BoundaryRunnerT = TypeVar("BoundaryRunnerT")
CandidateRunnerT = TypeVar("CandidateRunnerT")
ConditionalDiscreteCandidate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DummyProposalKernelHint(ProposalKernelHint):
    """Unexpected generic kernel hint used to test local-search validation."""


def evaluate_query_directly(
    query: ProposalBatchQuery[BoundaryRunnerT, CandidateRunnerT, ObservationPayload],
) -> EvaluationAttemptBatch[CandidateRunnerT, ObservationPayload]:
    """Evaluate one proposal batch directly through the problem objective."""
    if query.evaluation_budget is not None:
        query.evaluation_budget.consume(len(query.proposals))

    requests = tuple(
        EvaluationRequest(proposal=proposal) for proposal in query.proposals
    )
    successes = tuple(
        EvaluationSuccess.from_scalar_observation(
            observation=Observation.from_objective_value(
                request=request,
                candidate=request.candidate,
                value=query.problem.objective.evaluate(request.candidate),
                direction=query.problem.direction,
            ),
            evaluation_count=1,
        )
        for request in requests
    )
    return EvaluationAttemptBatch(
        attempts=successes,
    )


def record_and_evaluate_query_directly(
    query: ProposalBatchQuery[BoundaryRunnerT, CandidateRunnerT, ObservationPayload],
    evaluated_candidates: list[CandidateRunnerT],
) -> EvaluationAttemptBatch[CandidateRunnerT, ObservationPayload]:
    """Record candidates crossing the evaluator boundary before evaluation."""
    evaluated_candidates.extend(proposal.candidate for proposal in query.proposals)
    return evaluate_query_directly(query)


class IntegerObjective(Objective[int]):
    """One-dimensional integer objective with a known optimum."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float((candidate - 2) ** 2)


class CategoricalObjective(Objective[Color]):
    """Categorical objective with an ordered preference."""

    @override
    def evaluate(self, candidate: Color) -> float:
        if candidate == "blue":
            return 0.0
        if candidate == "green":
            return 1.0
        return 2.0


class FlatCategoricalObjective(Objective[int]):
    """Categorical objective where the incumbent is already globally best."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 0:
            return 0.0
        return 1.0


class ImproveAwayFromZeroObjective(Objective[int]):
    """Categorical objective where any non-zero sampled move improves."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 0:
            return 1.0
        return 0.0


class ArrayIntegerObjective(Objective[tuple[int, ...]]):
    """Small array objective used to exercise recursive leaf traversal."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        return float(abs(candidate[0] - 1) + abs(candidate[1] - 2))


class MixedRecordObjective(Objective[RecordCandidate]):
    """Mixed discrete record objective over integer and categorical fields."""

    @override
    def evaluate(self, candidate: RecordCandidate) -> float:
        level = record_int(candidate, "level")
        color = record_str(candidate, "color")
        color_penalty = 2.0
        if color == "green":
            color_penalty = 1.0
        elif color == "blue":
            color_penalty = 0.0
        return float(abs(level - 2) + color_penalty)


class UnsupportedRecordObjective(Objective[RecordCandidate]):
    """Dummy record objective used to reject unsupported leaf spaces."""

    @override
    def evaluate(self, candidate: RecordCandidate) -> float:
        _ = candidate
        return 0.0


class PairMoveObjective(Objective[tuple[int, ...]]):
    """Two-leaf objective where only a coordinated pair move improves."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        if candidate == (1, 1):
            return 0.0
        if candidate == (0, 0):
            return 1.0
        return 3.0


class PrioritizedPairMoveObjective(Objective[tuple[int, ...]]):
    """Three-leaf objective where only one prioritized pair move improves."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        if candidate == (0, 1, 1):
            return 0.0
        if candidate == (0, 0, 0):
            return 1.0
        return 3.0


class VariableNeighborhoodResetObjective(Objective[tuple[int, ...]]):
    """Three-leaf objective that requires stage reset after a pair improvement."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        if candidate == (1, 1, 1):
            return 0.0
        if candidate == (1, 1, 0):
            return 1.0
        if candidate == (0, 0, 0):
            return 2.0
        return 4.0


class StrictKickAcceptanceObjective(Objective[tuple[int, ...]]):
    """Three-leaf objective where one-leaf kicks improve only back to the incumbent."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        if candidate == (0, 0, 0):
            return 1.0
        if candidate == (1, 1, 0):
            return 0.0
        return 3.0


class ConditionalDiscreteObjective(Objective[ConditionalDiscreteCandidate]):
    """Two-leaf objective used only to trigger dynamic-topology rejection."""

    @override
    def evaluate(self, candidate: ConditionalDiscreteCandidate) -> float:
        if candidate == (1, 1):
            return 0.0
        if candidate == (0, 0):
            return 1.0
        return 3.0


@dataclass(frozen=True)
class ConditionalDiscretePairSpace(
    StructuredSearchSpace[ConditionalDiscreteCandidate, ConditionalDiscreteCandidate],
):
    """Test-only discrete structured space with candidate-conditioned topology."""

    head_space: IntegerSpace
    tail_space: IntegerSpace

    @override
    def normalize(
        self,
        raw_candidate: ConditionalDiscreteCandidate,
    ) -> ConditionalDiscreteCandidate:
        return (
            self.head_space.normalize(raw_candidate[0]),
            self.tail_space.normalize(raw_candidate[1]),
        )

    @override
    def validate(self, candidate: ConditionalDiscreteCandidate) -> None:
        self.head_space.validate(candidate[0])
        self.tail_space.validate(candidate[1])

    @override
    def sample(
        self, random_state: np.random.RandomState
    ) -> ConditionalDiscreteCandidate:
        return (
            self.head_space.sample(random_state),
            self.tail_space.sample(random_state),
        )

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        return (("head",), ("tail",))

    @override
    def has_static_topology(self) -> bool:
        return False

    @override
    def active_leaf_paths(
        self,
        candidate: ConditionalDiscreteCandidate,
    ) -> tuple[LeafPath, ...]:
        self.validate(candidate)
        if candidate[0] > 0:
            return (("head",), ("tail",))
        return (("head",),)

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        if path == ("head",):
            return self.head_space
        if path == ("tail",):
            return self.tail_space
        msg = f"invalid conditional discrete pair path: {path!r}"
        raise TypeError(msg)

    @override
    def leaf_value_at_path(
        self,
        candidate: ConditionalDiscreteCandidate,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        self.validate(candidate)
        if path == ("head",):
            return candidate[0]
        if path == ("tail",):
            return candidate[1]
        msg = f"invalid conditional discrete pair path: {path!r}"
        raise TypeError(msg)

    @override
    def replace_leaf_values(
        self,
        candidate: ConditionalDiscreteCandidate,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> ConditionalDiscreteCandidate:
        self.validate(candidate)
        head_value = candidate[0]
        tail_value = candidate[1]
        if ("head",) in replacements:
            replacement = replacements[("head",)]
            if type(replacement) is not int:
                msg = (
                    "conditional discrete head replacement must be a canonical integer"
                )
                raise TypeError(msg)
            head_value = self.head_space.normalize(replacement)
        if ("tail",) in replacements:
            replacement = replacements[("tail",)]
            if type(replacement) is not int:
                msg = (
                    "conditional discrete tail replacement must be a canonical integer"
                )
                raise TypeError(msg)
            tail_value = self.tail_space.normalize(replacement)
        return (head_value, tail_value)


class StructuredHillClimbKernelTests:
    """Regression tests for discrete structured local-search kernels."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_integer_hill_climber_converges_to_local_optimum(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == 2
        assert outcome.scalar_observation().value == 0.0
        assert outcome.evaluation_count == 6
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.backend == "structured.local_search"
        assert outcome.kernel_diagnostics.method == "leafwise_first_improvement"
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED
        assert outcome.refinement is not None
        assert outcome.refinement.source_candidate == 5
        assert outcome.refinement.refined_candidate == 2
        assert outcome.refinement.changed_leaf_paths == ((),)

    def test_hill_climber_reserves_budget_for_later_batch_proposals(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        evaluation_budget = EvaluationBudget(2)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(
                Proposal(candidate=5, proposal_id="p-1"),
                Proposal(candidate=5, proposal_id="p-2"),
            ),
            execution_resources=self.make_execution_resources(),
            evaluation_budget=evaluation_budget,
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        assert evaluation_budget.remaining == 0
        assert tuple(outcome.evaluation_count for outcome in outcomes) == (1, 1)
        assert tuple(
            outcome.scalar_observation().candidate for outcome in outcomes
        ) == (5, 5)
        assert all(
            outcome.kernel_diagnostics is not None
            and outcome.kernel_diagnostics.status == KernelStatus.STOPPED
            and outcome.kernel_diagnostics.message
            == "evaluation budget exhausted before local convergence"
            for outcome in outcomes
        )

    def test_failed_hill_climb_trial_is_preserved_and_skipped(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        def fail_neighbor_four(
            local_query: ProposalBatchQuery[int, int, ObservationPayload],
        ) -> EvaluationAttemptBatch[int, ObservationPayload]:
            proposal = local_query.proposals[0]
            request: EvaluationRequest[int] = EvaluationRequest(proposal=proposal)
            if proposal.candidate == 4:
                failure = EvaluationFailure[int](
                    request=request,
                    exception=EvaluationExceptionSnapshot.from_exception(
                        ValueError("bad structured trial")
                    ),
                )
                return EvaluationAttemptBatch(
                    attempts=(failure,),
                )
            return evaluate_query_directly(local_query)

        attempts = kernel.run(query, fail_neighbor_four)

        assert attempts.success_indices == (0,)
        assert attempts.failure_indices == ()
        assert attempts.successes[0].scalar_observation().candidate == 5
        assert attempts.successes[0].kernel_diagnostics is not None
        assert attempts.successes[0].kernel_diagnostics.failed_attempt_count == 1
        assert attempts.successes[0].kernel_diagnostics.failed_evaluation_count == 1
        assert attempts.successes[0].refinement is None
        assert attempts.evaluation_count == 2

    def test_study_run_accepts_successful_hill_climb_with_inner_failure(self) -> None:
        class FailsAtFourObjective(Objective[int]):
            """Objective that records a user-code failure for one inner trial."""

            @override
            def evaluate(self, candidate: int) -> float:
                if candidate == 4:
                    raise ValueError("bad structured trial")
                return float((candidate - 2) ** 2)

        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=FailsAtFourObjective(),
        )
        study = Study(
            problem=problem,
            run_method=BatchQueueOptimizer(
                [(Proposal(candidate=5, proposal_id="p-1"),)]
            ),
            evaluator=SequentialEvaluator(),
            kernel=StructuredHillClimbKernel[int, int](max_steps=8),
        )

        report, _state = study.run(max_evaluations=2)

        assert len(report.successes) == 1
        assert report.failures == ()
        assert report.evaluation_count == 2
        diagnostics = report.successes[0].kernel_diagnostics
        assert diagnostics is not None
        assert diagnostics.failed_attempt_count == 1
        assert diagnostics.failed_evaluation_count == 1

    def test_study_run_keeps_multi_proposal_alignment_with_inner_failure(
        self,
    ) -> None:
        class FailsAtFourObjective(Objective[int]):
            """Objective that records a user-code failure for one inner trial."""

            @override
            def evaluate(self, candidate: int) -> float:
                if candidate == 4:
                    raise ValueError("bad structured trial")
                return float((candidate - 2) ** 2)

        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=FailsAtFourObjective(),
        )
        study = Study(
            problem=problem,
            run_method=BatchQueueOptimizer(
                [
                    (
                        Proposal(candidate=5, proposal_id="p-1"),
                        Proposal(candidate=3, proposal_id="p-2"),
                    )
                ]
            ),
            evaluator=SequentialEvaluator(),
            kernel=StructuredHillClimbKernel[int, int](max_steps=8),
        )

        report, _state = study.run(max_evaluations=5, batch_size=2)

        assert len(report.successes) == 2
        assert report.failures == ()
        assert report.successes[0].proposal_id == "p-1"
        assert report.successes[1].proposal_id == "p-2"
        first_diagnostics = report.successes[0].kernel_diagnostics
        second_diagnostics = report.successes[1].kernel_diagnostics
        assert first_diagnostics is not None
        assert first_diagnostics.failed_attempt_count == 1
        assert first_diagnostics.failed_evaluation_count == 1
        assert second_diagnostics is not None
        assert second_diagnostics.failed_attempt_count == 0
        assert second_diagnostics.failed_evaluation_count == 0
        assert report.evaluation_count == sum(
            success.evaluation_count for success in report.successes
        )

    def test_initial_hill_climb_failure_returns_failure_only_attempt(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        def fail_every_attempt(
            local_query: ProposalBatchQuery[int, int, ObservationPayload],
        ) -> EvaluationAttemptBatch[int, ObservationPayload]:
            proposal = local_query.proposals[0]
            request: EvaluationRequest[int] = EvaluationRequest(proposal=proposal)
            failure = EvaluationFailure[int](
                request=request,
                exception=EvaluationExceptionSnapshot.from_exception(
                    RuntimeError("initial structured failure")
                ),
            )
            return EvaluationAttemptBatch(
                attempts=(failure,),
            )

        attempts = kernel.run(query, fail_every_attempt)

        assert attempts.successes == ()
        assert attempts.success_indices == ()
        assert attempts.failure_indices == (0,)
        assert attempts.failures[0].candidate == 5
        assert attempts.failures[0].proposal_id == "p-1"
        assert attempts.failures[0].exception.message == "initial structured failure"
        assert attempts.evaluation_count == 1

    def test_mixed_proposal_hill_climb_rebases_failure_and_success_indices(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=1)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(
                Proposal(candidate=5, proposal_id="p-fail"),
                Proposal(candidate=5, proposal_id="p-ok"),
            ),
            execution_resources=self.make_execution_resources(),
        )

        def fail_first_proposal(
            local_query: ProposalBatchQuery[int, int, ObservationPayload],
        ) -> EvaluationAttemptBatch[int, ObservationPayload]:
            proposal = local_query.proposals[0]
            request: EvaluationRequest[int] = EvaluationRequest(proposal=proposal)
            if proposal.proposal_id == "p-fail":
                failure = EvaluationFailure[int](
                    request=request,
                    exception=EvaluationExceptionSnapshot.from_exception(
                        ValueError("first proposal failed")
                    ),
                )
                return EvaluationAttemptBatch(
                    attempts=(failure,),
                )
            return evaluate_query_directly(local_query)

        attempts = kernel.run(query, fail_first_proposal)

        assert attempts.failure_indices == (0,)
        assert attempts.success_indices == (1,)
        assert attempts.failures[0].proposal_id == "p-fail"
        assert attempts.successes[0].scalar_observation().proposal.proposal_id == "p-ok"
        assert attempts.successes[0].scalar_observation().candidate == 4
        assert attempts.evaluation_count == 3

    def test_hill_climb_rejects_multi_slot_runner_attempt_for_single_proposal(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=1)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        def malformed_runner(
            local_query: ProposalBatchQuery[int, int, ObservationPayload],
        ) -> EvaluationAttemptBatch[int, ObservationPayload]:
            proposal = local_query.proposals[0]
            return evaluate_query_directly(
                ProposalBatchQuery(
                    problem=local_query.problem,
                    proposals=(proposal, Proposal(candidate=proposal.candidate)),
                    execution_resources=local_query.execution_resources,
                )
            )

        with pytest.raises(
            ValueError,
            match="kernel runner must return exactly one attempt for one proposal",
        ):
            _ = kernel.run(query, malformed_runner)

    def test_disabled_hill_climb_preserves_original_proposal_failure(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(ProposalLocalSearchContext(enabled=False),),
        )

        def fail_original_proposal(
            local_query: ProposalBatchQuery[int, int, ObservationPayload],
        ) -> EvaluationAttemptBatch[int, ObservationPayload]:
            proposal = local_query.proposals[0]
            request: EvaluationRequest[int] = EvaluationRequest(proposal=proposal)
            failure = EvaluationFailure[int](
                request=request,
                exception=EvaluationExceptionSnapshot.from_exception(
                    ValueError("disabled structured failure")
                ),
            )
            return EvaluationAttemptBatch(
                attempts=(failure,),
            )

        attempts = kernel.run(query, fail_original_proposal)

        assert attempts.successes == ()
        assert attempts.success_indices == ()
        assert attempts.failure_indices == (0,)
        assert attempts.failures[0].candidate == 5
        assert attempts.failures[0].proposal_id == "p-1"
        assert attempts.failures[0].exception.message == "disabled structured failure"
        assert attempts.evaluation_count == 1

    def test_categorical_hill_climber_moves_through_declared_alternatives(self) -> None:
        color_space: CategoricalSpace[Color] = CategoricalSpace(
            ("red", "green", "blue")
        )
        initial_candidate: Color = "red"
        problem: Problem[Color, Color, ObservationPayload] = Problem(
            space=color_space,
            objective=CategoricalObjective(),
        )
        kernel = StructuredHillClimbKernel[Color, Color](max_steps=4)
        query: ProposalBatchQuery[Color, Color, ObservationPayload] = (
            ProposalBatchQuery(
                problem=problem,
                proposals=(
                    Proposal[Color](candidate=initial_candidate, proposal_id="p-1"),
                ),
                execution_resources=self.make_execution_resources(),
            )
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == "blue"
        assert outcome.scalar_observation().value == 0.0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED
        assert outcome.refinement is not None
        assert outcome.refinement.source_candidate == "red"
        assert outcome.refinement.refined_candidate == "blue"
        assert outcome.refinement.changed_leaf_paths == ((),)

    def test_hill_climber_recurses_over_array_leaves(self) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 3), length=2),
            objective=ArrayIntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[Sequence[int], tuple[int, ...]](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == (1, 2)
        assert outcome.scalar_observation().value == 0.0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED
        assert outcome.refinement is not None
        assert outcome.refinement.source_candidate == (0, 0)
        assert outcome.refinement.refined_candidate == (1, 2)
        assert outcome.refinement.changed_leaf_paths == ((0,), (1,))

    def test_hill_climber_stops_when_step_budget_is_exhausted(self) -> None:
        space = RecordSpace(
            level=IntegerSpace(0, 5),
            color=CategoricalSpace(("red", "green", "blue")),
        )
        problem = Problem(
            space=space,
            objective=MixedRecordObjective(),
        )
        initial_candidate = space.normalize({"level": 0, "color": "red"})
        kernel = StructuredHillClimbKernel[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ](max_steps=1)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=initial_candidate, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert record_int(outcome.scalar_observation().candidate, "level") == 1
        assert record_str(outcome.scalar_observation().candidate, "color") == "red"
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED

    def test_context_can_disable_local_search(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(ProposalLocalSearchContext(enabled=False),),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == 5
        assert outcome.scalar_observation().value == 9.0
        assert outcome.evaluation_count == 1
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert (
            outcome.kernel_diagnostics.message
            == "local search disabled by run-method context"
        )
        assert outcome.refinement is None

    def test_context_can_override_step_budget(self) -> None:
        space = RecordSpace(
            level=IntegerSpace(0, 5),
            color=CategoricalSpace(("red", "green", "blue")),
        )
        problem = Problem(
            space=space,
            objective=MixedRecordObjective(),
        )
        initial_candidate = space.normalize({"level": 0, "color": "red"})
        kernel = StructuredHillClimbKernel[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=initial_candidate, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(ProposalLocalSearchContext(local_budget=1),),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert record_int(outcome.scalar_observation().candidate, "level") == 1
        assert record_str(outcome.scalar_observation().candidate, "color") == "red"
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED

    def test_context_can_prioritize_leaf_order(self) -> None:
        space = RecordSpace(
            level=IntegerSpace(0, 5),
            color=CategoricalSpace(("red", "green", "blue")),
        )
        problem = Problem(
            space=space,
            objective=MixedRecordObjective(),
        )
        initial_candidate = space.normalize({"level": 0, "color": "red"})
        kernel = StructuredHillClimbKernel[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ](max_steps=1)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=initial_candidate, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(prioritized_leaf_paths=(("color",),)),
            ),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert record_int(outcome.scalar_observation().candidate, "level") == 0
        assert record_str(outcome.scalar_observation().candidate, "color") == "green"
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED

    def test_rejects_non_local_search_kernel_hint(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(DummyProposalKernelHint(),),
        )

        with pytest.raises(TypeError, match="ProposalLocalSearchContext hints"):
            _ = kernel.run(query, evaluate_query_directly)

    def test_hill_climber_rejects_dynamic_topology_space(self) -> None:
        problem = Problem(
            space=ConditionalDiscretePairSpace(
                head_space=IntegerSpace(0, 3),
                tail_space=IntegerSpace(0, 3),
            ),
            objective=ConditionalDiscreteObjective(),
        )
        kernel = StructuredHillClimbKernel[
            ConditionalDiscreteCandidate,
            ConditionalDiscreteCandidate,
        ](max_steps=4)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        with pytest.raises(TypeError, match="static topology"):
            _ = kernel.run(query, evaluate_query_directly)

    def test_rejects_real_valued_leaves(self) -> None:
        problem = Problem(
            space=RecordSpace(
                level=IntegerSpace(0, 5),
                weight=RealSpace(0.0, 5.0),
            ),
            objective=UnsupportedRecordObjective(),
        )
        kernel = StructuredHillClimbKernel[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ](max_steps=4)
        candidate = problem.space.normalize({"level": 1, "weight": 2.0})
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=candidate, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        with pytest.raises(TypeError, match="IntegerSpace or CategoricalSpace"):
            _ = kernel.run(query, evaluate_query_directly)


class StructuredScheduledLocalSearchKernelTests:
    """Regression tests for staged structured local-search kernels."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_pair_move_stage_can_escape_single_leaf_trap(self) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=2),
            objective=PairMoveObjective(),
        )
        kernel = StructuredScheduledLocalSearchKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=1,
            pair_move_leaf_limit=2,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == (1, 1)
        assert outcome.scalar_observation().value == 0.0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "scheduled_single_then_pair"
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED

    def test_prioritized_leaf_paths_can_steer_pair_move_stage(self) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=3),
            objective=PrioritizedPairMoveObjective(),
        )
        kernel = StructuredScheduledLocalSearchKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=1,
            pair_move_leaf_limit=2,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(
                    prioritized_leaf_paths=((1,), (2,)),
                ),
            ),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == (0, 1, 1)
        assert outcome.scalar_observation().value == 0.0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED


class StructuredStochasticNeighborhoodKernelTests:
    """Regression tests for bounded stochastic structured local-search."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_stochastic_kernel_stops_after_sampled_neighborhood_without_improvement(
        self,
    ) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(6))),
            objective=FlatCategoricalObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=4,
            max_neighbors_per_step=2,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == 0
        assert outcome.scalar_observation().value == 0.0
        assert outcome.evaluation_count == 3
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "sampled_leafwise_first_improvement"
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert (
            outcome.kernel_diagnostics.message
            == "no improving move found in the sampled discrete neighborhood"
        )
        assert outcome.refinement is None

    def test_stochastic_kernel_converges_when_sampling_covers_full_neighborhood(
        self,
    ) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(3))),
            objective=FlatCategoricalObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=4,
            max_neighbors_per_step=4,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.evaluation_count == 3
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED
        assert (
            outcome.kernel_diagnostics.message
            == "no improving move found in the full discrete neighborhood"
        )
        assert outcome.refinement is None

    def test_stochastic_kernel_can_cap_categorical_neighbors_per_leaf(self) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(8))),
            objective=FlatCategoricalObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=4,
            max_neighbors_per_step=16,
            max_categorical_neighbors_per_leaf=3,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.evaluation_count == 4
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert (
            outcome.kernel_diagnostics.message
            == "no improving move found in the sampled discrete neighborhood"
        )

    def test_stochastic_kernel_can_take_one_sampled_improving_move(self) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(6))),
            objective=ImproveAwayFromZeroObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=1,
            max_neighbors_per_step=1,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate != 0
        assert outcome.scalar_observation().value == 0.0
        assert outcome.evaluation_count == 2
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert (
            outcome.kernel_diagnostics.message
            == "max_steps reached before stochastic local-search termination"
        )
        assert outcome.refinement is not None
        assert outcome.refinement.source_candidate == 0
        assert (
            outcome.refinement.refined_candidate
            == outcome.scalar_observation().candidate
        )
        assert outcome.refinement.changed_leaf_paths == ((),)

    def test_stochastic_kernel_advances_rng_stream_between_runs(self) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(10))),
            objective=ImproveAwayFromZeroObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=1,
            max_neighbors_per_step=1,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        first_evaluated_candidates: list[int] = []
        second_evaluated_candidates: list[int] = []
        _ = kernel.run(
            query,
            lambda query: record_and_evaluate_query_directly(
                query,
                first_evaluated_candidates,
            ),
        )
        _ = kernel.run(
            query,
            lambda query: record_and_evaluate_query_directly(
                query,
                second_evaluated_candidates,
            ),
        )

        assert tuple(first_evaluated_candidates) != tuple(second_evaluated_candidates)

    def test_stochastic_kernel_uses_context_rng_snapshot_between_runs(self) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(10))),
            objective=ImproveAwayFromZeroObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=1,
            max_neighbors_per_step=1,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(
                    random_state_snapshot=RandomStateSnapshot.from_seed(4),
                ),
            ),
        )

        first_evaluated_candidates: list[int] = []
        second_evaluated_candidates: list[int] = []
        _ = kernel.run(
            query,
            lambda query: record_and_evaluate_query_directly(
                query,
                first_evaluated_candidates,
            ),
        )
        _ = kernel.run(
            query,
            lambda query: record_and_evaluate_query_directly(
                query,
                second_evaluated_candidates,
            ),
        )

        assert tuple(first_evaluated_candidates) == tuple(second_evaluated_candidates)

    def test_stochastic_kernel_uses_aligned_context_rng_snapshot_per_proposal(
        self,
    ) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(10))),
            objective=ImproveAwayFromZeroObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=1,
            max_neighbors_per_step=1,
            random_state=0,
        )
        first_snapshot = RandomStateSnapshot.from_seed(4)
        second_snapshot = RandomStateSnapshot.from_seed(5)
        forward_query = ProposalBatchQuery(
            problem=problem,
            proposals=(
                Proposal(candidate=0, proposal_id="p-1"),
                Proposal(candidate=0, proposal_id="p-2"),
            ),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(random_state_snapshot=first_snapshot),
                ProposalLocalSearchContext(random_state_snapshot=second_snapshot),
            ),
        )
        reversed_query = ProposalBatchQuery(
            problem=problem,
            proposals=(
                Proposal(candidate=0, proposal_id="p-1"),
                Proposal(candidate=0, proposal_id="p-2"),
            ),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(random_state_snapshot=second_snapshot),
                ProposalLocalSearchContext(random_state_snapshot=first_snapshot),
            ),
        )

        forward_candidates = tuple(
            outcome.scalar_observation().candidate
            for outcome in kernel.run(
                forward_query,
                evaluate_query_directly,
            ).successes
        )
        reversed_candidates = tuple(
            outcome.scalar_observation().candidate
            for outcome in kernel.run(
                reversed_query,
                evaluate_query_directly,
            ).successes
        )

        assert forward_candidates[0] != forward_candidates[1]
        assert reversed_candidates == (
            forward_candidates[1],
            forward_candidates[0],
        )

    def test_stochastic_kernel_disabled_context_ignores_rng_snapshot(self) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(10))),
            objective=ImproveAwayFromZeroObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=1,
            max_neighbors_per_step=1,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(
                    enabled=False,
                    random_state_snapshot=RandomStateSnapshot.from_seed(4),
                ),
            ),
        )
        evaluated_candidates: list[int] = []

        outcomes = kernel.run(
            query,
            lambda query: record_and_evaluate_query_directly(
                query,
                evaluated_candidates,
            ),
        ).successes

        assert tuple(evaluated_candidates) == (0,)
        assert outcomes[0].scalar_observation().candidate == 0
        assert outcomes[0].refinement is None
        assert outcomes[0].kernel_diagnostics is not None
        assert outcomes[0].kernel_diagnostics.message == (
            "local search disabled by run-method context"
        )

    def test_stochastic_kernel_rejects_dynamic_topology_space(self) -> None:
        problem = Problem(
            space=ConditionalDiscretePairSpace(
                head_space=IntegerSpace(0, 3),
                tail_space=IntegerSpace(0, 3),
            ),
            objective=ConditionalDiscreteObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[
            ConditionalDiscreteCandidate,
            ConditionalDiscreteCandidate,
        ](
            max_steps=4,
            max_neighbors_per_step=2,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        with pytest.raises(TypeError, match="static topology"):
            _ = kernel.run(query, evaluate_query_directly)

    def test_stochastic_kernel_rejects_invalid_sampling_budgets(self) -> None:
        with pytest.raises(ValueError, match="max_neighbors_per_step"):
            _ = StructuredStochasticNeighborhoodKernel[int, int](
                max_neighbors_per_step=0,
            )

        with pytest.raises(ValueError, match="max_categorical_neighbors_per_leaf"):
            _ = StructuredStochasticNeighborhoodKernel[int, int](
                max_categorical_neighbors_per_leaf=0,
            )


class StructuredVariableNeighborhoodKernelTests:
    """Regression tests for structured variable-neighborhood wrappers."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_variable_neighborhood_kernel_resets_to_first_stage_after_improvement(
        self,
    ) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=3),
            objective=VariableNeighborhoodResetObjective(),
        )
        kernel = StructuredVariableNeighborhoodKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=2,
            stages=(
                StructuredVariableNeighborhoodStage.leafwise_first_improvement(),
                StructuredVariableNeighborhoodStage.scheduled_single_then_pair(
                    pair_move_leaf_limit=2,
                ),
            ),
        )
        query: ProposalBatchQuery[
            Sequence[int], tuple[int, ...], ObservationPayload
        ] = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == (1, 1, 1)
        assert outcome.scalar_observation().value == 0.0
        assert outcome.evaluation_count == 11
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "variable_neighborhood_search"
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert (
            outcome.kernel_diagnostics.message
            == "max_steps reached before variable-neighborhood termination"
        )
        assert outcome.refinement is not None
        assert outcome.refinement.source_candidate == (0, 0, 0)
        assert outcome.refinement.refined_candidate == (1, 1, 1)
        assert outcome.refinement.changed_leaf_paths == ((0,), (1,), (2,))

    def test_variable_neighborhood_kernel_uses_sampled_stage_terminal_status(
        self,
    ) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(6))),
            objective=FlatCategoricalObjective(),
        )
        kernel = StructuredVariableNeighborhoodKernel[int, int](
            max_steps=4,
            stages=(
                StructuredVariableNeighborhoodStage.sampled_leafwise_first_improvement(
                    max_neighbors_per_step=2,
                ),
            ),
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == 0
        assert outcome.evaluation_count == 3
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == (
            "no improving move found in the sampled variable neighborhood "
            "after exhausting the configured variable-neighborhood stages"
        )
        assert outcome.refinement is None

    def test_variable_neighborhood_kernel_advances_rng_stream_between_runs(
        self,
    ) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(10))),
            objective=ImproveAwayFromZeroObjective(),
        )
        kernel = StructuredVariableNeighborhoodKernel[int, int](
            max_steps=1,
            stages=(
                StructuredVariableNeighborhoodStage.sampled_leafwise_first_improvement(
                    max_neighbors_per_step=1,
                ),
            ),
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        first_evaluated_candidates: list[int] = []
        second_evaluated_candidates: list[int] = []
        _ = kernel.run(
            query,
            lambda query: record_and_evaluate_query_directly(
                query,
                first_evaluated_candidates,
            ),
        )
        _ = kernel.run(
            query,
            lambda query: record_and_evaluate_query_directly(
                query,
                second_evaluated_candidates,
            ),
        )

        assert tuple(first_evaluated_candidates) != tuple(second_evaluated_candidates)

    def test_variable_neighborhood_kernel_rejects_invalid_stage_metadata(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="stages must contain at least one"):
            _ = StructuredVariableNeighborhoodKernel[int, int](stages=())

        with pytest.raises(ValueError, match="must not set max_neighbors_per_step"):
            _ = StructuredVariableNeighborhoodStage(
                kind="leafwise_first_improvement",
                max_neighbors_per_step=1,
            )

        with pytest.raises(ValueError, match="pair_move_leaf_limit must be positive"):
            _ = StructuredVariableNeighborhoodStage.scheduled_single_then_pair(
                pair_move_leaf_limit=0,
            )


class StructuredIteratedLocalSearchKernelTests:
    """Regression tests for structured iterated local-search wrappers."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_iterated_local_search_kernel_can_escape_pair_move_trap_via_kick(
        self,
    ) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=2),
            objective=PairMoveObjective(),
        )
        kernel = StructuredIteratedLocalSearchKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=4,
            max_kicks=1,
            kick_policy=StructuredKickPolicy(kick_leaf_count=2),
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == (1, 1)
        assert outcome.scalar_observation().value == 0.0
        assert outcome.evaluation_count == 6
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "iterated_local_search"
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert (
            outcome.kernel_diagnostics.message
            == "max_kicks reached before iterated local-search termination"
        )
        assert outcome.refinement is not None
        assert outcome.refinement.source_candidate == (0, 0)
        assert outcome.refinement.refined_candidate == (1, 1)
        assert outcome.refinement.changed_leaf_paths == ((0,), (1,))

    def test_iterated_local_search_kernel_requires_strict_improvement_to_accept(
        self,
    ) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=3),
            objective=StrictKickAcceptanceObjective(),
        )
        kernel = StructuredIteratedLocalSearchKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=4,
            max_kicks=1,
            kick_policy=StructuredKickPolicy(kick_leaf_count=1),
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly).successes

        outcome = outcomes[0]
        assert outcome.scalar_observation().candidate == (0, 0, 0)
        assert outcome.scalar_observation().value == 1.0
        assert outcome.evaluation_count == 11
        assert outcome.refinement is None
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert (
            outcome.kernel_diagnostics.message
            == "max_kicks reached before iterated local-search termination"
        )

    def test_iterated_local_search_kernel_advances_kick_rng_stream_between_runs(
        self,
    ) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=3),
            objective=StrictKickAcceptanceObjective(),
        )
        kernel = StructuredIteratedLocalSearchKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=4,
            max_kicks=1,
            kick_policy=StructuredKickPolicy(kick_leaf_count=1),
            random_state=4,
        )
        query: ProposalBatchQuery[
            Sequence[int], tuple[int, ...], ObservationPayload
        ] = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )
        evaluated_candidate_runs: list[tuple[tuple[int, ...], ...]] = []

        for _ in range(3):
            evaluated_candidates: list[tuple[int, ...]] = []
            _ = kernel.run(
                query,
                lambda query: record_and_evaluate_query_directly(
                    query,
                    evaluated_candidates,
                ),
            )
            evaluated_candidate_runs.append(tuple(evaluated_candidates))

        assert len(frozenset(evaluated_candidate_runs)) > 1

    def test_iterated_local_search_kernel_rejects_invalid_kick_metadata(self) -> None:
        with pytest.raises(ValueError, match="max_kicks must be positive"):
            _ = StructuredIteratedLocalSearchKernel[int, int](max_kicks=0)

        with pytest.raises(ValueError, match="kick_leaf_count must be positive"):
            _ = StructuredKickPolicy(kick_leaf_count=0)

        with pytest.raises(
            ValueError, match="max_categorical_alternatives_per_leaf must be positive"
        ):
            _ = StructuredKickPolicy(max_categorical_alternatives_per_leaf=0)


def record_int(candidate: RecordCandidate, field_name: str) -> int:
    """Return one canonical integer-valued record field."""
    value = candidate[field_name]
    if type(value) is not int:
        msg = f"record field {field_name!r} must be a canonical integer"
        raise TypeError(msg)
    return value


def record_str(candidate: RecordCandidate, field_name: str) -> str:
    """Return one canonical string-valued record field."""
    value = candidate[field_name]
    if type(value) is not str:
        msg = f"record field {field_name!r} must be a canonical string"
        raise TypeError(msg)
    return value
