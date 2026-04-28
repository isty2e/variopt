"""Thin public facade for optimization runs."""

from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..artifacts import EvaluationRequest, Proposal, RunReport, RunResult
from ..evaluators.base import Evaluator
from ..execution import (
    STALE_ASYNC_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    ExecutionModel,
)
from ..kernel import DirectKernel, Kernel, ProposalBatchQuery
from ..methods import RunMethod
from ..outcomes import EvaluationOutcome
from ..problem import Problem
from ..typevars import CandidateT, RunMethodStateT
from .common import StudyEvaluationRecordT
from .exact_async.artifacts import (
    StudyExactAsyncStepResumeHandle,
)
from .exact_async.orchestration import (
    open_exact_async_step_session as open_exact_async_step_session_for_study,
)
from .exact_async.orchestration import (
    resume_exact_async_step_session as resume_exact_async_step_session_for_study,
)
from .exact_async.session import StudyExactAsyncStepSession
from .execution import optimize as optimize_study
from .execution import run as run_study
from .execution import step as step_study
from .stale_async import run_stale_async

BoundaryT = TypeVar("BoundaryT")


@dataclass(frozen=True, slots=True, init=False)
class Study(FrozenGenericSlotsCompat,
    Generic[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
):
    """User-facing facade for running an optimization study.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, StudyEvaluationRecordT]
        Problem that defines the search space and evaluation semantics.
    run_method : RunMethod[RunMethodStateT, Proposal[CandidateT], StudyEvaluationRecordT]
        Stateful optimizer that owns cross-step search memory.
    evaluator : Evaluator[Problem[BoundaryT, CandidateT, StudyEvaluationRecordT], EvaluationRequest[CandidateT], EvaluationOutcome[CandidateT, StudyEvaluationRecordT]]
        Execution backend that turns requests into evaluation outcomes.
    kernel : Kernel[ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT], tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]] | None, optional
        Optional within-step refinement kernel. Defaults to
        :class:`~variopt.kernel.DirectKernel`.

    Notes
    -----
    ``Study`` is the main user entry point for running optimizers. It binds the
    problem, optimizer, evaluator, and optional kernel into a single execution
    object, then exposes sync, exact-async, and stale-async execution methods.
    """

    problem: Problem[BoundaryT, CandidateT, StudyEvaluationRecordT]
    run_method: RunMethod[
        RunMethodStateT,
        Proposal[CandidateT],
        StudyEvaluationRecordT,
    ]
    evaluator: Evaluator[
        Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
    ]
    kernel: Kernel[
        ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT],
        tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...],
    ]

    def __init__(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        run_method: RunMethod[
            RunMethodStateT,
            Proposal[CandidateT],
            StudyEvaluationRecordT,
        ],
        evaluator: Evaluator[
            Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
            EvaluationRequest[CandidateT],
            EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
        ],
        kernel: Kernel[
            ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT],
            tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...],
        ]
        | None = None,
    ) -> None:
        """Create a study over one problem and execution stack.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, StudyEvaluationRecordT]
            Problem that defines candidate validity and evaluation semantics.
        run_method : RunMethod[RunMethodStateT, Proposal[CandidateT], StudyEvaluationRecordT]
            Stateful optimizer that emits proposals and assimilates records.
        evaluator : Evaluator[Problem[BoundaryT, CandidateT, StudyEvaluationRecordT], EvaluationRequest[CandidateT], EvaluationOutcome[CandidateT, StudyEvaluationRecordT]]
            Execution backend that evaluates requests.
        kernel : Kernel[ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT], tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]] | None, optional
            Optional within-step refinement kernel.
        """
        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "run_method", run_method)
        object.__setattr__(self, "evaluator", evaluator)
        object.__setattr__(
            self,
            "kernel",
            DirectKernel() if kernel is None else kernel,
        )

    def open_exact_async_step_session(
        self,
        state: RunMethodStateT,
        batch_size: int = 1,
    ) -> StudyExactAsyncStepSession[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ]:
        """Open a resumable exact-async step session.

        Parameters
        ----------
        state : RunMethodStateT
            Run-method state to advance.
        batch_size : int, default=1
            Number of proposals to ask for in the exact-async step.

        Returns
        -------
        StudyExactAsyncStepSession[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
            Open session that can be polled, suspended, resumed, and eventually
            committed back into the run method.
        """
        return open_exact_async_step_session_for_study(
            self,
            state,
            batch_size=batch_size,
        )

    def resume_exact_async_step_session(
        self,
        handle: StudyExactAsyncStepResumeHandle[
            CandidateT,
            RunMethodStateT,
            StudyEvaluationRecordT,
        ],
    ) -> StudyExactAsyncStepSession[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ]:
        """Resume a suspended exact-async step session.

        Parameters
        ----------
        handle : StudyExactAsyncStepResumeHandle[CandidateT, RunMethodStateT, StudyEvaluationRecordT]
            Resume handle produced when an exact-async session was suspended.

        Returns
        -------
        StudyExactAsyncStepSession[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
            Resumed session ready for further polling or completion.
        """
        return resume_exact_async_step_session_for_study(self, handle)

    def create_run_method_state(self) -> RunMethodStateT:
        """Create the canonical initial optimizer state.

        Returns
        -------
        RunMethodStateT
            Initial state produced by the configured run method.
        """
        return self.run_method.create_initial_state()

    def step(
        self,
        state: RunMethodStateT,
        batch_size: int = 1,
        *,
        execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
    ) -> tuple[tuple[StudyEvaluationRecordT, ...], RunMethodStateT]:
        """Run one ask/evaluate/tell step.

        Parameters
        ----------
        state : RunMethodStateT
            Run-method state to advance.
        batch_size : int, default=1
            Number of proposals to ask for in the step.
        execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
            Execution-model contract that controls completion and assimilation
            order.

        Returns
        -------
        tuple[tuple[StudyEvaluationRecordT, ...], RunMethodStateT]
            Step records followed by the next run-method state.
        """
        return step_study(
            self,
            state,
            batch_size=batch_size,
            execution_model=execution_model,
        )

    def run(
        self,
        max_evaluations: int,
        batch_size: int = 1,
        *,
        execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
        count_evaluation_cost: bool = False,
        initial_state: RunMethodStateT | None = None,
    ) -> tuple[RunReport[CandidateT, StudyEvaluationRecordT], RunMethodStateT]:
        """Run the study until the evaluation budget is exhausted.

        Parameters
        ----------
        max_evaluations : int
            Evaluation budget for the run.
        batch_size : int, default=1
            Number of proposals to ask for per logical step.
        execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
            Execution-model contract that controls completion and assimilation
            order.
        count_evaluation_cost : bool, default=False
            Whether to consume budget using each outcome's logical evaluation
            cost instead of simple record count.
        initial_state : RunMethodStateT | None, optional
            Optional run-method state to start from.

        Returns
        -------
        tuple[RunReport[CandidateT, StudyEvaluationRecordT], RunMethodStateT]
            Terminal run report followed by the final run-method state.
        """
        if execution_model == STALE_ASYNC_EXECUTION_MODEL:
            return run_stale_async(
                self,
                max_evaluations=max_evaluations,
                batch_size=batch_size,
                count_evaluation_cost=count_evaluation_cost,
                initial_state=initial_state,
            )

        return run_study(
            self,
            max_evaluations=max_evaluations,
            batch_size=batch_size,
            execution_model=execution_model,
            count_evaluation_cost=count_evaluation_cost,
            initial_state=initial_state,
        )

    def optimize(
        self,
        max_evaluations: int,
        batch_size: int = 1,
        *,
        execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
        count_evaluation_cost: bool = False,
        initial_state: RunMethodStateT | None = None,
    ) -> tuple[RunResult[CandidateT], RunMethodStateT]:
        """Run the study and materialize the scalar best-result summary.

        Parameters
        ----------
        max_evaluations : int
            Evaluation budget for the run.
        batch_size : int, default=1
            Number of proposals to ask for per logical step.
        execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
            Execution-model contract that controls completion and assimilation
            order.
        count_evaluation_cost : bool, default=False
            Whether to consume budget using each outcome's logical evaluation
            cost instead of simple record count.
        initial_state : RunMethodStateT | None, optional
            Optional run-method state to start from.

        Returns
        -------
        tuple[RunResult[CandidateT], RunMethodStateT]
            Terminal scalar run summary followed by the final run-method state.
        """
        return optimize_study(
            self,
            max_evaluations=max_evaluations,
            batch_size=batch_size,
            execution_model=execution_model,
            count_evaluation_cost=count_evaluation_cost,
            initial_state=initial_state,
        )
