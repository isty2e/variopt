"""Optional MPI-backed evaluator for synchronous batch execution."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

from typing_extensions import TypeVar as DefaultTypeVar
from typing_extensions import override

from ..artifacts import (
    EvaluationRequest,
    ObjectiveVectorPayload,
    ObservationPayload,
)
from ..artifacts.records import RequestAlignedEvaluationRecord
from ..evaluation_pipeline import evaluate_request_attempt, evaluate_request_outcome
from ..execution import ExecutionResources, NestedParallelismPolicy
from ..outcomes import EvaluationAttemptBatch, EvaluationOutcome
from ..problem import Problem
from ..typevars import CandidateT
from .base import Evaluator

BoundaryT = TypeVar("BoundaryT")
CandidateExecutorT = TypeVar("CandidateExecutorT", covariant=True)
MpiFutureResultT = TypeVar("MpiFutureResultT", covariant=True)
MpiSubmitResultT = TypeVar("MpiSubmitResultT")
MpiExecutorRecordT = TypeVar(
    "MpiExecutorRecordT",
    bound=RequestAlignedEvaluationRecord,
    covariant=True,
)
MpiEvaluationPayload: TypeAlias = (
    RequestAlignedEvaluationRecord | ObservationPayload | ObjectiveVectorPayload
)
MpiEvaluationPayloadT = DefaultTypeVar(
    "MpiEvaluationPayloadT",
    bound=MpiEvaluationPayload,
    default=ObservationPayload,
)


class MpiFuture(Protocol, Generic[MpiFutureResultT]):
    """Typed view of one MPI-backed future.

    Notes
    -----
    The MPI evaluator only relies on a small future surface: ordered result
    retrieval for already submitted work items.
    """

    def result(
        self,
    ) -> tuple[int, MpiFutureResultT]:
        """Return one completed result or raise the worker failure.

        Returns
        -------
        tuple[int, MpiFutureResultT]
            Original logical index and its evaluation result.
        """
        ...


class MpiExecutor(Protocol, Generic[CandidateExecutorT, MpiExecutorRecordT]):
    """Typed view of one MPI executor.

    Notes
    -----
    The evaluator keeps the executor protocol intentionally small so the MPI
    backend can be swapped or mocked without affecting evaluator semantics.
    """

    def submit(
        self,
        function: Callable[
            ...,
            tuple[int, MpiSubmitResultT],
        ],
        /,
        *args: object,
        **kwargs: object,
    ) -> MpiFuture[MpiSubmitResultT]:
        """Submit one callable for proposal-local execution.

        Parameters
        ----------
        function : Callable[..., tuple[int, MpiSubmitResultT]]
            Callable to execute on the MPI worker.
        *args : object
            Positional arguments forwarded to ``function``.
        **kwargs : object
            Keyword arguments forwarded to ``function``.

        Returns
        -------
        MpiFuture[MpiSubmitResultT]
            Future representing the submitted work item.
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Tear down the executor.

        Parameters
        ----------
        wait : bool, default=True
            Whether to wait for worker completion before returning.
        """
        ...


class MpiExecutorFactory(Protocol, Generic[CandidateExecutorT, MpiExecutorRecordT]):
    """Factory for creating one MPI executor instance.

    Notes
    -----
    The evaluator uses a factory rather than constructing executors directly so
    tests and custom runtimes can inject their own executor implementation.
    """

    def __call__(
        self,
        *,
        max_workers: int | None = None,
    ) -> MpiExecutor[CandidateExecutorT, MpiExecutorRecordT]:
        """Create an executor configured for one evaluator batch.

        Parameters
        ----------
        max_workers : int | None, optional
            Optional worker limit for the created executor.

        Returns
        -------
        MpiExecutor[CandidateExecutorT, MpiExecutorRecordT]
            Concrete executor instance.
        """
        ...


def _evaluate_indexed_proposal_outcome(
    *,
    index: int,
    problem: Problem[BoundaryT, CandidateT, MpiEvaluationPayloadT],
    request: EvaluationRequest[CandidateT],
) -> tuple[int, EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]]:
    """Execute one request and carry its original logical index."""
    return (
        index,
        evaluate_request_outcome(
            problem=problem,
            request=request,
        ),
    )


def _evaluate_indexed_request_attempt(
    *,
    index: int,
    problem: Problem[BoundaryT, CandidateT, MpiEvaluationPayloadT],
    request: EvaluationRequest[CandidateT],
) -> tuple[int, EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord]]:
    """Execute one request attempt and carry its original logical index."""
    return (
        index,
        evaluate_request_attempt(
            problem=problem,
            request=request,
        ),
    )


def _build_execution_resources(*, max_workers: int | None) -> ExecutionResources:
    """Return evaluator-owned execution resources for one MPI batch."""
    return ExecutionResources(
        parallel_owner="evaluator",
        nested_parallelism_policy=NestedParallelismPolicy.FORBID,
        owner_worker_count=max_workers,
        owner_backend="mpi",
    )


def _validate_mpi_configuration(*, max_workers: int | None) -> None:
    """Reject invalid MPI evaluator configuration."""
    if max_workers is not None and max_workers <= 0:
        msg = "max_workers must be positive when provided"
        raise ValueError(msg)


@dataclass(slots=True)
class MpiEvaluator(
    Evaluator[
        Problem[BoundaryT, CandidateT, MpiEvaluationPayloadT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord],
    ],
    Generic[BoundaryT, CandidateT, MpiEvaluationPayloadT],
):
    """MPI-backed synchronous batch evaluator.

    Parameters
    ----------
    max_workers : int | None, optional
        Optional worker limit for the MPI executor.
    _executor_factory : MpiExecutorFactory[CandidateT, RequestAlignedEvaluationRecord] | None, optional
        Optional executor factory used primarily for testing or custom runtime
        integration.

    Notes
    -----
    This evaluator is intentionally narrow: it preserves the canonical ordered
    batch contract and treats MPI purely as an optional execution backend.
    RunMethod semantics remain unchanged.
    """

    max_workers: int | None = None
    _executor_factory: MpiExecutorFactory[
        CandidateT,
        RequestAlignedEvaluationRecord,
    ] | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate MPI evaluator configuration."""
        _validate_mpi_configuration(max_workers=self.max_workers)

    @override
    def execution_resources(self) -> ExecutionResources:
        """Return execution resources for MPI evaluation.

        Returns
        -------
        ExecutionResources
            Resource contract describing evaluator-owned MPI parallelism.
        """
        return _build_execution_resources(max_workers=self.max_workers)

    def _create_default_executor(
        self,
    ) -> MpiExecutor[CandidateT, RequestAlignedEvaluationRecord]:
        """Create the default mpi4py-backed executor or raise a helpful error."""
        try:
            mpi4py_futures = import_module("mpi4py.futures")
        except ImportError as error:
            msg = (
                "mpi4py is required for MpiEvaluator. "
                "Install the optional mpi extra to use this backend."
            )
            raise ImportError(msg) from error

        executor_class = cast(
            Callable[..., MpiExecutor[CandidateT, RequestAlignedEvaluationRecord]],
            getattr(mpi4py_futures, "MPIPoolExecutor"),
        )
        return executor_class(max_workers=self.max_workers)

    def _create_executor(self) -> MpiExecutor[
        CandidateT,
        RequestAlignedEvaluationRecord,
    ]:
        """Return one concrete executor for one evaluator batch."""
        if self._executor_factory is None:
            return self._create_default_executor()

        return self._executor_factory(max_workers=self.max_workers)

    @override
    def evaluate(
        self,
        problem: Problem[BoundaryT, CandidateT, MpiEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> tuple[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord], ...]:
        """Execute a request batch through MPI.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, MpiEvaluationPayloadT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute.

        Returns
        -------
        tuple[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord], ...]
            Ordered outcomes aligned one-to-one with ``requests``.
        """
        executor = self._create_executor()

        try:
            futures = tuple(
                executor.submit(
                    _evaluate_indexed_proposal_outcome,
                    index=index,
                    problem=problem,
                    request=request,
                )
                for index, request in enumerate(requests)
            )
            return tuple(
                self._resolve_ordered_outcome(
                    future=future,
                    expected_index=expected_index,
                )
                for expected_index, future in enumerate(futures)
            )
        finally:
            executor.shutdown(wait=True)

    def evaluate_attempts(
        self,
        problem: Problem[BoundaryT, CandidateT, MpiEvaluationPayloadT],
        requests: Sequence[EvaluationRequest[CandidateT]],
    ) -> EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord]:
        """Execute a request batch through MPI into a dense attempt batch.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, MpiEvaluationPayloadT]
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequest[CandidateT]]
            Request batch to execute.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord]
            Dense attempt batch aligned to ``requests``.
        """
        executor = self._create_executor()

        try:
            futures = tuple(
                executor.submit(
                    _evaluate_indexed_request_attempt,
                    index=index,
                    problem=problem,
                    request=request,
                )
                for index, request in enumerate(requests)
            )
            attempts = tuple(
                self._resolve_ordered_attempt(
                    future=future,
                    expected_index=expected_index,
                )
                for expected_index, future in enumerate(futures)
            )
            return EvaluationAttemptBatch[
                CandidateT,
                RequestAlignedEvaluationRecord,
            ].from_single_request_attempts(attempts)
        finally:
            executor.shutdown(wait=True)

    def _resolve_ordered_result(
        self,
        *,
        future: MpiFuture[MpiSubmitResultT],
        expected_index: int,
        result_label: str,
    ) -> MpiSubmitResultT:
        """Return one future result and verify logical batch alignment."""
        resolved_index, result = future.result()
        if resolved_index != expected_index:
            msg = f"MPI executor returned a misaligned proposal {result_label}"
            raise ValueError(msg)

        return result

    def _resolve_ordered_outcome(
        self,
        *,
        future: MpiFuture[EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]],
        expected_index: int,
    ) -> EvaluationOutcome[CandidateT, RequestAlignedEvaluationRecord]:
        """Return one future result and verify logical batch alignment."""
        return self._resolve_ordered_result(
            future=future,
            expected_index=expected_index,
            result_label="outcome",
        )

    def _resolve_ordered_attempt(
        self,
        *,
        future: MpiFuture[
            EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord]
        ],
        expected_index: int,
    ) -> EvaluationAttemptBatch[CandidateT, RequestAlignedEvaluationRecord]:
        """Return one future attempt and verify logical batch alignment."""
        return self._resolve_ordered_result(
            future=future,
            expected_index=expected_index,
            result_label="attempt",
        )
