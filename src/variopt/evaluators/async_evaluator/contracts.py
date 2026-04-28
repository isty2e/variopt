"""Exact-async evaluator abstract contracts."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, cast

from typing_extensions import override

from ...typevars import EvaluationRequestT, EvaluationT, ProblemT
from ..base import Evaluator
from .artifacts import (
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchResumeHandle,
    store_completion_group,
)
from .runtime import EvaluatorBackedBatchSession
from .sessions import EvaluationBatchSession


class AsyncEvaluator(
    Evaluator[ProblemT, EvaluationRequestT, EvaluationT],
    ABC,
    Generic[ProblemT, EvaluationRequestT, EvaluationT],
):
    """Evaluator specialization that can surface out-of-order completion.

    Notes
    -----
    Async evaluators expose an explicit session lifecycle while still
    supporting the plain batch :meth:`evaluate` contract for compatibility.
    """

    def open_session(
        self,
        problem: ProblemT,
        requests: Sequence[EvaluationRequestT],
    ) -> EvaluationBatchSession[EvaluationT]:
        """Open an exact-async batch session.

        Parameters
        ----------
        problem : ProblemT
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequestT]
            Request batch to execute asynchronously.

        Returns
        -------
        EvaluationBatchSession[EvaluationT]
            Session that can be polled and cancelled.
        """
        return EvaluatorBackedBatchSession(
            evaluator=self,
            _handle=self.submit_batch(problem, requests),
        )

    @abstractmethod
    def submit_batch(
        self,
        problem: ProblemT,
        requests: Sequence[EvaluationRequestT],
    ) -> EvaluationBatchHandle:
        """Submit one exact-async logical batch.

        Parameters
        ----------
        problem : ProblemT
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequestT]
            Request batch to execute asynchronously.

        Returns
        -------
        EvaluationBatchHandle
            Immutable handle for the submitted batch.

        Notes
        -----
        This remains the compatibility submission hook behind
        :meth:`open_session`.
        """

    @abstractmethod
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[CompletionGroup[EvaluationT]]:
        """Poll one submitted exact-async batch by immutable handle.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Handle returned when the batch was submitted.

        Returns
        -------
        Sequence[CompletionGroup[EvaluationT]]
            Newly completed groups in logical batch order.

        Notes
        -----
        This remains the compatibility polling hook behind
        :meth:`open_session`.
        """

    @abstractmethod
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        """Cancel one submitted exact-async batch by immutable handle.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Handle identifying the submitted batch.

        Notes
        -----
        This remains the compatibility cancellation hook behind
        :meth:`open_session`.
        """

    @override
    def evaluate(
        self,
        problem: ProblemT,
        requests: Sequence[EvaluationRequestT],
    ) -> tuple[EvaluationT, ...]:
        """Execute a logical batch and collect ordered outcomes.

        Parameters
        ----------
        problem : ProblemT
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequestT]
            Request batch to execute.

        Returns
        -------
        tuple[EvaluationT, ...]
            Ordered outcomes aligned one-to-one with ``requests``.
        """
        session = self.open_session(problem, requests)
        ordered_outcomes: list[EvaluationT | None] = [
            None
        ] * session.handle.request_count
        completed_count = 0
        try:
            while completed_count < session.handle.request_count:
                completion_groups = tuple(session.poll())
                for completion_group in completion_groups:
                    completed_count += store_completion_group(
                        ordered_outcomes,
                        completion_group,
                        request_count=session.handle.request_count,
                    )
        except BaseException:
            session.cancel()
            raise

        return tuple(
            cast(EvaluationT, outcome)
            for outcome in ordered_outcomes
        )


class ResumableAsyncEvaluator(
    AsyncEvaluator[ProblemT, EvaluationRequestT, EvaluationT],
    ABC,
    Generic[ProblemT, EvaluationRequestT, EvaluationT],
):
    """Optional async-evaluator capability that can reopen suspended sessions.

    Notes
    -----
    This capability is separate from ``AsyncEvaluator`` so resumability remains
    an explicit opt-in contract.
    """

    @abstractmethod
    def resume_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[EvaluationT]:
        """Resume a previously suspended logical batch session.

        Parameters
        ----------
        handle : EvaluationBatchResumeHandle
            Evaluator-owned resume handle produced during suspension.

        Returns
        -------
        EvaluationBatchSession[EvaluationT]
            Resumed batch session.
        """
