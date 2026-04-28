"""Runtime helpers backing exact-async evaluator contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol

from typing_extensions import TypeVar, override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .artifacts import CompletionGroup, EvaluationBatchHandle
from .sessions import EvaluationBatchSession

EvaluationT = TypeVar("EvaluationT")


class AsyncBatchHooks(Protocol[EvaluationT]):
    """Minimal evaluator hook surface needed by one evaluator-backed session.

    Notes
    -----
    The protocol captures the small poll/cancel surface required by the
    exact-async evaluator runtime.
    """

    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[CompletionGroup[EvaluationT]]:
        """Poll one submitted batch handle for newly completed groups.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Immutable handle identifying the submitted evaluation batch.

        Returns
        -------
        Sequence[CompletionGroup[EvaluationT]]
            Newly completed groups observed since the previous poll.
        """
        ...

    def cancel(self, handle: EvaluationBatchHandle) -> None:
        """Cancel one submitted batch handle.

        Parameters
        ----------
        handle : EvaluationBatchHandle
            Immutable handle identifying the submitted evaluation batch.
        """
        ...


@dataclass(frozen=True, slots=True)
class EvaluatorBackedBatchSession(FrozenGenericSlotsCompat,
    EvaluationBatchSession[EvaluationT],
    Generic[EvaluationT],
):
    """Session wrapper that delegates lifecycle hooks back to one evaluator.

    Parameters
    ----------
    evaluator : AsyncBatchHooks[EvaluationT]
        Evaluator hook object that owns polling and cancellation.
    _handle : EvaluationBatchHandle
        Immutable handle for the open evaluation batch.
    """

    evaluator: AsyncBatchHooks[EvaluationT]
    _handle: EvaluationBatchHandle

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        """Return the immutable handle owned by the wrapped evaluator."""
        return self._handle

    @override
    def poll(self) -> tuple[CompletionGroup[EvaluationT], ...]:
        """Delegate batch polling back to the wrapped evaluator.

        Returns
        -------
        tuple[CompletionGroup[EvaluationT], ...]
            Newly completed groups observed since the previous poll.
        """
        return tuple(self.evaluator.poll(self.handle))

    @override
    def cancel(self) -> None:
        """Delegate cancellation back to the wrapped evaluator."""
        self.evaluator.cancel(self.handle)
