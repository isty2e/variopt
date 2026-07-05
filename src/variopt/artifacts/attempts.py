"""Evaluation-attempt failure artifact definitions."""

from dataclasses import dataclass
from typing import Generic

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..typevars import CandidateT
from .requests import EvaluationRequest, Proposal


@dataclass(frozen=True, slots=True)
class EvaluationExceptionSnapshot(FrozenGenericSlotsCompat):
    """Serializable summary of an exception raised during candidate evaluation.

    Parameters
    ----------
    exception_module : str
        Module that defines the exception type.
    exception_qualname : str
        Qualified name of the exception type inside ``exception_module``.
    message : str, default=""
        Stringified exception message.
    """

    exception_module: str
    exception_qualname: str
    message: str = ""

    @classmethod
    def from_exception(cls, exception: BaseException) -> Self:
        """Build a snapshot from a recordable user-code exception.

        Parameters
        ----------
        exception : BaseException
            Exception raised while evaluating a concrete candidate. The raw
            exception object is not stored. ``KeyboardInterrupt`` and
            ``SystemExit`` are rejected because they are not recordable
            evaluation failures.

        Returns
        -------
        Self
            JSON- and pickle-friendly exception summary.

        Raises
        ------
        TypeError
            If ``exception`` is not an ``Exception`` instance.
        """
        if not isinstance(exception, Exception):
            msg = "exception must be an Exception instance"
            raise TypeError(msg)

        exception_type = type(exception)
        return cls(
            exception_module=exception_type.__module__,
            exception_qualname=exception_type.__qualname__,
            message=str(exception),
        )

    def __post_init__(self) -> None:
        """Validate snapshot fields."""
        if type(self.exception_module) is not str:
            msg = "exception_module must be str"
            raise TypeError(msg)

        if self.exception_module == "":
            msg = "exception_module must not be empty"
            raise ValueError(msg)

        if type(self.exception_qualname) is not str:
            msg = "exception_qualname must be str"
            raise TypeError(msg)

        if self.exception_qualname == "":
            msg = "exception_qualname must not be empty"
            raise ValueError(msg)

        if type(self.message) is not str:
            msg = "message must be str"
            raise TypeError(msg)

    @property
    def exception_type(self) -> str:
        """Return the fully qualified exception type name.

        Returns
        -------
        str
            ``"<module>.<qualname>"`` for the captured exception type.
        """
        return f"{self.exception_module}.{self.exception_qualname}"


@dataclass(frozen=True, slots=True)
class EvaluationFailure(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Recorded failure for one concrete evaluation request.

    Parameters
    ----------
    request : EvaluationRequest[CandidateT]
        Canonical request whose candidate evaluation failed.
    exception : EvaluationExceptionSnapshot
        Serializable exception summary. Raw exception objects are intentionally
        excluded from this artifact.
    evaluation_count : int, default=1
        Logical evaluation cost consumed by the failed attempt.
    """

    request: EvaluationRequest[CandidateT]
    exception: EvaluationExceptionSnapshot
    evaluation_count: int = 1

    @classmethod
    def from_exception(
        cls,
        *,
        request: EvaluationRequest[CandidateT],
        exception: Exception,
        evaluation_count: int = 1,
    ) -> Self:
        """Build a failure artifact from a recordable user-code exception.

        Parameters
        ----------
        request : EvaluationRequest[CandidateT]
            Request being evaluated when ``exception`` was raised.
        exception : Exception
            Exception raised by user evaluation code.
        evaluation_count : int, default=1
            Logical evaluation cost consumed by the failed attempt.

        Returns
        -------
        Self
            Request-aligned failure artifact.
        """
        return cls(
            request=request,
            exception=EvaluationExceptionSnapshot.from_exception(exception),
            evaluation_count=evaluation_count,
        )

    def __post_init__(self) -> None:
        """Validate failure artifact invariants."""
        if type(self.request) is not EvaluationRequest:
            msg = "request must be an EvaluationRequest"
            raise TypeError(msg)

        if type(self.exception) is not EvaluationExceptionSnapshot:
            msg = "exception must be an EvaluationExceptionSnapshot"
            raise TypeError(msg)

        if type(self.evaluation_count) is not int:
            msg = "evaluation_count must be int"
            raise TypeError(msg)

        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

    @property
    def candidate(self) -> CandidateT:
        """Return the candidate carried by the failed request.

        Returns
        -------
        CandidateT
            Candidate that failed during evaluation.
        """
        return self.request.candidate

    @property
    def proposal(self) -> Proposal[CandidateT]:
        """Return the proposal compatibility view.

        Returns
        -------
        Proposal[CandidateT]
            Proposal owned by :attr:`request`.
        """
        return self.request.proposal

    @property
    def proposal_id(self) -> str | None:
        """Return the optional proposal identifier for the failed request.

        Returns
        -------
        str | None
            Proposal identifier carried by the failed request, if present.
        """
        return self.request.proposal_id
