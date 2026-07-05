"""Typed joblib API shims shared by joblib-backed evaluators."""

from collections.abc import Callable, Generator, Iterable
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar

from typing_extensions import TypeVar as DefaultTypeVar

from ...artifacts import (
    ObjectiveVectorPayload,
    ObservationPayload,
)
from ...artifacts.records import RequestAlignedEvaluationRecord

BoundaryT = TypeVar("BoundaryT")
JoblibEvaluationPayload: TypeAlias = (
    RequestAlignedEvaluationRecord | ObservationPayload | ObjectiveVectorPayload
)
JoblibEvaluationPayloadT = DefaultTypeVar(
    "JoblibEvaluationPayloadT",
    bound=JoblibEvaluationPayload,
    default=ObservationPayload,
)
ListResultT = TypeVar("ListResultT")
YieldResultT = TypeVar("YieldResultT", covariant=True)


class JoblibDelayedFactory(Protocol):
    """Typed view of ``joblib.delayed`` for one result type.

    Notes
    -----
    The protocol abstracts over the small portion of ``joblib.delayed`` used by
    the joblib-backed evaluator family.
    """

    def __call__(
        self,
        function: Callable[..., object],
    ) -> Callable[..., object]:
        """Wrap one callable for delayed joblib execution."""
        ...


class JoblibListParallelRunner(Protocol, Generic[ListResultT]):
    """Typed view of list-returning ``joblib.Parallel`` calls.

    Notes
    -----
    The runner consumes delayed tasks eagerly and materializes a realized list
    of results.
    """

    def __call__(self, tasks: Iterable[object]) -> list[ListResultT]:
        """Execute one task iterable and return a realized list."""
        ...


class JoblibGeneratorParallelRunner(Protocol, Generic[YieldResultT]):
    """Typed view of generator-returning ``joblib.Parallel`` calls.

    Notes
    -----
    The runner yields results from ``joblib.Parallel`` in unordered generator
    mode.
    """

    def __call__(
        self,
        tasks: Iterable[object],
    ) -> Generator[YieldResultT, None, None]:
        """Execute one task iterable and stream results."""
        ...


class JoblibListParallelFactory(Protocol, Generic[ListResultT]):
    """Typed view of list-returning ``joblib.Parallel`` construction.

    Notes
    -----
    The factory captures the list-returning ``Parallel`` configuration used by
    the synchronous and exact-async evaluators.
    """

    def __call__(
        self,
        *,
        n_jobs: int,
        backend: Literal["loky", "threading"],
        return_as: Literal["list"] = "list",
    ) -> JoblibListParallelRunner[ListResultT]:
        """Construct one list-returning joblib runner."""
        ...


class JoblibGeneratorParallelFactory(Protocol, Generic[YieldResultT]):
    """Typed view of generator-returning ``joblib.Parallel`` construction.

    Notes
    -----
    The factory captures the unordered generator configuration used by the
    async joblib evaluator.
    """

    def __call__(
        self,
        *,
        n_jobs: int,
        backend: Literal["loky", "threading"],
        return_as: Literal["generator_unordered"],
    ) -> JoblibGeneratorParallelRunner[YieldResultT]:
        """Construct one unordered generator joblib runner."""
        ...
