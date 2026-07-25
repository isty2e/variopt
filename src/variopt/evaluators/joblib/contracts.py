"""Typed joblib API shims shared by joblib-backed evaluators."""

from collections.abc import Callable, Generator, Iterable
from types import TracebackType
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

    def __enter__(self) -> "JoblibListParallelRunner[ListResultT]":
        """Enter one persistent parallel execution scope."""
        ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close one persistent parallel execution scope."""
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
        backend: Literal["loky", "threading"] | None = None,
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


class JoblibEffectiveJobs(Protocol):
    """Typed view of ``joblib.effective_n_jobs``."""

    def __call__(self, n_jobs: int) -> int:
        """Resolve the effective worker count for one request."""
        ...


class JoblibParallelConfiguration(Protocol):
    """Typed context-manager view of ``joblib.parallel_config``."""

    def __enter__(self) -> object:
        """Activate the selected backend configuration."""
        ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the previous backend configuration."""
        ...


class JoblibParallelConfigurationFactory(Protocol):
    """Typed factory view of ``joblib.parallel_config``."""

    def __call__(
        self,
        *,
        backend: Literal["loky", "threading"],
        idle_worker_timeout: float | None = None,
    ) -> JoblibParallelConfiguration:
        """Create one temporary backend configuration."""
        ...
