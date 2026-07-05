"""Kernel diagnostic artifact definitions."""

from dataclasses import dataclass
from enum import Enum

from variopt.generic_runtime import FrozenGenericSlotsCompat


class KernelStatus(Enum):
    """Execution status for one kernel episode.

    Notes
    -----
    These statuses describe the outcome of a single kernel invocation, not the
    enclosing run method or study as a whole.
    """

    CONVERGED = "converged"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class KernelDiagnostics(FrozenGenericSlotsCompat):
    """Execution-facing diagnostics for one kernel episode.

    Parameters
    ----------
    backend : str
        Name of the backend or implementation family that produced the
        diagnostics.
    method : str | None, optional
        Optional backend-specific method name.
    status : KernelStatus | None, optional
        Optional terminal status reported by the kernel backend.
    message : str | None, optional
        Optional human-readable detail for logs and traces.
    failed_attempt_count : int, default=0
        Number of failed inner attempts observed inside the kernel episode.
        These failures are diagnostic evidence, not additional top-level Study
        feedback slots.
    failed_evaluation_count : int, default=0
        Logical evaluation cost consumed by failed inner attempts.
    """

    backend: str
    method: str | None = None
    status: KernelStatus | None = None
    message: str | None = None
    failed_attempt_count: int = 0
    failed_evaluation_count: int = 0

    def __post_init__(self) -> None:
        """Validate diagnostic metadata.

        Raises
        ------
        ValueError
            If ``backend`` is empty, or if optional string fields are provided
            as empty strings.
        """
        if self.backend == "":
            msg = "backend must not be empty"
            raise ValueError(msg)

        if self.method == "":
            msg = "method must not be empty"
            raise ValueError(msg)

        if self.message == "":
            msg = "message must not be empty"
            raise ValueError(msg)

        if type(self.failed_attempt_count) is not int:
            msg = "failed_attempt_count must be int"
            raise TypeError(msg)

        if self.failed_attempt_count < 0:
            msg = "failed_attempt_count must be non-negative"
            raise ValueError(msg)

        if type(self.failed_evaluation_count) is not int:
            msg = "failed_evaluation_count must be int"
            raise TypeError(msg)

        if self.failed_evaluation_count < 0:
            msg = "failed_evaluation_count must be non-negative"
            raise ValueError(msg)

    def with_failed_attempts(
        self,
        *,
        failed_attempt_count: int,
        failed_evaluation_count: int,
    ) -> "KernelDiagnostics":
        """Return diagnostics annotated with inner failed-attempt accounting.

        Parameters
        ----------
        failed_attempt_count : int
            Number of failed inner attempts observed by the kernel episode.
        failed_evaluation_count : int
            Logical evaluation cost consumed by those failed attempts.

        Returns
        -------
        KernelDiagnostics
            Diagnostics with the same backend, method, status, and message plus
            the supplied failed-attempt summary.
        """
        return KernelDiagnostics(
            backend=self.backend,
            method=self.method,
            status=self.status,
            message=self.message,
            failed_attempt_count=failed_attempt_count,
            failed_evaluation_count=failed_evaluation_count,
        )
