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
    """

    backend: str
    method: str | None = None
    status: KernelStatus | None = None
    message: str | None = None

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
