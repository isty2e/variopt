"""Execution-side evaluation outcome artifacts."""

from dataclasses import dataclass
from typing import Generic, cast

from typing_extensions import TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .artifacts import Observation, RequestAlignedEvaluationRecord
from .kernel import KernelDiagnostics
from .typevars import CandidateT

OutcomeRecordT = TypeVar(
    "OutcomeRecordT",
    bound=RequestAlignedEvaluationRecord,
    default=Observation[CandidateT],
)


@dataclass(frozen=True, slots=True, init=False)
class EvaluationOutcome(FrozenGenericSlotsCompat, Generic[CandidateT, OutcomeRecordT]):
    """Executed evaluation outcome with explicit execution accounting.

    Parameters
    ----------
    record : OutcomeRecordT | None, optional
        Canonical request-aligned record produced by the evaluator or kernel.
    observation : Observation[CandidateT] | None, optional
        Scalar compatibility alias for ``record`` at the API boundary.
    evaluation_count : int, default=1
        Logical evaluation cost associated with the outcome.
    kernel_diagnostics : KernelDiagnostics | None, optional
        Optional execution-side diagnostics emitted by the kernel.

    Notes
    -----
    Exactly one of ``record`` or ``observation`` must be supplied. ``record``
    remains the canonical internal contract; ``observation`` is the scalar
    compatibility alias for request-local scalar studies.
    """

    record: OutcomeRecordT
    evaluation_count: int = 1
    kernel_diagnostics: KernelDiagnostics | None = None

    def __init__(
        self,
        *,
        record: OutcomeRecordT | None = None,
        observation: Observation[CandidateT] | None = None,
        evaluation_count: int = 1,
        kernel_diagnostics: KernelDiagnostics | None = None,
    ) -> None:
        """Create one canonical evaluation outcome.

        Parameters
        ----------
        record : OutcomeRecordT | None, optional
            Canonical request-aligned evaluation record.
        observation : Observation[CandidateT] | None, optional
            Scalar compatibility alias for ``record``.
        evaluation_count : int, default=1
            Logical evaluation cost associated with the outcome.
        kernel_diagnostics : KernelDiagnostics | None, optional
            Optional kernel-side diagnostics.

        Raises
        ------
        ValueError
            If neither or both of ``record`` and ``observation`` are provided.
        RuntimeError
            If record normalization fails unexpectedly.
        """
        if (record is None) == (observation is None):
            msg = "exactly one of record or observation must be provided"
            raise ValueError(msg)

        if record is not None:
            normalized_record = record
        elif observation is not None:
            normalized_record = cast(OutcomeRecordT, observation)
        else:
            msg = "evaluation record normalization failed"
            raise RuntimeError(msg)

        object.__setattr__(self, "record", normalized_record)
        object.__setattr__(self, "evaluation_count", evaluation_count)
        object.__setattr__(self, "kernel_diagnostics", kernel_diagnostics)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate outcome accounting metadata.

        Raises
        ------
        ValueError
            If ``evaluation_count`` is negative.
        """
        if self.evaluation_count < 0:
            msg = "evaluation_count must be non-negative"
            raise ValueError(msg)

    @property
    def observation(self) -> Observation[CandidateT]:
        """Return the scalar observation compatibility view.

        Returns
        -------
        Observation[CandidateT]
            Scalar observation carried by this outcome.

        Raises
        ------
        TypeError
            If the outcome record is not a scalar
            :class:`~variopt.artifacts.Observation`.

        Notes
        -----
        Prefer :attr:`record` in canonical internal code.
        """
        if not isinstance(self.record, Observation):
            msg = "evaluation outcome does not carry a scalar Observation"
            raise TypeError(msg)
        return cast(Observation[CandidateT], self.record)
