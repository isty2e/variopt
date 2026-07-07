"""Run-scoped search-method contracts."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from typing_extensions import override

from ..artifacts import (
    EvaluationAttemptBatch,
    ProposalEvaluationSpec,
    materialize_success_records,
)
from ..artifacts.records import RequestAlignedEvaluationRecord
from ..execution import (
    SEQUENTIAL_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    ExecutionModel,
)
from ..kernel import ProposalKernelHint
from ..typevars import ProposalT, RunMethodStateT
from .base import SearchMethod

OutcomeCandidateT = TypeVar("OutcomeCandidateT")
RunMethodRecordT = TypeVar(
    "RunMethodRecordT",
    bound=RequestAlignedEvaluationRecord,
)


class UnsupportedEvaluationFailureError(RuntimeError):
    """Raised when a run method cannot safely assimilate failed attempts.

    Parameters
    ----------
    failure_count : int
        Number of recorded failures in the attempt batch.
    attempt_count : int
        Total number of request slots represented by the attempt batch.
    """

    failure_count: int
    attempt_count: int
    _message: str

    def __init__(self, failure_count: int, attempt_count: int) -> None:
        """Create an unsupported-failure assimilation error.

        Parameters
        ----------
        failure_count : int
            Number of recorded failures in the attempt batch.
        attempt_count : int
            Total number of request slots represented by the attempt batch.

        Raises
        ------
        TypeError
            If either count is not an ``int``.
        ValueError
            If ``failure_count`` is not positive, if ``attempt_count`` is
            negative, or if ``failure_count`` exceeds ``attempt_count``.
        """
        if type(failure_count) is not int:
            msg = "failure_count must be int"
            raise TypeError(msg)
        if type(attempt_count) is not int:
            msg = "attempt_count must be int"
            raise TypeError(msg)
        if failure_count <= 0:
            msg = "failure_count must be positive"
            raise ValueError(msg)
        if attempt_count < 0:
            msg = "attempt_count must be non-negative"
            raise ValueError(msg)
        if failure_count > attempt_count:
            msg = "failure_count must not exceed attempt_count"
            raise ValueError(msg)

        self.failure_count = failure_count
        self.attempt_count = attempt_count
        self._message = (
            f"run method does not support evaluation failure assimilation "
            f"({failure_count} failures in {attempt_count} attempts)"
        )
        super().__init__(failure_count, attempt_count)

    @override
    def __str__(self) -> str:
        """Return the human-readable unsupported-assimilation message."""
        return self._message


class RunMethod(
    SearchMethod,
    ABC,
    Generic[RunMethodStateT, ProposalT, RunMethodRecordT],
):
    """Run-scoped transition law over explicit search state.

    Any durable subordinate runtime needed by the algorithm, such as trust-region
    state, warm-start local-model memory, or surrogate state, remains owned by
    the run-method state until recurring reusable lifecycle pressure justifies a
    separate stable public contract.

    Notes
    -----
    Run methods own persistent optimizer state across repeated ``ask``/``tell``
    transitions. They do not own request-local execution backends.
    """

    @abstractmethod
    def create_initial_state(self) -> RunMethodStateT:
        """Return the canonical initial run-method state."""

    @abstractmethod
    def is_exhausted(self, state: RunMethodStateT) -> bool:
        """Report whether a run-method state can emit further proposals.

        Parameters
        ----------
        state : RunMethodStateT
            Run-method state to inspect.

        Returns
        -------
        bool
            ``True`` when the run method cannot emit additional proposals.
        """

    @abstractmethod
    def ask(
        self,
        state: RunMethodStateT,
        batch_size: int = 1,
    ) -> tuple[tuple[ProposalT, ...], RunMethodStateT]:
        """Emit the next proposal batch and advanced run-method state.

        Parameters
        ----------
        state : RunMethodStateT
            Current immutable run-method state.
        batch_size : int, default=1
            Maximum number of proposals to emit.

        Returns
        -------
        tuple[tuple[ProposalT, ...], RunMethodStateT]
            Proposal batch together with the advanced immutable state.
        """

    @abstractmethod
    def tell(
        self,
        state: RunMethodStateT,
        observations: Sequence[RunMethodRecordT],
    ) -> RunMethodStateT:
        """Advance the run-method state with request-aligned feedback records.

        Parameters
        ----------
        state : RunMethodStateT
            Current immutable run-method state.
        observations : Sequence[RunMethodRecordT]
            Request-aligned feedback records materialized from successful
            evaluation attempts for proposals issued by ``ask``.

        Returns
        -------
        RunMethodStateT
            Advanced immutable run-method state.
        """

    def tell_attempts(
        self,
        state: RunMethodStateT,
        attempts: EvaluationAttemptBatch[OutcomeCandidateT, RunMethodRecordT],
    ) -> RunMethodStateT:
        """Advance state from a dense evaluation-attempt batch.

        Parameters
        ----------
        state : RunMethodStateT
            Current immutable run-method state.
        attempts : EvaluationAttemptBatch[OutcomeCandidateT, RunMethodRecordT]
            Dense request-aligned batch containing successful materialized
            feedback records and recorded user-code evaluation failures.

        Returns
        -------
        RunMethodStateT
            Advanced immutable run-method state.

        Raises
        ------
        UnsupportedEvaluationFailureError
            If ``attempts`` contains failures and the concrete run method has not
            overridden this hook to consume failed proposal lifecycle explicitly.

        Notes
        -----
        The default implementation delegates success-only batches to
        :meth:`tell` using each success payload. It never drops failures before
        delegation because failed attempts may own pending proposal lifecycle in
        stateful optimizers.
        """
        if attempts.has_failures:
            raise UnsupportedEvaluationFailureError(
                failure_count=len(attempts.failures),
                attempt_count=attempts.attempt_count,
            )

        return self.tell(state, materialize_success_records(attempts.successes))

    def proposal_kernel_hints(
        self,
        state: RunMethodStateT,
        proposals: Sequence[ProposalT],
    ) -> tuple[ProposalKernelHint | None, ...] | None:
        """Return optional per-proposal kernel hints for an issued batch.

        Parameters
        ----------
        state : RunMethodStateT
            Post-``ask`` run-method state that owns bookkeeping for
            ``proposals``.
        proposals : Sequence[ProposalT]
            Issued proposals for which kernel hints are requested.

        Returns
        -------
        tuple[ProposalKernelHint | None, ...] | None
            Per-proposal kernel hints, or ``None`` when the run method does not
            supply them.

        The supplied ``state`` is the post-``ask`` run-method state that owns
        any run-scoped bookkeeping associated with ``proposals``. Concrete hint
        semantics belong to the kernel family that consumes them.
        """
        _ = state, proposals
        return None

    def proposal_evaluation_specs(
        self,
        state: RunMethodStateT,
        proposals: Sequence[ProposalT],
    ) -> tuple[ProposalEvaluationSpec | None, ...] | None:
        """Return optional per-proposal evaluation specs for an issued batch.

        Parameters
        ----------
        state : RunMethodStateT
            Post-``ask`` run-method state that owns bookkeeping for
            ``proposals``.
        proposals : Sequence[ProposalT]
            Issued proposals for which evaluation specs are requested.

        Returns
        -------
        tuple[ProposalEvaluationSpec | None, ...] | None
            Per-proposal evaluation specs, or ``None`` when the run method
            does not supply them.

        The supplied ``state`` is the post-``ask`` run-method state that owns
        any run-scoped bookkeeping associated with ``proposals``. Concrete
        request semantics belong to evaluation-request-spec realizations rather
        than to this generic run-method contract.
        """
        _ = state, proposals
        return None

    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        """Return execution models this run method preserves exactly."""
        return frozenset(
            {
                SEQUENTIAL_EXECUTION_MODEL,
                SYNC_BATCH_EXECUTION_MODEL,
            },
        )

    def is_checkpoint_safe_state(self, state: RunMethodStateT) -> bool:
        """Return whether ``state`` can be persisted as a checkpoint.

        Parameters
        ----------
        state : RunMethodStateT
            Run-method state to inspect.

        Returns
        -------
        bool
            ``True`` when checkpointing this state preserves resumable optimizer
            semantics. Stateless or always-serializable run methods use the
            default ``True`` contract.
        """
        _ = state
        return True
