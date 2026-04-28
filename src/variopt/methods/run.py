"""Run-scoped search-method contracts."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

from ..artifacts import ProposalEvaluationSpec
from ..execution import (
    SEQUENTIAL_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    ExecutionModel,
)
from ..kernel import ProposalKernelHint
from ..typevars import EvaluationRecordT, ProposalT, RunMethodStateT
from .base import SearchMethod


class RunMethod(
    SearchMethod,
    ABC,
    Generic[RunMethodStateT, ProposalT, EvaluationRecordT],
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
        observations: Sequence[EvaluationRecordT],
    ) -> RunMethodStateT:
        """Advance the run-method state with one observation batch.

        Parameters
        ----------
        state : RunMethodStateT
            Current immutable run-method state.
        observations : Sequence[EvaluationRecordT]
            Evaluation records aligned to the proposals issued by ``ask``.

        Returns
        -------
        RunMethodStateT
            Advanced immutable run-method state.
        """

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
