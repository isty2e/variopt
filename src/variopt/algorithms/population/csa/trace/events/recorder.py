"""Mutable CSA event-trace recorder seam."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from ......typevars import CandidateT
from ...banking.bank import Bank
from ...banking.reference import ReferenceBank
from .artifacts import (
    BoundaryActionName,
    CSAActiveGenerationTrace,
    CSAChildFamily,
    CSAGenerationTrace,
    CSAPrimarySource,
    CSAProposalFamilyTrace,
)
from .state import CSAEventTraceState


@dataclass(slots=True, init=False)
class CSAEventTraceRecorder(Generic[CandidateT]):
    """Mutable adapter over ``CSAEventTraceState`` for diagnostics seams.

    Parameters
    ----------
    trace_state : CSAEventTraceState[CandidateT]
        Immutable trace state wrapped by the recorder. When omitted, the
        recorder starts from an empty trace state.
    """

    trace_state: CSAEventTraceState[CandidateT]

    def __init__(
        self,
        trace_state: CSAEventTraceState[CandidateT] | None = None,
    ) -> None:
        """Initialize the recorder with an optional immutable trace state.

        Parameters
        ----------
        trace_state : CSAEventTraceState[CandidateT] | None, default=None
            Existing immutable trace state to wrap. ``None`` creates a fresh
            empty trace state.
        """
        self.trace_state = (
            CSAEventTraceState[CandidateT]() if trace_state is None else trace_state
        )

    def start_generation(
        self,
        *,
        stage_index: int,
        stage_round: int,
        cycle_count: int,
        bank: Bank[CandidateT],
        reference_bank: ReferenceBank[CandidateT],
        bank_status_before: tuple[bool, ...],
        seed_mask: frozenset[int],
        partner_mask: frozenset[int],
        seed_batch: tuple[int, ...],
        proposal_families_before: tuple[CSAProposalFamilyTrace, ...],
    ) -> None:
        """Start one traced generation in the wrapped immutable state.

        Parameters
        ----------
        stage_index : int
            Active CSA stage index.
        stage_round : int
            Round count within the active stage.
        cycle_count : int
            Global cycle count before generation execution.
        bank : Bank[CandidateT]
            Bank snapshot before child generation.
        reference_bank : ReferenceBank[CandidateT]
            Reference-bank snapshot before child generation.
        bank_status_before : tuple[bool, ...]
            Per-entry admissibility/status flags aligned with ``bank``.
        seed_mask : frozenset[int]
            Seed-selection mask active for the generation.
        partner_mask : frozenset[int]
            Partner-selection mask active for the generation.
        seed_batch : tuple[int, ...]
            Concrete seed indices selected for the generation.
        proposal_families_before : tuple[CSAProposalFamilyTrace, ...]
            Proposal-family statistics before emitting children.
        """
        self.trace_state = self.trace_state.start_generation(
            stage_index=stage_index,
            stage_round=stage_round,
            cycle_count=cycle_count,
            bank=bank,
            reference_bank=reference_bank,
            bank_status_before=bank_status_before,
            seed_mask=seed_mask,
            partner_mask=partner_mask,
            seed_batch=seed_batch,
            proposal_families_before=proposal_families_before,
        )

    def record_emitted_child(
        self,
        *,
        family: CSAChildFamily,
        proposal_family_key: str | None,
        seed_index: int,
        primary_source: CSAPrimarySource,
        partner_indices: tuple[int, ...],
        candidate: CandidateT,
    ) -> None:
        """Append one emitted child to the wrapped immutable trace state.

        Parameters
        ----------
        family : CSAChildFamily
            Child family used to emit the candidate.
        proposal_family_key : str | None
            Optional proposal-family key associated with the emission.
        seed_index : int
            Seed index responsible for the child.
        primary_source : CSAPrimarySource
            Primary source category for the emitted child.
        partner_indices : tuple[int, ...]
            Partner indices used by the operator, if any.
        candidate : CandidateT
            Emitted child candidate.
        """
        self.trace_state = self.trace_state.record_emitted_child(
            family=family,
            proposal_family_key=proposal_family_key,
            seed_index=seed_index,
            primary_source=primary_source,
            partner_indices=partner_indices,
            candidate=candidate,
        )

    def record_shuffled_pool(self, shuffled_pool: Sequence[CandidateT]) -> None:
        """Record one shuffled pool in the wrapped immutable trace state.

        Parameters
        ----------
        shuffled_pool : Sequence[CandidateT]
            Candidate pool after generation-level shuffling.
        """
        self.trace_state = self.trace_state.record_shuffled_pool(shuffled_pool)

    def record_bank_update_step(
        self,
        *,
        candidate: CandidateT,
        value: float,
        bank_before: Bank[CandidateT],
        bank_after: Bank[CandidateT],
        distance_cutoff_before: float | None,
        distance_cutoff_after: float | None,
        changed_indices: frozenset[int],
        significant_update_indices: frozenset[int],
    ) -> None:
        """Append one bank-update step to the wrapped immutable trace state.

        Parameters
        ----------
        candidate : CandidateT
            Candidate evaluated against the bank.
        value : float
            Objective value associated with ``candidate``.
        bank_before : Bank[CandidateT]
            Bank snapshot before the update step.
        bank_after : Bank[CandidateT]
            Bank snapshot after the update step.
        distance_cutoff_before : float | None
            Distance cutoff before the update step.
        distance_cutoff_after : float | None
            Distance cutoff after the update step.
        changed_indices : frozenset[int]
            Bank indices touched by the update.
        significant_update_indices : frozenset[int]
            Indices marked as significant updates by the banking logic.
        """
        self.trace_state = self.trace_state.record_bank_update_step(
            candidate=candidate,
            value=value,
            bank_before=bank_before,
            bank_after=bank_after,
            distance_cutoff_before=distance_cutoff_before,
            distance_cutoff_after=distance_cutoff_after,
            changed_indices=changed_indices,
            significant_update_indices=significant_update_indices,
        )

    def finish_generation(
        self,
        *,
        cycle_count: int,
        bank: Bank[CandidateT],
        reference_bank: ReferenceBank[CandidateT],
        bank_status_after: tuple[bool, ...],
        proposal_families_after: tuple[CSAProposalFamilyTrace, ...],
        pending_boundary_action_after: BoundaryActionName,
        refresh_active_after: bool,
    ) -> None:
        """Finish one generation in the wrapped immutable trace state.

        Parameters
        ----------
        cycle_count : int
            Global cycle count after generation execution.
        bank : Bank[CandidateT]
            Bank snapshot after generation execution.
        reference_bank : ReferenceBank[CandidateT]
            Reference-bank snapshot after generation execution.
        bank_status_after : tuple[bool, ...]
            Per-entry admissibility/status flags aligned with ``bank``.
        proposal_families_after : tuple[CSAProposalFamilyTrace, ...]
            Proposal-family statistics after generation execution.
        pending_boundary_action_after : BoundaryActionName
            Pending boundary action queued by the generation.
        refresh_active_after : bool
            Whether refresh mode remains active after the generation.
        """
        self.trace_state = self.trace_state.finish_generation(
            cycle_count=cycle_count,
            bank=bank,
            reference_bank=reference_bank,
            bank_status_after=bank_status_after,
            proposal_families_after=proposal_families_after,
            pending_boundary_action_after=pending_boundary_action_after,
            refresh_active_after=refresh_active_after,
        )

    def snapshot(self) -> tuple[CSAGenerationTrace[CandidateT], ...]:
        """Return the completed generation traces.

        Returns
        -------
        tuple[CSAGenerationTrace[CandidateT], ...]
            Completed generation traces, excluding any in-flight generation.
        """
        return self.trace_state.snapshot()

    def completed_snapshot(self) -> tuple[CSAGenerationTrace[CandidateT], ...]:
        """Return the completed generation traces regardless of active state.

        Returns
        -------
        tuple[CSAGenerationTrace[CandidateT], ...]
            Completed generation traces accumulated so far.
        """
        return self.trace_state.completed_snapshot()

    def active_snapshot(self) -> CSAActiveGenerationTrace[CandidateT] | None:
        """Return the in-flight generation trace snapshot, if any.

        Returns
        -------
        CSAActiveGenerationTrace[CandidateT] | None
            Active generation trace snapshot, or ``None`` when no generation is
            currently open.
        """
        return self.trace_state.active_snapshot()
