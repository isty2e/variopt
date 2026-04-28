"""Immutable CSA event-trace reducer state."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ......typevars import CandidateT
from ...banking.bank import Bank
from ...banking.reference import ReferenceBank
from .artifacts import (
    BoundaryActionName,
    CSAActiveGenerationTrace,
    CSABankEntryTrace,
    CSABankUpdateStepTrace,
    CSAChildEmissionTrace,
    CSAChildFamily,
    CSAGenerationTrace,
    CSAPrimarySource,
    CSAProposalFamilyTrace,
    trace_bank_entries,
)


@dataclass(frozen=True, slots=True)
class GenerationTraceState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Immutable in-flight generation trace state.

    Parameters
    ----------
    generation_index : int
        Zero-based generation index.
    stage_index : int
        Active CSA stage index for the generation.
    stage_round : int
        Number of completed rounds within the active stage.
    cycle_count_before : int
        Cycle counter value before the generation starts.
    bank_before : tuple[CSABankEntryTrace[CandidateT], ...]
        Bank snapshot before generation execution.
    reference_before : tuple[CSABankEntryTrace[CandidateT], ...]
        Reference-bank snapshot before generation execution.
    bank_status_before : tuple[bool, ...]
        Bank occupancy mask before generation execution.
    seed_mask : tuple[int, ...]
        Seed indices active for the generation.
    partner_mask : tuple[int, ...]
        Partner indices active for the generation.
    seed_batch : tuple[int, ...]
        Active seed batch for the generation.
    proposal_families_before : tuple[CSAProposalFamilyTrace, ...]
        Proposal-family weights observed before child emission.
    child_pool : tuple[CSAChildEmissionTrace[CandidateT], ...], default=()
        Emitted child traces in generation order.
    shuffled_pool : tuple[CandidateT, ...], default=()
        Shuffled child pool after post-emission reordering.
    update_steps : tuple[CSABankUpdateStepTrace[CandidateT], ...], default=()
        Bank update steps recorded while consuming the child pool.
    """

    generation_index: int
    stage_index: int
    stage_round: int
    cycle_count_before: int
    bank_before: tuple[CSABankEntryTrace[CandidateT], ...]
    reference_before: tuple[CSABankEntryTrace[CandidateT], ...]
    bank_status_before: tuple[bool, ...]
    seed_mask: tuple[int, ...]
    partner_mask: tuple[int, ...]
    seed_batch: tuple[int, ...]
    proposal_families_before: tuple[CSAProposalFamilyTrace, ...]
    child_pool: tuple[CSAChildEmissionTrace[CandidateT], ...] = ()
    shuffled_pool: tuple[CandidateT, ...] = ()
    update_steps: tuple[CSABankUpdateStepTrace[CandidateT], ...] = ()

    def record_emitted_child(
        self,
        *,
        family: CSAChildFamily,
        proposal_family_key: str | None,
        seed_index: int,
        primary_source: CSAPrimarySource,
        partner_indices: tuple[int, ...],
        candidate: CandidateT,
    ) -> Self:
        """Return a trace state with one emitted child appended.

        Parameters
        ----------
        family : CSAChildFamily
            Child family label for the emitted proposal.
        proposal_family_key : str | None
            Proposal-family key, when available.
        seed_index : int
            Seed index that generated the child.
        primary_source : CSAPrimarySource
            Primary source classification for the child.
        partner_indices : tuple[int, ...]
            Partner indices used while generating the child.
        candidate : CandidateT
            Emitted child candidate.

        Returns
        -------
        Self
            Updated in-flight generation state with the child appended.
        """
        return type(self)(
            generation_index=self.generation_index,
            stage_index=self.stage_index,
            stage_round=self.stage_round,
            cycle_count_before=self.cycle_count_before,
            bank_before=self.bank_before,
            reference_before=self.reference_before,
            bank_status_before=self.bank_status_before,
            seed_mask=self.seed_mask,
            partner_mask=self.partner_mask,
            seed_batch=self.seed_batch,
            proposal_families_before=self.proposal_families_before,
            child_pool=self.child_pool
            + (
                CSAChildEmissionTrace(
                    family=family,
                    proposal_family_key=proposal_family_key,
                    seed_index=seed_index,
                    primary_source=primary_source,
                    partner_indices=partner_indices,
                    candidate=candidate,
                ),
            ),
            shuffled_pool=self.shuffled_pool,
            update_steps=self.update_steps,
        )

    def record_shuffled_pool(self, shuffled_pool: Sequence[CandidateT]) -> Self:
        """Return a trace state with one recorded shuffled pool.

        Parameters
        ----------
        shuffled_pool : Sequence[CandidateT]
            Child pool after generation-local shuffling.

        Returns
        -------
        Self
            Updated in-flight generation state with the shuffled pool recorded.
        """
        return type(self)(
            generation_index=self.generation_index,
            stage_index=self.stage_index,
            stage_round=self.stage_round,
            cycle_count_before=self.cycle_count_before,
            bank_before=self.bank_before,
            reference_before=self.reference_before,
            bank_status_before=self.bank_status_before,
            seed_mask=self.seed_mask,
            partner_mask=self.partner_mask,
            seed_batch=self.seed_batch,
            proposal_families_before=self.proposal_families_before,
            child_pool=self.child_pool,
            shuffled_pool=tuple(shuffled_pool),
            update_steps=self.update_steps,
        )

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
    ) -> Self:
        """Return a trace state with one bank-update step appended.

        Parameters
        ----------
        candidate : CandidateT
            Candidate consumed by the bank update step.
        value : float
            Objective value associated with ``candidate``.
        bank_before : Bank[CandidateT]
            Bank snapshot before the update step.
        bank_after : Bank[CandidateT]
            Bank snapshot after the update step.
        distance_cutoff_before : float | None
            Distance cutoff before the update step, if initialized.
        distance_cutoff_after : float | None
            Distance cutoff after the update step, if initialized.
        changed_indices : frozenset[int]
            Bank indices changed by the update step.
        significant_update_indices : frozenset[int]
            Bank indices marked as significant updates.

        Returns
        -------
        Self
            Updated in-flight generation state with the bank update appended.
        """
        return type(self)(
            generation_index=self.generation_index,
            stage_index=self.stage_index,
            stage_round=self.stage_round,
            cycle_count_before=self.cycle_count_before,
            bank_before=self.bank_before,
            reference_before=self.reference_before,
            bank_status_before=self.bank_status_before,
            seed_mask=self.seed_mask,
            partner_mask=self.partner_mask,
            seed_batch=self.seed_batch,
            proposal_families_before=self.proposal_families_before,
            child_pool=self.child_pool,
            shuffled_pool=self.shuffled_pool,
            update_steps=self.update_steps
            + (
                CSABankUpdateStepTrace(
                    candidate=candidate,
                    value=value,
                    bank_before=trace_bank_entries(bank_before.entries),
                    bank_after=trace_bank_entries(bank_after.entries),
                    distance_cutoff_before=distance_cutoff_before,
                    distance_cutoff_after=distance_cutoff_after,
                    changed_indices=changed_indices,
                    significant_update_indices=significant_update_indices,
                ),
            ),
        )

    def active_snapshot(self) -> CSAActiveGenerationTrace[CandidateT]:
        """Return the public active-generation trace snapshot."""
        return CSAActiveGenerationTrace(
            generation_index=self.generation_index,
            stage_index=self.stage_index,
            stage_round=self.stage_round,
            cycle_count_before=self.cycle_count_before,
            bank_before=self.bank_before,
            reference_before=self.reference_before,
            bank_status_before=self.bank_status_before,
            seed_mask=self.seed_mask,
            partner_mask=self.partner_mask,
            seed_batch=self.seed_batch,
            proposal_families_before=self.proposal_families_before,
            child_pool=self.child_pool,
            shuffled_pool=self.shuffled_pool,
            update_steps=self.update_steps,
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
    ) -> CSAGenerationTrace[CandidateT]:
        """Return the completed generation trace snapshot.

        Parameters
        ----------
        cycle_count : int
            Cycle counter value after the generation finishes.
        bank : Bank[CandidateT]
            Bank snapshot after generation execution.
        reference_bank : ReferenceBank[CandidateT]
            Reference-bank snapshot after generation execution.
        bank_status_after : tuple[bool, ...]
            Bank occupancy mask after generation execution.
        proposal_families_after : tuple[CSAProposalFamilyTrace, ...]
            Proposal-family weights observed after generation execution.
        pending_boundary_action_after : BoundaryActionName
            Pending run-boundary action after the generation.
        refresh_active_after : bool
            Whether refresh mode is active after the generation.

        Returns
        -------
        CSAGenerationTrace[CandidateT]
            Completed generation trace snapshot.
        """
        return CSAGenerationTrace(
            generation_index=self.generation_index,
            stage_index=self.stage_index,
            stage_round=self.stage_round,
            cycle_count_before=self.cycle_count_before,
            cycle_count_after=cycle_count,
            bank_before=self.bank_before,
            reference_before=self.reference_before,
            bank_after=trace_bank_entries(bank.entries),
            reference_after=trace_bank_entries(reference_bank.entries),
            bank_status_before=self.bank_status_before,
            bank_status_after=bank_status_after,
            seed_mask=self.seed_mask,
            partner_mask=self.partner_mask,
            seed_batch=self.seed_batch,
            proposal_families_before=self.proposal_families_before,
            proposal_families_after=proposal_families_after,
            child_pool=self.child_pool,
            shuffled_pool=self.shuffled_pool,
            update_steps=self.update_steps,
            pending_boundary_action_after=pending_boundary_action_after,
            refresh_active_after=refresh_active_after,
        )


@dataclass(frozen=True, slots=True)
class CSAEventTraceState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Immutable CSA parity trace state.

    Parameters
    ----------
    completed_generations : tuple[CSAGenerationTrace[CandidateT], ...], default=()
        Completed generation traces in execution order.
    active_generation : GenerationTraceState[CandidateT] | None, default=None
        In-flight generation trace, when one is currently being recorded.
    """

    completed_generations: tuple[CSAGenerationTrace[CandidateT], ...] = ()
    active_generation: GenerationTraceState[CandidateT] | None = None

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
    ) -> Self:
        """Return a trace state that starts one new generation.

        Parameters
        ----------
        stage_index : int
            Active CSA stage index.
        stage_round : int
            Number of completed rounds within the active stage.
        cycle_count : int
            Cycle counter value before the generation starts.
        bank : Bank[CandidateT]
            Bank snapshot before generation execution.
        reference_bank : ReferenceBank[CandidateT]
            Reference-bank snapshot before generation execution.
        bank_status_before : tuple[bool, ...]
            Bank occupancy mask before generation execution.
        seed_mask : frozenset[int]
            Seed indices active for the generation.
        partner_mask : frozenset[int]
            Partner indices active for the generation.
        seed_batch : tuple[int, ...]
            Active seed batch for the generation.
        proposal_families_before : tuple[CSAProposalFamilyTrace, ...]
            Proposal-family weights observed before child emission.

        Returns
        -------
        Self
            Trace state with a newly initialized active generation.

        Raises
        ------
        RuntimeError
            If another generation is already active.
        """
        if self.active_generation is not None:
            msg = "cannot start a new generation trace while another is active"
            raise RuntimeError(msg)

        return type(self)(
            completed_generations=self.completed_generations,
            active_generation=GenerationTraceState(
                generation_index=len(self.completed_generations),
                stage_index=stage_index,
                stage_round=stage_round,
                cycle_count_before=cycle_count,
                bank_before=trace_bank_entries(bank.entries),
                reference_before=trace_bank_entries(reference_bank.entries),
                bank_status_before=bank_status_before,
                seed_mask=tuple(sorted(seed_mask)),
                partner_mask=tuple(sorted(partner_mask)),
                seed_batch=seed_batch,
                proposal_families_before=proposal_families_before,
            ),
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
    ) -> Self:
        """Return a trace state with one emitted child appended.

        Parameters
        ----------
        family : CSAChildFamily
            Child family label for the emitted proposal.
        proposal_family_key : str | None
            Proposal-family key, when available.
        seed_index : int
            Seed index that generated the child.
        primary_source : CSAPrimarySource
            Primary source classification for the child.
        partner_indices : tuple[int, ...]
            Partner indices used while generating the child.
        candidate : CandidateT
            Emitted child candidate.

        Returns
        -------
        Self
            Trace state with the child appended to the active generation.
        """
        active_generation = self._require_active_generation()
        return type(self)(
            completed_generations=self.completed_generations,
            active_generation=active_generation.record_emitted_child(
                family=family,
                proposal_family_key=proposal_family_key,
                seed_index=seed_index,
                primary_source=primary_source,
                partner_indices=partner_indices,
                candidate=candidate,
            ),
        )

    def record_shuffled_pool(self, shuffled_pool: Sequence[CandidateT]) -> Self:
        """Return a trace state with one recorded shuffled pool.

        Parameters
        ----------
        shuffled_pool : Sequence[CandidateT]
            Child pool after generation-local shuffling.

        Returns
        -------
        Self
            Trace state with the shuffled pool recorded on the active
            generation.
        """
        active_generation = self._require_active_generation()
        return type(self)(
            completed_generations=self.completed_generations,
            active_generation=active_generation.record_shuffled_pool(shuffled_pool),
        )

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
    ) -> Self:
        """Return a trace state with one bank-update step appended.

        Parameters
        ----------
        candidate : CandidateT
            Candidate consumed by the bank update step.
        value : float
            Objective value associated with ``candidate``.
        bank_before : Bank[CandidateT]
            Bank snapshot before the update step.
        bank_after : Bank[CandidateT]
            Bank snapshot after the update step.
        distance_cutoff_before : float | None
            Distance cutoff before the update step, if initialized.
        distance_cutoff_after : float | None
            Distance cutoff after the update step, if initialized.
        changed_indices : frozenset[int]
            Bank indices changed by the update step.
        significant_update_indices : frozenset[int]
            Bank indices marked as significant updates.

        Returns
        -------
        Self
            Trace state with the bank update appended to the active generation.
        """
        active_generation = self._require_active_generation()
        return type(self)(
            completed_generations=self.completed_generations,
            active_generation=active_generation.record_bank_update_step(
                candidate=candidate,
                value=value,
                bank_before=bank_before,
                bank_after=bank_after,
                distance_cutoff_before=distance_cutoff_before,
                distance_cutoff_after=distance_cutoff_after,
                changed_indices=changed_indices,
                significant_update_indices=significant_update_indices,
            ),
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
    ) -> Self:
        """Return a trace state with the active generation completed.

        Parameters
        ----------
        cycle_count : int
            Cycle counter value after the generation finishes.
        bank : Bank[CandidateT]
            Bank snapshot after generation execution.
        reference_bank : ReferenceBank[CandidateT]
            Reference-bank snapshot after generation execution.
        bank_status_after : tuple[bool, ...]
            Bank occupancy mask after generation execution.
        proposal_families_after : tuple[CSAProposalFamilyTrace, ...]
            Proposal-family weights observed after generation execution.
        pending_boundary_action_after : BoundaryActionName
            Pending run-boundary action after the generation.
        refresh_active_after : bool
            Whether refresh mode is active after the generation.

        Returns
        -------
        Self
            Trace state with the active generation moved into the completed
            generation list.
        """
        active_generation = self._require_active_generation()
        completed_generation = active_generation.finish_generation(
            cycle_count=cycle_count,
            bank=bank,
            reference_bank=reference_bank,
            bank_status_after=bank_status_after,
            proposal_families_after=proposal_families_after,
            pending_boundary_action_after=pending_boundary_action_after,
            refresh_active_after=refresh_active_after,
        )
        return type(self)(
            completed_generations=self.completed_generations + (completed_generation,),
            active_generation=None,
        )

    def snapshot(self) -> tuple[CSAGenerationTrace[CandidateT], ...]:
        """Return the completed generation traces when no generation is active."""
        if self.active_generation is not None:
            msg = "cannot snapshot traces while a generation is still active"
            raise RuntimeError(msg)

        return self.completed_generations

    def completed_snapshot(self) -> tuple[CSAGenerationTrace[CandidateT], ...]:
        """Return the completed generation traces regardless of active state."""
        return self.completed_generations

    def active_snapshot(self) -> CSAActiveGenerationTrace[CandidateT] | None:
        """Return the in-flight generation trace snapshot, if any."""
        if self.active_generation is None:
            return None

        return self.active_generation.active_snapshot()

    def _require_active_generation(self) -> GenerationTraceState[CandidateT]:
        if self.active_generation is None:
            msg = "no active generation trace is available"
            raise RuntimeError(msg)

        return self.active_generation
