"""Immutable CSA trace artifact nouns."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Literal

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ......typevars import CandidateT
from ...banking.bank import BankEntry

CSAChildFamily = Literal["regular", "initial", "mutation"]
CSAPrimarySource = Literal["bank", "reference"]
BoundaryActionName = Literal["refresh", "stage_transition"] | None


@dataclass(frozen=True, slots=True)
class CSABankEntryTrace(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Immutable snapshot of one bank entry in a trace.

    Parameters
    ----------
    candidate : CandidateT
        Candidate stored in the bank entry.
    value : float
        Objective value stored in the bank entry.
    """

    candidate: CandidateT
    value: float


def trace_bank_entries(
    entries: Sequence[BankEntry[CandidateT]],
) -> tuple[CSABankEntryTrace[CandidateT], ...]:
    """Return immutable bank-entry snapshots for tracing.

    Parameters
    ----------
    entries : Sequence[BankEntry[CandidateT]]
        Bank entries to snapshot.

    Returns
    -------
    tuple[CSABankEntryTrace[CandidateT], ...]
        Immutable trace snapshots aligned with ``entries``.
    """
    return tuple(
        CSABankEntryTrace(
            candidate=entry.candidate,
            value=entry.value,
        )
        for entry in entries
    )


@dataclass(frozen=True, slots=True)
class CSAChildEmissionTrace(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Trace record for one emitted CSA child candidate.

    Parameters
    ----------
    family : CSAChildFamily
        Child-family label used to emit the candidate.
    proposal_family_key : str | None
        Optional proposal-family key associated with the emission.
    seed_index : int
        Seed index responsible for the child.
    primary_source : CSAPrimarySource
        Primary population source for the emitted child.
    partner_indices : tuple[int, ...]
        Partner indices consumed by the operator, if any.
    candidate : CandidateT
        Emitted child candidate.
    """

    family: CSAChildFamily
    proposal_family_key: str | None
    seed_index: int
    primary_source: CSAPrimarySource
    partner_indices: tuple[int, ...]
    candidate: CandidateT


@dataclass(frozen=True, slots=True)
class CSAProposalFamilyTrace:
    """One proposal-family telemetry snapshot attached to one generation trace.

    Parameters
    ----------
    family_key : str
        Stable proposal-family identifier.
    observation_count : int
        Number of observations attributed to the family.
    effective_score_credit : float
        Decayed score credit assigned to the family.
    mutation_weight : float | None
        Current mutation weight for the family, when applicable.
    """

    family_key: str
    observation_count: int
    effective_score_credit: float
    mutation_weight: float | None


@dataclass(frozen=True, slots=True)
class CSABankUpdateStepTrace(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Trace record for one shadow-bank admission step.

    Parameters
    ----------
    candidate : CandidateT
        Candidate evaluated against the shadow bank.
    value : float
        Objective value associated with ``candidate``.
    bank_before : tuple[CSABankEntryTrace[CandidateT], ...]
        Bank snapshot before the admission step.
    bank_after : tuple[CSABankEntryTrace[CandidateT], ...]
        Bank snapshot after the admission step.
    distance_cutoff_before : float | None
        Active cutoff before the admission step.
    distance_cutoff_after : float | None
        Active cutoff after the admission step.
    changed_indices : frozenset[int]
        Bank indices changed by the admission step.
    significant_update_indices : frozenset[int]
        Indices marked as significant updates by the banking logic.
    """

    candidate: CandidateT
    value: float
    bank_before: tuple[CSABankEntryTrace[CandidateT], ...]
    bank_after: tuple[CSABankEntryTrace[CandidateT], ...]
    distance_cutoff_before: float | None
    distance_cutoff_after: float | None
    changed_indices: frozenset[int]
    significant_update_indices: frozenset[int]


@dataclass(frozen=True, slots=True)
class CSAGenerationTrace(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Trace record for one full CSA child-pool generation.

    Parameters
    ----------
    generation_index : int
        Zero-based generation index within the trace.
    stage_index : int
        Active CSA stage index.
    stage_round : int
        Round count within the active stage.
    cycle_count_before : int
        Cycle count before generation execution.
    cycle_count_after : int
        Cycle count after generation execution.
    bank_before : tuple[CSABankEntryTrace[CandidateT], ...]
        Bank snapshot before generation execution.
    reference_before : tuple[CSABankEntryTrace[CandidateT], ...]
        Reference-bank snapshot before generation execution.
    bank_after : tuple[CSABankEntryTrace[CandidateT], ...]
        Bank snapshot after generation execution.
    reference_after : tuple[CSABankEntryTrace[CandidateT], ...]
        Reference-bank snapshot after generation execution.
    bank_status_before : tuple[bool, ...]
        Per-entry status flags before generation execution.
    bank_status_after : tuple[bool, ...]
        Per-entry status flags after generation execution.
    seed_mask : tuple[int, ...]
        Seed mask active for the generation.
    partner_mask : tuple[int, ...]
        Partner mask active for the generation.
    seed_batch : tuple[int, ...]
        Concrete seed indices used for the generation.
    proposal_families_before : tuple[CSAProposalFamilyTrace, ...]
        Proposal-family telemetry before generation execution.
    proposal_families_after : tuple[CSAProposalFamilyTrace, ...]
        Proposal-family telemetry after generation execution.
    child_pool : tuple[CSAChildEmissionTrace[CandidateT], ...]
        Emitted child candidates in emission order.
    shuffled_pool : tuple[CandidateT, ...]
        Child pool after generation-level shuffling.
    update_steps : tuple[CSABankUpdateStepTrace[CandidateT], ...]
        Shadow-bank update trace steps.
    pending_boundary_action_after : BoundaryActionName
        Boundary action queued after generation execution.
    refresh_active_after : bool
        Whether refresh mode remains active after generation execution.
    """

    generation_index: int
    stage_index: int
    stage_round: int
    cycle_count_before: int
    cycle_count_after: int
    bank_before: tuple[CSABankEntryTrace[CandidateT], ...]
    reference_before: tuple[CSABankEntryTrace[CandidateT], ...]
    bank_after: tuple[CSABankEntryTrace[CandidateT], ...]
    reference_after: tuple[CSABankEntryTrace[CandidateT], ...]
    bank_status_before: tuple[bool, ...]
    bank_status_after: tuple[bool, ...]
    seed_mask: tuple[int, ...]
    partner_mask: tuple[int, ...]
    seed_batch: tuple[int, ...]
    proposal_families_before: tuple[CSAProposalFamilyTrace, ...]
    proposal_families_after: tuple[CSAProposalFamilyTrace, ...]
    child_pool: tuple[CSAChildEmissionTrace[CandidateT], ...]
    shuffled_pool: tuple[CandidateT, ...]
    update_steps: tuple[CSABankUpdateStepTrace[CandidateT], ...]
    pending_boundary_action_after: BoundaryActionName
    refresh_active_after: bool


@dataclass(frozen=True, slots=True)
class CSAActiveGenerationTrace(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Trace record for an in-flight CSA generation when a run stops mid-pool.

    Parameters
    ----------
    generation_index : int
        Zero-based generation index within the trace.
    stage_index : int
        Active CSA stage index.
    stage_round : int
        Round count within the active stage.
    cycle_count_before : int
        Cycle count before generation execution.
    bank_before : tuple[CSABankEntryTrace[CandidateT], ...]
        Bank snapshot before generation execution.
    reference_before : tuple[CSABankEntryTrace[CandidateT], ...]
        Reference-bank snapshot before generation execution.
    bank_status_before : tuple[bool, ...]
        Per-entry status flags before generation execution.
    seed_mask : tuple[int, ...]
        Seed mask active for the generation.
    partner_mask : tuple[int, ...]
        Partner mask active for the generation.
    seed_batch : tuple[int, ...]
        Concrete seed indices used for the generation.
    proposal_families_before : tuple[CSAProposalFamilyTrace, ...]
        Proposal-family telemetry before generation execution.
    child_pool : tuple[CSAChildEmissionTrace[CandidateT], ...]
        Emitted child candidates in emission order.
    shuffled_pool : tuple[CandidateT, ...]
        Child pool after generation-level shuffling.
    update_steps : tuple[CSABankUpdateStepTrace[CandidateT], ...]
        Shadow-bank update trace steps completed so far.
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
    child_pool: tuple[CSAChildEmissionTrace[CandidateT], ...]
    shuffled_pool: tuple[CandidateT, ...]
    update_steps: tuple[CSABankUpdateStepTrace[CandidateT], ...]
