"""CSA bank-update results and delta summaries."""

from dataclasses import dataclass
from typing import Generic

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ......typevars import CandidateT
from ...progression.state import CSAProgressionState
from ...scoring.model_state import CSAScoreModelState
from ...trace.events.state import CSAEventTraceState
from ..bank import Bank
from ..clustering import CSAClusteringState
from ..growth import CSABankGrowthState


@dataclass(frozen=True, slots=True)
class BankUpdateResult(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Result of reducing one observation batch against a CSA shadow bank.

    Parameters
    ----------
    bank : Bank[CandidateT]
        Updated bank after reducing the observation batch.
    state : CSAProgressionState
        Updated progression state.
    score_model_state : CSAScoreModelState[CandidateT]
        Updated score-model state.
    growth_state : CSABankGrowthState[CandidateT]
        Updated bank-growth state.
    clustering_state : CSAClusteringState[CandidateT]
        Updated clustering state.
    trace_state : CSAEventTraceState[CandidateT] | None
        Optional updated trace reducer state.
    changed_indices : frozenset[int]
        Bank indices whose entries changed before post-batch removals are applied.
    significant_update_indices : frozenset[int]
        Changed bank indices whose score gap exceeded the significance floor
        before post-batch removals are applied.
    removed_indices : frozenset[int]
        Bank indices removed from the previous bank snapshot.
    """

    bank: Bank[CandidateT]
    state: CSAProgressionState
    score_model_state: CSAScoreModelState[CandidateT]
    growth_state: CSABankGrowthState[CandidateT]
    clustering_state: CSAClusteringState[CandidateT]
    trace_state: CSAEventTraceState[CandidateT] | None
    changed_indices: frozenset[int]
    significant_update_indices: frozenset[int]
    removed_indices: frozenset[int]


def changed_indices(
    *,
    previous_bank: Bank[CandidateT],
    next_bank: Bank[CandidateT],
) -> frozenset[int]:
    """Return all bank indices whose entries changed at all.

    Parameters
    ----------
    previous_bank : Bank[CandidateT]
        Bank snapshot before reduction.
    next_bank : Bank[CandidateT]
        Bank snapshot after reduction.

    Returns
    -------
    frozenset[int]
        Indices whose entries changed or were appended.
    """
    updated_indices: set[int] = set()
    common_entry_count = min(
        len(previous_bank.entries),
        len(next_bank.entries),
    )
    for index in range(common_entry_count):
        if previous_bank.entries[index] != next_bank.entries[index]:
            updated_indices.add(index)

    for index in range(common_entry_count, len(next_bank.entries)):
        updated_indices.add(index)

    return frozenset(updated_indices)


def significant_update_indices(
    *,
    previous_bank: Bank[CandidateT],
    next_bank: Bank[CandidateT],
    minimum_significant_score_gap: float,
) -> frozenset[int]:
    """Return changed indices whose score gap exceeds the significance floor.

    Parameters
    ----------
    previous_bank : Bank[CandidateT]
        Bank snapshot before reduction.
    next_bank : Bank[CandidateT]
        Bank snapshot after reduction.
    minimum_significant_score_gap : float
        Minimum absolute score delta required to mark a change as significant.

    Returns
    -------
    frozenset[int]
        Significant updated indices.
    """
    updated_indices: set[int] = set()
    common_entry_count = min(
        len(previous_bank.entries),
        len(next_bank.entries),
    )
    for index in range(common_entry_count):
        if previous_bank.entries[index] != next_bank.entries[index]:
            score_delta = abs(
                previous_bank.entries[index].value - next_bank.entries[index].value
            )
            if score_delta > minimum_significant_score_gap:
                updated_indices.add(index)

    for index in range(common_entry_count, len(next_bank.entries)):
        updated_indices.add(index)

    return frozenset(updated_indices)
