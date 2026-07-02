"""Private reference-bank and refresh-pool state for CSA."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....artifacts import Observation
from .....json_types import JSONDict, JSONValue, require_json_bool, require_json_int
from .....typevars import CandidateT
from .bank import Bank, BankEntry


@dataclass(frozen=True, slots=True)
class ReferenceBank(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Persistent reference bank used by legacy-style initial crossover.

    Parameters
    ----------
    capacity : int
        Maximum number of entries the reference bank may store.
    entries : tuple[BankEntry[CandidateT], ...], optional
        Entries currently stored in the reference bank.
    initialized : bool, default=False
        Whether the entries represent a completed reference snapshot.
    """

    capacity: int
    entries: tuple[BankEntry[CandidateT], ...] = ()
    initialized: bool = False

    def __post_init__(self) -> None:
        """Validate reference-bank state.

        Raises
        ------
        ValueError
            Raised when capacity is invalid or the entry count exceeds it.
        """
        if self.capacity <= 0:
            msg = "capacity must be positive"
            raise ValueError(msg)

        if len(self.entries) > self.capacity:
            msg = "entries must not exceed capacity"
            raise ValueError(msg)

        if self.initialized and not self.is_full:
            msg = "initialized reference banks must be full"
            raise ValueError(msg)

        if not self.initialized and self.is_full:
            object.__setattr__(self, "initialized", True)

    @property
    def is_full(self) -> bool:
        """Return whether the reference bank has reached capacity.

        Returns
        -------
        bool
            ``True`` when the reference bank is full.
        """
        return len(self.entries) >= self.capacity

    def to_dict(
        self,
        *,
        candidate_to_dict: Callable[[CandidateT], JSONValue],
    ) -> JSONDict:
        """Return a JSON-safe mapping for the reference bank.

        Parameters
        ----------
        candidate_to_dict : Callable[[CandidateT], JSONValue]
            Callback that converts canonical candidates into JSON-safe values.

        Returns
        -------
        JSONDict
            JSON-safe reference-bank snapshot.
        """
        return {
            "capacity": self.capacity,
            "entries": [
                entry.to_dict(candidate_to_dict=candidate_to_dict)
                for entry in self.entries
            ],
            "initialized": self.initialized,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        candidate_from_dict: Callable[[JSONValue], CandidateT],
    ) -> "ReferenceBank[CandidateT]":
        """Build a reference bank from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe reference-bank snapshot.
        candidate_from_dict : Callable[[JSONValue], CandidateT]
            Callback that reconstructs canonical candidates from JSON-safe
            values.

        Returns
        -------
        ReferenceBank[CandidateT]
            Reconstructed reference-bank snapshot.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        capacity = require_json_int(data.get("capacity"), field_name="capacity")
        raw_entries = data.get("entries")
        raw_initialized = data.get("initialized")
        if not isinstance(raw_entries, list):
            msg = "reference bank snapshot requires entry list"
            raise TypeError(msg)
        initialized = (
            None
            if raw_initialized is None
            else require_json_bool(
                raw_initialized,
                field_name="reference bank initialized",
            )
        )

        entries: list[BankEntry[CandidateT]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                msg = "reference bank snapshot entries must be mappings"
                raise TypeError(msg)
            entries.append(
                BankEntry[CandidateT].from_dict(
                    raw_entry,
                    candidate_from_dict=candidate_from_dict,
                ),
            )

        return cls(
            capacity=capacity,
            entries=tuple(entries),
            initialized=(
                len(entries) >= capacity
                if initialized is None
                else initialized
            ),
        )


@dataclass(frozen=True, slots=True)
class ReferenceRefreshState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Refresh-time candidate pool used to rebuild bank and reference bank.

    Parameters
    ----------
    target_capacity : int
        Target capacity of the refreshed bank and reference bank.
    preserved_bank_entries : tuple[BankEntry[CandidateT], ...], optional
        Entries preserved directly in the refreshed current bank.
    preserved_reference_entries : tuple[BankEntry[CandidateT], ...], optional
        Entries preserved directly in the refreshed reference bank.
    pool_entries : tuple[BankEntry[CandidateT], ...], optional
        Newly sampled entries accumulated during refresh.
    """

    target_capacity: int
    preserved_bank_entries: tuple[BankEntry[CandidateT], ...] = ()
    preserved_reference_entries: tuple[BankEntry[CandidateT], ...] = ()
    pool_entries: tuple[BankEntry[CandidateT], ...] = ()

    def __post_init__(self) -> None:
        """Validate refresh state payloads.

        Raises
        ------
        ValueError
            Raised when target capacity or preserved-entry alignment is
            invalid.
        """
        if self.target_capacity <= 0:
            msg = "target_capacity must be positive"
            raise ValueError(msg)

        if len(self.preserved_bank_entries) != len(self.preserved_reference_entries):
            msg = "preserved bank and reference entries must have the same length"
            raise ValueError(msg)

        if len(self.preserved_bank_entries) > self.target_capacity:
            msg = "preserved entries must not exceed the target capacity"
            raise ValueError(msg)

    @property
    def required_pool_entry_count(self) -> int:
        """Return the number of newly sampled entries still needed.

        Returns
        -------
        int
            Remaining number of pool entries required to rebuild the target
            capacity.
        """
        return self.target_capacity - len(self.preserved_reference_entries)

    @property
    def has_enough_entries(self) -> bool:
        """Return whether the refresh pool can rebuild the target bank.

        Returns
        -------
        bool
            ``True`` when the accumulated pool is large enough to rebuild the
            target bank.
        """
        return len(self.pool_entries) >= self.required_pool_entry_count

    def append_observation(self, observation: Observation[CandidateT]) -> Self:
        """Append one refresh observation to the pool.

        Parameters
        ----------
        observation : Observation[CandidateT]
            Observation to append to the refresh pool.

        Returns
        -------
        Self
            Updated refresh state with the new pooled entry appended.
        """
        return type(self)(
            target_capacity=self.target_capacity,
            preserved_bank_entries=self.preserved_bank_entries,
            preserved_reference_entries=self.preserved_reference_entries,
            pool_entries=self.pool_entries
            + (
                BankEntry(
                    candidate=observation.candidate,
                    value=observation.score,
                    proposal_id=observation.proposal.proposal_id,
                ),
            ),
        )

    def build_reference_bank(self) -> ReferenceBank[CandidateT]:
        """Build the refreshed reference bank from the accumulated pool.

        Returns
        -------
        ReferenceBank[CandidateT]
            Refreshed reference bank.

        Raises
        ------
        ValueError
            Raised when the refresh pool does not yet contain enough entries.
        """
        if not self.has_enough_entries:
            msg = "refresh pool does not contain enough entries"
            raise ValueError(msg)

        return build_reference_bank_from_refresh_pool(
            capacity=self.target_capacity,
            preserved_entries=self.preserved_reference_entries,
            pool_entries=self.pool_entries,
        )

    def build_bank(self) -> Bank[CandidateT]:
        """Build the refreshed current bank from the accumulated pool.

        Returns
        -------
        Bank[CandidateT]
            Refreshed current bank.
        """
        reference_bank = self.build_reference_bank()
        preserved_reference_count = len(self.preserved_reference_entries)
        appended_entries = reference_bank.entries[preserved_reference_count:]
        return Bank(
            capacity=self.target_capacity,
            entries=self.preserved_bank_entries + appended_entries,
        )


def build_reference_bank_from_bank(
    bank: Bank[CandidateT],
) -> ReferenceBank[CandidateT]:
    """Build a reference bank from a current bank snapshot.

    Parameters
    ----------
    bank : Bank[CandidateT]
        Current bank snapshot whose entries are copied into reference order.

    Returns
    -------
    ReferenceBank[CandidateT]
        Reference bank sorted by value.
    """
    sorted_entries = sort_entries_by_value(bank.entries)
    return ReferenceBank(
        capacity=bank.capacity,
        entries=sorted_entries,
        initialized=True,
    )


def build_sorted_bank_from_bank(
    bank: Bank[CandidateT],
) -> Bank[CandidateT]:
    """Build a bank whose entries are ordered by score.

    Parameters
    ----------
    bank : Bank[CandidateT]
        Bank snapshot whose entries are sorted.

    Returns
    -------
    Bank[CandidateT]
        Bank with entries sorted by objective value.
    """
    return Bank(
        capacity=bank.capacity,
        entries=sort_entries_by_value(bank.entries),
    )


def build_reference_bank_from_refresh_pool(
    *,
    capacity: int,
    preserved_entries: Sequence[BankEntry[CandidateT]] = (),
    pool_entries: Sequence[BankEntry[CandidateT]],
) -> ReferenceBank[CandidateT]:
    """Build a reference bank by appending picked pool entries.

    Parameters
    ----------
    capacity : int
        Target reference-bank capacity.
    preserved_entries : Sequence[BankEntry[CandidateT]], optional
        Entries preserved directly in the reference bank.
    pool_entries : Sequence[BankEntry[CandidateT]]
        Candidate pool from which new reference entries are chosen.

    Returns
    -------
    ReferenceBank[CandidateT]
        Rebuilt reference bank.

    Raises
    ------
    ValueError
        Raised when the preserved entries exceed the requested capacity.
    """
    preserved_entry_tuple = tuple(preserved_entries)
    if len(preserved_entry_tuple) > capacity:
        msg = "preserved entries must not exceed capacity"
        raise ValueError(msg)

    selection_count = capacity - len(preserved_entry_tuple)
    selected_entries = pick_reference_entries(
        pool_entries=pool_entries,
        selection_count=selection_count,
    )
    return ReferenceBank(
        capacity=capacity,
        entries=preserved_entry_tuple + selected_entries,
        initialized=True,
    )


def pick_reference_entries(
    *,
    pool_entries: Sequence[BankEntry[CandidateT]],
    selection_count: int,
) -> tuple[BankEntry[CandidateT], ...]:
    """Pick the best-scoring entries from a refresh pool.

    Parameters
    ----------
    pool_entries : Sequence[BankEntry[CandidateT]]
        Candidate pool from which entries are selected.
    selection_count : int
        Number of entries to select.

    Returns
    -------
    tuple[BankEntry[CandidateT], ...]
        Best-scoring selected entries.

    Raises
    ------
    ValueError
        Raised when the selection count is negative or exceeds the pool size.
    """
    if selection_count < 0:
        msg = "selection_count must not be negative"
        raise ValueError(msg)

    if selection_count > len(pool_entries):
        msg = "selection_count must not exceed the number of pool entries"
        raise ValueError(msg)

    sorted_entries = sort_entries_by_value(pool_entries)
    return tuple(sorted_entries[:selection_count])


def sort_entries_by_value(
    entries: Sequence[BankEntry[CandidateT]],
) -> tuple[BankEntry[CandidateT], ...]:
    """Sort entries by objective value.

    Parameters
    ----------
    entries : Sequence[BankEntry[CandidateT]]
        Entries to sort.

    Returns
    -------
    tuple[BankEntry[CandidateT], ...]
        Entries sorted by ascending objective value.
    """
    return tuple(sorted(entries, key=lambda entry: entry.value))
