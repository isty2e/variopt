"""Private seed-selection state for the CSA optimizer."""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from typing_extensions import Self

from .....json_types import (
    JSONDict,
    JSONValue,
    require_json_bool,
    require_json_int,
    require_json_list,
)
from ..indexing import remap_indices_after_removal

EMPTY_IGNORED_INDICES: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class SeedSelectionState:
    """Immutable seed-batch state for CSA proposal generation.

    Parameters
    ----------
    used_entry_indices : frozenset[int], default=frozenset()
        Bank indices already marked as used by prior seed batches.
    bank_status : tuple[bool, ...], default=()
        Per-entry used flags aligned with the current bank snapshot.
    active_seed_indices : tuple[int, ...], default=()
        Currently active seed batch.
    next_seed_offset : int, default=0
        Offset of the next seed to consume from ``active_seed_indices``.
    """

    used_entry_indices: frozenset[int] = field(default_factory=frozenset)
    bank_status: tuple[bool, ...] = ()
    active_seed_indices: tuple[int, ...] = ()
    next_seed_offset: int = 0

    def __post_init__(self) -> None:
        """Reject invalid active-seed offsets."""
        if self.next_seed_offset < 0:
            msg = "next_seed_offset must be non-negative"
            raise ValueError(msg)

        if self.next_seed_offset > len(self.active_seed_indices):
            msg = "next_seed_offset must not exceed the active seed count"
            raise ValueError(msg)

    @property
    def has_active_seed(self) -> bool:
        """Return whether there is an unconsumed active seed."""
        return self.next_seed_offset < len(self.active_seed_indices)

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the seed-selection state.

        Returns
        -------
        JSONDict
            JSON-safe seed-selection snapshot.
        """
        return {
            "used_entry_indices": list(self.used_entry_indices),
            "bank_status": list(self.bank_status),
            "active_seed_indices": list(self.active_seed_indices),
            "next_seed_offset": self.next_seed_offset,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
    ) -> Self:
        """Build a seed-selection state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe seed-selection snapshot.

        Returns
        -------
        Self
            Reconstructed seed-selection state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        raw_used_entry_indices = require_json_list(
            data.get("used_entry_indices"),
            field_name="used_entry_indices",
        )
        raw_bank_status = require_json_list(
            data.get("bank_status"),
            field_name="bank_status",
        )
        raw_active_seed_indices = require_json_list(
            data.get("active_seed_indices"),
            field_name="active_seed_indices",
        )
        next_seed_offset = require_json_int(
            data.get("next_seed_offset"),
            field_name="next_seed_offset",
        )
        used_entry_indices: list[int] = []
        for raw_position, raw_index in enumerate(raw_used_entry_indices):
            used_entry_indices.append(
                require_json_int(
                    raw_index,
                    field_name=f"used_entry_indices[{raw_position}]",
                ),
            )

        bank_status: list[bool] = []
        for raw_position, raw_is_used in enumerate(raw_bank_status):
            bank_status.append(
                require_json_bool(
                    raw_is_used,
                    field_name=f"bank_status[{raw_position}]",
                ),
            )

        active_seed_indices: list[int] = []
        for raw_position, raw_index in enumerate(raw_active_seed_indices):
            active_seed_indices.append(
                require_json_int(
                    raw_index,
                    field_name=f"active_seed_indices[{raw_position}]",
                ),
            )

        return cls(
            used_entry_indices=frozenset(used_entry_indices),
            bank_status=tuple(bank_status),
            active_seed_indices=tuple(active_seed_indices),
            next_seed_offset=next_seed_offset,
        )

    def count_unused_entries(
        self,
        *,
        entry_count: int,
        ignored_indices: frozenset[int] = EMPTY_IGNORED_INDICES,
    ) -> int:
        """Return the number of bank entries still marked unused.

        Parameters
        ----------
        entry_count : int
            Current bank size.
        ignored_indices : frozenset[int], default=EMPTY_IGNORED_INDICES
            Indices to exclude from the unused-count calculation.

        Returns
        -------
        int
            Number of entries currently available for seed selection.
        """
        status = self.resize_bank_status(entry_count=entry_count)
        return sum(
            1
            for index, is_used in enumerate(status)
            if index not in ignored_indices and not is_used
        )

    def consume_seed(self) -> tuple[int, Self]:
        """Return the next seed index and the advanced state."""
        if not self.has_active_seed:
            msg = "cannot consume a seed when no active seed is available"
            raise ValueError(msg)

        seed_index = self.active_seed_indices[self.next_seed_offset]
        return seed_index, type(self)(
            used_entry_indices=self.used_entry_indices,
            bank_status=self.bank_status,
            active_seed_indices=self.active_seed_indices,
            next_seed_offset=self.next_seed_offset + 1,
        )

    def invalidate_for_bank_update(
        self,
        *,
        updated_indices: AbstractSet[int],
        entry_count: int,
    ) -> Self:
        """Return a state adapted to the updated bank contents.

        Parameters
        ----------
        updated_indices : collections.abc.Set[int]
            Bank indices updated in place.
        entry_count : int
            Current bank size after the update.

        Returns
        -------
        Self
            Seed-selection state with invalidated indices cleared and active
            seeds reset.
        """
        retained_used_indices = frozenset(
            index
            for index in self.used_entry_indices
            if 0 <= index < entry_count and index not in updated_indices
        )
        bank_status = list(self.resize_bank_status(entry_count=entry_count))
        for index in updated_indices:
            if 0 <= index < entry_count:
                bank_status[index] = False
        return type(self)(
            used_entry_indices=retained_used_indices,
            bank_status=tuple(bank_status),
            active_seed_indices=(),
            next_seed_offset=0,
        )

    def activate_seed_batch(
        self,
        *,
        selected_seed_indices: tuple[int, ...],
        bank_status: tuple[bool, ...],
        used_entry_indices: frozenset[int],
    ) -> Self:
        """Return a state with a new active seed batch.

        Parameters
        ----------
        selected_seed_indices : tuple[int, ...]
            Seed indices selected for the next generation batch.
        bank_status : tuple[bool, ...]
            Bank-status flags aligned with the current bank snapshot.
        used_entry_indices : frozenset[int]
            Bank indices already considered used.

        Returns
        -------
        Self
            Seed-selection state with the new active seed batch installed.
        """
        return type(self)(
            used_entry_indices=used_entry_indices,
            bank_status=bank_status,
            active_seed_indices=selected_seed_indices,
            next_seed_offset=0,
        )

    def reset_bank_status(self, *, entry_count: int) -> Self:
        """Return a copy whose bank-status flags are all reset to unused.

        Parameters
        ----------
        entry_count : int
            Current bank size.

        Returns
        -------
        Self
            Seed-selection state with reset bank-status flags.
        """
        return type(self)(
            used_entry_indices=self.used_entry_indices,
            bank_status=tuple(False for _ in range(entry_count)),
            active_seed_indices=self.active_seed_indices,
            next_seed_offset=self.next_seed_offset,
        )

    def remove_indices(
        self,
        *,
        removed_indices: AbstractSet[int],
        entry_count: int,
    ) -> Self:
        """Return a copy remapped after bank indices have been removed.

        Parameters
        ----------
        removed_indices : collections.abc.Set[int]
            Bank indices removed from the bank snapshot.
        entry_count : int
            Current bank size after removal.

        Returns
        -------
        Self
            Seed-selection state remapped to the new bank indexing.
        """
        if not removed_indices:
            return self

        remapped_used_indices = remap_indices_after_removal(
            self.used_entry_indices,
            removed_indices=removed_indices,
            entry_count=entry_count,
        )
        removed_index_set = frozenset(index for index in removed_indices if index >= 0)
        resized_status = [
            is_used
            for index, is_used in enumerate(self.resize_bank_status(entry_count=entry_count + len(removed_index_set)))
            if index not in removed_index_set
        ]
        return type(self)(
            used_entry_indices=remapped_used_indices,
            bank_status=tuple(resized_status[:entry_count]),
            active_seed_indices=(),
            next_seed_offset=0,
        )

    def resize_bank_status(self, *, entry_count: int) -> tuple[bool, ...]:
        """Return bank-status flags resized to match the current bank.

        Parameters
        ----------
        entry_count : int
            Current bank size.

        Returns
        -------
        tuple[bool, ...]
            Bank-status flags resized to ``entry_count``.

        Raises
        ------
        ValueError
            If ``entry_count`` is negative.
        """
        if entry_count < 0:
            msg = "entry_count must be non-negative"
            raise ValueError(msg)

        resized_status = list(self.bank_status[:entry_count])
        if len(resized_status) < entry_count:
            resized_status.extend(False for _ in range(entry_count - len(resized_status)))
        return tuple(resized_status)
