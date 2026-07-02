"""Bank-index algebra shared by CSA state projections."""

from bisect import bisect_left
from collections.abc import Set as AbstractSet


def remap_indices_after_removal(
    indices: AbstractSet[int],
    *,
    removed_indices: AbstractSet[int],
    entry_count: int,
) -> frozenset[int]:
    """Return indices remapped after removing entries from a bank snapshot.

    Parameters
    ----------
    indices : collections.abc.Set[int]
        Indices aligned to the previous bank snapshot.
    removed_indices : collections.abc.Set[int]
        Bank indices removed from the previous bank snapshot. Negative values
        are ignored.
    entry_count : int
        Current bank size after removal.

    Returns
    -------
    frozenset[int]
        Indices aligned to the current bank snapshot. Removed indices and
        indices outside the current bank size are dropped.

    Raises
    ------
    ValueError
        If ``entry_count`` is negative.
    """
    if entry_count < 0:
        msg = "entry_count must be non-negative"
        raise ValueError(msg)

    if not indices:
        return frozenset()

    ordered_removed_indices = tuple(
        sorted(index for index in removed_indices if index >= 0)
    )
    removed_index_set = frozenset(ordered_removed_indices)
    remapped_indices: set[int] = set()
    for index in indices:
        if index < 0 or index in removed_index_set:
            continue

        removed_before_count = bisect_left(ordered_removed_indices, index)
        remapped_index = index - removed_before_count
        if remapped_index < entry_count:
            remapped_indices.add(remapped_index)

    return frozenset(remapped_indices)
