"""Private initial-crossover routing rules for CSA."""


def should_use_reference_primary(
    *,
    cycle_count: int,
    entry_count: int,
    active_seed_count: int,
    unused_entry_count: int,
    new_bank_cut: int,
) -> bool:
    """Return whether initial crossover should use the reference-bank seed.

    Parameters
    ----------
    cycle_count : int
        Current progression cycle count.
    entry_count : int
        Number of current bank entries.
    active_seed_count : int
        Number of active seed entries in the bank.
    unused_entry_count : int
        Number of entries currently marked unused.
    new_bank_cut : int
        Cutoff controlling when the reference-bank seed should be preferred.

    Returns
    -------
    bool
        Whether the routing logic should use the reference-bank seed as the
        primary crossover parent.

    Raises
    ------
    ValueError
        If the supplied bank counters are inconsistent.
    """
    if entry_count < 0:
        msg = "entry_count must be non-negative"
        raise ValueError(msg)

    if active_seed_count < 0 or active_seed_count > entry_count:
        msg = "active_seed_count must be in the interval [0, entry_count]"
        raise ValueError(msg)

    if unused_entry_count < 0 or unused_entry_count > entry_count:
        msg = "unused_entry_count must be in the interval [0, entry_count]"
        raise ValueError(msg)

    if new_bank_cut < 0:
        msg = "new_bank_cut must be non-negative"
        raise ValueError(msg)

    old_used_entry_count = entry_count - active_seed_count - unused_entry_count
    if old_used_entry_count < 0:
        msg = "active_seed_count and unused_entry_count must not exceed entry_count"
        raise ValueError(msg)

    return cycle_count == 0 and old_used_entry_count < new_bank_cut
