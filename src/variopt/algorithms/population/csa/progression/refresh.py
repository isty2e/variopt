"""Boundary policy for CSA refresh and restart behavior."""

from dataclasses import dataclass
from typing import Literal

RefreshMode = Literal["legacy", "adaptive_refresh"]


@dataclass(frozen=True, slots=True)
class CSARefreshPolicy:
    """Boundary-level policy for CSA refresh payload construction.

    ``legacy`` preserves the current semantics. Plain refresh rebuilds the bank
    entirely from new samples, while staged-growth refresh keeps all previously
    existing entries.

    ``adaptive_refresh`` preserves a score-sorted elite fraction across refresh
    and refills the remainder with fresh samples. Plain refresh can optionally
    bias the next cycle toward newly added entries.

    Parameters
    ----------
    mode : RefreshMode, default="legacy"
        Refresh strategy to use.
    preserve_fraction : float, default=0.25
        Fraction of entries to preserve during adaptive refresh.
    newcomer_first_round : bool, default=True
        Whether plain refresh should bias the next round toward newcomers.
    """

    mode: RefreshMode = "legacy"
    preserve_fraction: float = 0.25
    newcomer_first_round: bool = True

    def __post_init__(self) -> None:
        """Reject invalid refresh-policy boundary settings."""
        if self.preserve_fraction < 0.0 or self.preserve_fraction > 1.0:
            msg = "preserve_fraction must lie in [0.0, 1.0]"
            raise ValueError(msg)

    def resolve_preserved_entry_count(
        self,
        *,
        entry_count: int,
        target_capacity: int,
    ) -> int:
        """Resolve how many entries survive an adaptive refresh.

        Parameters
        ----------
        entry_count : int
            Number of entries currently in the bank.
        target_capacity : int
            Target bank capacity after refresh.

        Returns
        -------
        int
            Number of entries to preserve before refilling the bank.
        """
        if self.mode == "legacy":
            return 0

        if entry_count <= 1 or target_capacity <= 1 or self.preserve_fraction <= 0.0:
            return 0

        max_preserved_entry_count = min(entry_count, target_capacity) - 1
        requested_entry_count = int(entry_count * self.preserve_fraction)
        if requested_entry_count == 0:
            requested_entry_count = 1

        return min(max_preserved_entry_count, requested_entry_count)
