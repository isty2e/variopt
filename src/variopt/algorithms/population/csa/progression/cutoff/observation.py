"""Canonical runtime observations for CSA cutoff control."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real


def _require_count(value: int, *, field_name: str) -> None:
    if type(value) is not int:
        msg = f"{field_name} must be an integer"
        raise TypeError(msg)
    if value < 0:
        msg = f"{field_name} must be non-negative"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CSACutoffObservation:
    """Immutable post-update evidence for one cutoff transition.

    Parameters
    ----------
    score_gap : float | None
        Current finite non-negative bank score gap, when available.
    eligible_entry_count : int
        Number of current bank entries eligible for seed selection.
    unused_entry_count : int
        Number of eligible entries still marked unused.
    bank_entry_count : int
        Number of entries in the post-update bank.
    crowded_entry_count : int | None, optional
        Number of bank entries whose nearest neighbor lies inside the active
        cutoff. ``None`` means geometry was intentionally not materialized.
    cutoff_sensitive_transition_count : int, default=0
        Number of current-batch transitions routed through local, cluster, or
        far full-bank admission.
    local_transition_count : int, default=0
        Number of cutoff-sensitive transitions routed through local admission.
    """

    score_gap: float | None
    eligible_entry_count: int
    unused_entry_count: int
    bank_entry_count: int
    crowded_entry_count: int | None = None
    cutoff_sensitive_transition_count: int = 0
    local_transition_count: int = 0

    def __post_init__(self) -> None:
        """Reject malformed or non-finite cutoff observations."""
        for field_name, value in (
            ("eligible_entry_count", self.eligible_entry_count),
            ("unused_entry_count", self.unused_entry_count),
            ("bank_entry_count", self.bank_entry_count),
            (
                "cutoff_sensitive_transition_count",
                self.cutoff_sensitive_transition_count,
            ),
            ("local_transition_count", self.local_transition_count),
        ):
            _require_count(value, field_name=field_name)

        if self.eligible_entry_count > self.bank_entry_count:
            msg = "eligible_entry_count must not exceed bank_entry_count"
            raise ValueError(msg)
        if self.unused_entry_count > self.eligible_entry_count:
            msg = "unused_entry_count must not exceed eligible_entry_count"
            raise ValueError(msg)
        if self.local_transition_count > self.cutoff_sensitive_transition_count:
            msg = (
                "local_transition_count must not exceed "
                "cutoff_sensitive_transition_count"
            )
            raise ValueError(msg)

        crowded_entry_count = self.crowded_entry_count
        if crowded_entry_count is not None:
            _require_count(
                crowded_entry_count,
                field_name="crowded_entry_count",
            )
            if crowded_entry_count > self.bank_entry_count:
                msg = "crowded_entry_count must not exceed bank_entry_count"
                raise ValueError(msg)
            if self.bank_entry_count < 2 and crowded_entry_count != 0:
                msg = "a bank with fewer than two entries cannot be crowded"
                raise ValueError(msg)

        score_gap = self.score_gap
        if score_gap is None:
            return
        if type(score_gap) is bool or not isinstance(score_gap, Real):
            msg = "score_gap must be numeric"
            raise TypeError(msg)
        normalized_score_gap = float(score_gap)
        if not isfinite(normalized_score_gap) or normalized_score_gap < 0.0:
            msg = "score_gap must be a finite non-negative float"
            raise ValueError(msg)
        object.__setattr__(self, "score_gap", normalized_score_gap)

    @property
    def crowded_entry_fraction(self) -> float | None:
        """Return the fraction of bank entries with a near neighbor.

        Returns
        -------
        float | None
            Crowded-entry fraction, or ``None`` when geometry was not observed
            or the bank has fewer than two entries.
        """
        if self.crowded_entry_count is None or self.bank_entry_count < 2:
            return None
        return self.crowded_entry_count / self.bank_entry_count

    @property
    def local_route_fraction(self) -> float | None:
        """Return the local share of cutoff-sensitive admission routes.

        Returns
        -------
        float | None
            Local-route fraction, or ``None`` when the batch contains no
            cutoff-sensitive transitions.
        """
        if self.cutoff_sensitive_transition_count == 0:
            return None
        return self.local_transition_count / self.cutoff_sensitive_transition_count

    @property
    def used_entry_fraction(self) -> float | None:
        """Return normalized eligible-entry utilization, when defined.

        Returns
        -------
        float | None
            Fraction of eligible entries already used, or ``None`` when no
            entries are eligible.
        """
        if self.eligible_entry_count == 0:
            return None
        return 1.0 - self.unused_entry_count / self.eligible_entry_count
