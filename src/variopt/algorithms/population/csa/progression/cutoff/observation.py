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
    unused_entry_count : int
        Number of selectable bank entries still marked unused.
    full_bank_transition_count : int, default=0
        Number of current-batch transitions routed through local, cluster, or
        far full-bank decisions. Initial-bank and growth appends are excluded.
    local_transition_count : int, default=0
        Number of full-bank transitions routed through the local decision.
    """

    score_gap: float | None
    unused_entry_count: int
    full_bank_transition_count: int = 0
    local_transition_count: int = 0

    def __post_init__(self) -> None:
        """Reject malformed or non-finite cutoff observations."""
        for field_name, value in (
            ("unused_entry_count", self.unused_entry_count),
            ("full_bank_transition_count", self.full_bank_transition_count),
            ("local_transition_count", self.local_transition_count),
        ):
            _require_count(value, field_name=field_name)

        if self.local_transition_count > self.full_bank_transition_count:
            msg = "local_transition_count must not exceed full_bank_transition_count"
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
    def local_route_fraction(self) -> float | None:
        """Return the local share of current full-bank decision routes.

        Returns
        -------
        float | None
            Local-route fraction, or ``None`` when the batch contains no
            full-bank transitions.
        """
        if self.full_bank_transition_count == 0:
            return None
        return self.local_transition_count / self.full_bank_transition_count
