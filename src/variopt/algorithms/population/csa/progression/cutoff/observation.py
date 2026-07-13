"""Canonical runtime observations for CSA cutoff control."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real


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
    pairwise_distances : tuple[float, ...] | None, optional
        Validated current-bank pair distances when requested by the configured
        cutoff schedule. ``None`` means the distances were intentionally not
        materialized; an empty tuple is an observed bank with fewer than two
        entries.
    """

    score_gap: float | None
    eligible_entry_count: int
    unused_entry_count: int
    pairwise_distances: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Reject malformed or non-finite cutoff observations."""
        if type(self.eligible_entry_count) is not int:
            msg = "eligible_entry_count must be an integer"
            raise TypeError(msg)
        if type(self.unused_entry_count) is not int:
            msg = "unused_entry_count must be an integer"
            raise TypeError(msg)
        if self.eligible_entry_count < 0:
            msg = "eligible_entry_count must be non-negative"
            raise ValueError(msg)
        if not 0 <= self.unused_entry_count <= self.eligible_entry_count:
            msg = "unused_entry_count must not exceed eligible_entry_count"
            raise ValueError(msg)

        score_gap = self.score_gap
        if score_gap is not None:
            if type(score_gap) is bool or not isinstance(score_gap, Real):
                msg = "score_gap must be numeric"
                raise TypeError(msg)
            score_gap = float(score_gap)
            if not isfinite(score_gap) or score_gap < 0.0:
                msg = "score_gap must be a finite non-negative float"
                raise ValueError(msg)
            object.__setattr__(self, "score_gap", score_gap)

        pairwise_distances = self.pairwise_distances
        if pairwise_distances is None:
            return

        normalized_distances: list[float] = []
        for distance in pairwise_distances:
            if type(distance) is bool or not isinstance(distance, Real):
                msg = "pairwise_distances must contain numeric values"
                raise TypeError(msg)
            normalized_distance = float(distance)
            if not isfinite(normalized_distance) or normalized_distance < 0.0:
                msg = "pairwise_distances must contain finite non-negative floats"
                raise ValueError(msg)
            normalized_distances.append(normalized_distance)
        object.__setattr__(self, "pairwise_distances", tuple(normalized_distances))

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
