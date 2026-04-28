"""Private bank query primitives shared by CSA bank update components."""

from collections.abc import Sequence
from typing import Protocol, TypeVar

from .....distance import require_valid_distance
from .....diversity import DiversityMetric
from .....typevars import CandidateT
from .update.policy import CSANicheQualityPolicy

EntryCandidateT = TypeVar("EntryCandidateT", covariant=True)


class CandidateEntry(Protocol[EntryCandidateT]):
    """Minimal entry view needed by bank update queries.

    Notes
    -----
    Bank-query helpers depend only on candidate/value access, so this protocol
    keeps them reusable across concrete entry types.
    """

    @property
    def candidate(self) -> EntryCandidateT:
        """Return the candidate stored by the entry.

        Returns
        -------
        EntryCandidateT
            Candidate carried by the entry.
        """
        ...

    @property
    def value(self) -> float:
        """Return the objective value stored by the entry.

        Returns
        -------
        float
            Objective value associated with the entry.
        """
        ...


def nearest_entry(
    *,
    entries: Sequence[CandidateEntry[CandidateT]],
    candidate: CandidateT,
    diversity_metric: DiversityMetric[CandidateT],
) -> tuple[int, float | None]:
    """Locate the nearest entry to a candidate.

    Parameters
    ----------
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries searched for the nearest neighbor.
    candidate : CandidateT
        Candidate whose nearest entry is requested.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute distances.

    Returns
    -------
    tuple[int, float | None]
        Index of the nearest entry and its distance, or ``(-1, None)`` when
        ``entries`` is empty.
    """
    nearest_index = -1
    nearest_distance: float | None = None

    for index, entry in enumerate(entries):
        distance = require_valid_distance(
            diversity_metric.distance(candidate, entry.candidate)
        )
        if nearest_distance is None or distance < nearest_distance:
            nearest_index = index
            nearest_distance = distance

    return nearest_index, nearest_distance


def worst_index(
    entries: Sequence[CandidateEntry[CandidateT]],
    candidate_indices: Sequence[int] | None = None,
) -> int:
    """Locate the worst objective-value entry index.

    Parameters
    ----------
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries searched for the worst value.
    candidate_indices : Sequence[int] | None, default=None
        Optional subset of indices to consider. ``None`` considers all
        entries.

    Returns
    -------
    int
        Index whose entry carries the largest objective value.

    Raises
    ------
    ValueError
        Raised when ``candidate_indices`` is empty.
    """
    if candidate_indices is None:
        candidate_indices = tuple(range(len(entries)))

    if len(candidate_indices) == 0:
        msg = "candidate_indices must not be empty"
        raise ValueError(msg)

    current_worst_index = candidate_indices[0]
    for index in candidate_indices[1:]:
        entry = entries[index]
        if entry.value > entries[current_worst_index].value:
            current_worst_index = index

    return current_worst_index


def crowded_indices(
    *,
    entries: Sequence[CandidateEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
) -> frozenset[int]:
    """Return entry indices that already have a near neighbor inside the bank.

    Parameters
    ----------
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries whose crowding state is measured.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute pairwise distances.
    distance_cutoff : float
        Distance threshold below which two entries are considered neighbors.

    Returns
    -------
    frozenset[int]
        Entry indices that have at least one near neighbor.
    """
    return frozenset(
        index
        for index, count in enumerate(
            crowding_counts(
                entries=entries,
                diversity_metric=diversity_metric,
                distance_cutoff=distance_cutoff,
            )
        )
        if count > 0
    )


def crowding_counts(
    *,
    entries: Sequence[CandidateEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
) -> tuple[int, ...]:
    """Count near neighbors for each entry inside the bank.

    Parameters
    ----------
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries whose crowding counts are measured.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute pairwise distances.
    distance_cutoff : float
        Distance threshold below which two entries are considered neighbors.

    Returns
    -------
    tuple[int, ...]
        Near-neighbor count for each entry.

    Raises
    ------
    ValueError
        Raised when ``distance_cutoff`` is negative.
    """
    if distance_cutoff < 0.0:
        msg = "distance_cutoff must be non-negative"
        raise ValueError(msg)

    counts = [0] * len(entries)
    for left_index, left_entry in enumerate(entries[:-1]):
        for right_index, right_entry in enumerate(entries[left_index + 1 :], start=left_index + 1):
            distance = require_valid_distance(
                diversity_metric.distance(left_entry.candidate, right_entry.candidate),
            )
            if distance < distance_cutoff:
                counts[left_index] += 1
                counts[right_index] += 1

    return tuple(counts)


def crowding_aware_scores(
    *,
    base_scores: Sequence[float],
    entries: Sequence[CandidateEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
    penalty_ratio: float,
    niche_quality_policy: CSANicheQualityPolicy,
) -> tuple[float, ...]:
    """Bias removal scores toward crowded entries with comparable quality.

    Parameters
    ----------
    base_scores : Sequence[float]
        Base removal scores before crowding penalties are applied.
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries associated with ``base_scores``.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute crowding.
    distance_cutoff : float
        Distance threshold below which two entries are considered neighbors.
    penalty_ratio : float
        Relative strength of the crowding penalty.
    niche_quality_policy : CSANicheQualityPolicy
        Additional niche-quality policy used to bias removal scores.

    Returns
    -------
    tuple[float, ...]
        Adjusted removal scores.

    Raises
    ------
    ValueError
        Raised when the penalty ratio is negative or the score/entry lengths do
        not match.
    """
    if penalty_ratio < 0.0:
        msg = "penalty_ratio must be non-negative"
        raise ValueError(msg)

    if len(base_scores) != len(entries):
        msg = "base_scores and entries must have the same length"
        raise ValueError(msg)

    if len(entries) == 0:
        return tuple(base_scores)

    counts = crowding_counts(
        entries=entries,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
    )
    maximum_count = max(counts, default=0)
    if maximum_count == 0:
        return tuple(base_scores)

    score_span = max(base_scores) - min(base_scores)
    score_scale = max(score_span, 1.0)
    crowding_penalties = tuple(
        penalty_ratio * score_scale * (count / maximum_count)
        for count in counts
    )
    quality_penalties = niche_quality_penalties(
        base_scores=base_scores,
        entries=entries,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
        counts=counts,
        score_scale=score_scale,
        policy=niche_quality_policy,
    )
    return tuple(
        score + crowding_penalty + niche_penalty
        for score, crowding_penalty, niche_penalty in zip(
            base_scores,
            crowding_penalties,
            quality_penalties,
            strict=True,
        )
    )


def niche_quality_penalties(
    *,
    base_scores: Sequence[float],
    entries: Sequence[CandidateEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
    counts: Sequence[int],
    score_scale: float,
    policy: CSANicheQualityPolicy,
) -> tuple[float, ...]:
    """Compute additional removal penalties from local niche quality.

    Parameters
    ----------
    base_scores : Sequence[float]
        Base removal scores before niche-quality adjustment.
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries associated with ``base_scores``.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute local niches.
    distance_cutoff : float
        Distance threshold below which two entries are considered neighbors.
    counts : Sequence[int]
        Precomputed crowding counts for each entry.
    score_scale : float
        Score scale used to normalize penalties.
    policy : CSANicheQualityPolicy
        Niche-quality policy controlling the penalty mode.

    Returns
    -------
    tuple[float, ...]
        Additional niche-quality penalties for each entry.

    Raises
    ------
    ValueError
        Raised when the policy mode is unsupported.
    """
    if policy.mode == "disabled" or policy.ratio == 0.0 or len(entries) == 0:
        return (0.0,) * len(entries)

    if policy.mode == "mean":
        niche_scores = mean_niche_scores(
            base_scores=base_scores,
            entries=entries,
            diversity_metric=diversity_metric,
            distance_cutoff=distance_cutoff,
        )
        niche_score_span = max(niche_scores) - min(niche_scores)
        if niche_score_span == 0.0:
            return (0.0,) * len(entries)

        maximum_count = max(counts, default=0)
        if maximum_count == 0:
            return (0.0,) * len(entries)

        minimum_niche_score = min(niche_scores)
        return tuple(
            policy.ratio
            * score_scale
            * (count / maximum_count)
            * ((niche_score - minimum_niche_score) / niche_score_span)
            for count, niche_score in zip(counts, niche_scores, strict=True)
        )

    if policy.mode == "best_mean":
        niche_scores = best_mean_niche_scores(
            base_scores=base_scores,
            entries=entries,
            diversity_metric=diversity_metric,
            distance_cutoff=distance_cutoff,
        )
        niche_score_span = max(niche_scores) - min(niche_scores)
        if niche_score_span == 0.0:
            return (0.0,) * len(entries)

        maximum_count = max(counts, default=0)
        if maximum_count == 0:
            return (0.0,) * len(entries)

        minimum_niche_score = min(niche_scores)
        return tuple(
            policy.ratio
            * score_scale
            * (count / maximum_count)
            * ((niche_score - minimum_niche_score) / niche_score_span)
            for count, niche_score in zip(counts, niche_scores, strict=True)
        )

    msg = f"unsupported niche-quality mode: {policy.mode}"
    raise ValueError(msg)


def mean_niche_scores(
    *,
    base_scores: Sequence[float],
    entries: Sequence[CandidateEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
) -> tuple[float, ...]:
    """Compute the mean score inside each entry's cutoff-neighborhood.

    Parameters
    ----------
    base_scores : Sequence[float]
        Base scores associated with the entries.
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries whose neighborhoods are scored.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute local neighborhoods.
    distance_cutoff : float
        Distance threshold below which two entries are considered neighbors.

    Returns
    -------
    tuple[float, ...]
        Mean neighborhood score for each entry.

    Raises
    ------
    ValueError
        Raised when ``base_scores`` and ``entries`` have different lengths.
    """
    if len(base_scores) != len(entries):
        msg = "base_scores and entries must have the same length"
        raise ValueError(msg)

    sums = list(base_scores)
    counts = [1] * len(entries)
    for left_index, left_entry in enumerate(entries[:-1]):
        for right_index, right_entry in enumerate(
            entries[left_index + 1 :],
            start=left_index + 1,
        ):
            distance = require_valid_distance(
                diversity_metric.distance(left_entry.candidate, right_entry.candidate),
            )
            if distance < distance_cutoff:
                sums[left_index] += base_scores[right_index]
                sums[right_index] += base_scores[left_index]
                counts[left_index] += 1
                counts[right_index] += 1

    return tuple(
        total / count
        for total, count in zip(sums, counts, strict=True)
    )


def best_mean_niche_scores(
    *,
    base_scores: Sequence[float],
    entries: Sequence[CandidateEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
) -> tuple[float, ...]:
    """Compute a mixed niche score combining local best and local mean.

    Parameters
    ----------
    base_scores : Sequence[float]
        Base scores associated with the entries.
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries whose neighborhoods are scored.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute local neighborhoods.
    distance_cutoff : float
        Distance threshold below which two entries are considered neighbors.

    Returns
    -------
    tuple[float, ...]
        Mixed niche score for each entry.
    """
    mean_scores = mean_niche_scores(
        base_scores=base_scores,
        entries=entries,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
    )
    best_scores = best_niche_scores(
        base_scores=base_scores,
        entries=entries,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
    )
    return tuple(
        0.5 * (mean_score + best_score)
        for mean_score, best_score in zip(mean_scores, best_scores, strict=True)
    )


def best_niche_scores(
    *,
    base_scores: Sequence[float],
    entries: Sequence[CandidateEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
) -> tuple[float, ...]:
    """Compute the best score inside each entry's cutoff-neighborhood.

    Parameters
    ----------
    base_scores : Sequence[float]
        Base scores associated with the entries.
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries whose neighborhoods are scored.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute local neighborhoods.
    distance_cutoff : float
        Distance threshold below which two entries are considered neighbors.

    Returns
    -------
    tuple[float, ...]
        Best neighborhood score for each entry.

    Raises
    ------
    ValueError
        Raised when ``base_scores`` and ``entries`` have different lengths.
    """
    if len(base_scores) != len(entries):
        msg = "base_scores and entries must have the same length"
        raise ValueError(msg)

    best_scores = list(base_scores)
    for left_index, left_entry in enumerate(entries[:-1]):
        for right_index, right_entry in enumerate(
            entries[left_index + 1 :],
            start=left_index + 1,
        ):
            distance = require_valid_distance(
                diversity_metric.distance(left_entry.candidate, right_entry.candidate),
            )
            if distance < distance_cutoff:
                left_score = base_scores[left_index]
                right_score = base_scores[right_index]
                if right_score < best_scores[left_index]:
                    best_scores[left_index] = right_score
                if left_score < best_scores[right_index]:
                    best_scores[right_index] = left_score

    return tuple(best_scores)
