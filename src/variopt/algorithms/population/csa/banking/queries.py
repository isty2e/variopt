"""Private bank query primitives shared by CSA bank update components."""

from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from math import isfinite
from typing import ClassVar, Generic, Protocol, TypeVar

from .....distance import require_valid_distance
from .....diversity import DiversityMetric
from .....diversity.space_metric import (
    CompiledStructuredDistanceView,
    structured_distance_between_validated_candidates,
    supports_candidate_typed_structured_distance,
)
from .....typevars import CandidateT
from .update.policy import CSANicheQualityPolicy

EntryCandidateT_co = TypeVar("EntryCandidateT_co", covariant=True)


class CandidateEntry(Protocol[EntryCandidateT_co]):
    """Minimal entry view needed by bank update queries.

    Notes
    -----
    Bank-query helpers depend only on candidate/value access, so this protocol
    keeps them reusable across concrete entry types.
    """

    @property
    def candidate(self) -> EntryCandidateT_co:
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


def infer_score_gap(
    entries: Sequence[CandidateEntry[CandidateT]],
) -> float | None:
    """Infer the finite min-max objective-value gap across bank entries.

    Parameters
    ----------
    entries : Sequence[CandidateEntry[CandidateT]]
        Entries whose objective-value spread is summarized.

    Returns
    -------
    float | None
        Finite difference between maximum and minimum objective value, or
        ``None`` when no entries are available or the difference overflows.
    """
    if not entries:
        return None

    values = tuple(entry.value for entry in entries)
    score_gap = max(values) - min(values)
    if not isfinite(score_gap):
        return None
    return score_gap


class BankDistanceWorkspace(Generic[CandidateT]):
    """Operation-local pairwise distance workspace for one bank snapshot.

    Parameters
    ----------
    entries : Sequence[CandidateEntry[CandidateT]]
        Bank entries whose pairwise distances may be requested.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute and validate distances.
    compiled_distance_view : CompiledStructuredDistanceView[CandidateT] | None, default=None
        Previously prepared view aligned to ``entries``. This is used only when
        rebasing the operation-local workspace; ordinary construction compiles
        an eligible exact structured metric automatically.

    Notes
    -----
    This workspace is intentionally mutable and request-local. It must not be
    stored in checkpoint state or persistent optimizer state.
    """

    entries: tuple[CandidateEntry[CandidateT], ...]
    diversity_metric: DiversityMetric[CandidateT]
    distances: dict[tuple[int, int], float]
    compiled_distance_view: CompiledStructuredDistanceView[CandidateT] | None

    __slots__: ClassVar[tuple[str, ...]] = (
        "compiled_distance_view",
        "distances",
        "diversity_metric",
        "entries",
    )

    def __init__(
        self,
        *,
        entries: Sequence[CandidateEntry[CandidateT]],
        diversity_metric: DiversityMetric[CandidateT],
        compiled_distance_view: CompiledStructuredDistanceView[CandidateT]
        | None = None,
    ) -> None:
        self.entries = tuple(entries)
        self.diversity_metric = diversity_metric
        self.distances = {}
        candidates = tuple(entry.candidate for entry in self.entries)
        if compiled_distance_view is not None:
            if (
                not supports_candidate_typed_structured_distance(diversity_metric)
                or not diversity_metric._owns_distance_view(compiled_distance_view)
                or not compiled_distance_view.is_aligned_with(candidates)
            ):
                msg = "compiled distance view must align with the metric and entries"
                raise ValueError(msg)
            self.compiled_distance_view = compiled_distance_view
        elif supports_candidate_typed_structured_distance(diversity_metric):
            self.compiled_distance_view = diversity_metric._compile_distance_view(
                candidates
            )
        else:
            self.compiled_distance_view = None

    def rebase(
        self,
        *,
        entries: Sequence[CandidateEntry[CandidateT]],
        invalidated_indices: AbstractSet[int],
    ) -> "BankDistanceWorkspace[CandidateT]":
        """Return a workspace aligned to a related bank snapshot.

        Parameters
        ----------
        entries : Sequence[CandidateEntry[CandidateT]]
            New bank entries for the rebased workspace.
        invalidated_indices : collections.abc.Set[int]
            Indices whose candidate changed or was appended in ``entries``.

        Returns
        -------
        BankDistanceWorkspace[CandidateT]
            Workspace aligned to ``entries`` with reusable pair distances
            retained.
        """
        if entries is self.entries and not invalidated_indices:
            return self

        compiled_distance_view = self.compiled_distance_view
        if compiled_distance_view is not None:
            compiled_distance_view = compiled_distance_view.rebase(
                candidates=tuple(entry.candidate for entry in entries),
                invalidated_indices=invalidated_indices,
            )

        workspace = type(self)(
            entries=entries,
            diversity_metric=self.diversity_metric,
            compiled_distance_view=compiled_distance_view,
        )
        if not self.distances or len(entries) < len(self.entries):
            return workspace

        common_entry_count = min(len(self.entries), len(entries))
        invalidated_index_set = frozenset(
            index for index in invalidated_indices if index >= 0
        )
        reusable_indices = frozenset(
            index
            for index in range(common_entry_count)
            if index not in invalidated_index_set
            and entries[index].candidate is self.entries[index].candidate
        )
        workspace.distances.update(
            (key, distance)
            for key, distance in self.distances.items()
            if key[1] < common_entry_count
            and key[0] in reusable_indices
            and key[1] in reusable_indices
        )
        return workspace

    def distance(self, left_index: int, right_index: int) -> float:
        """Return one validated pairwise distance, computing it at most once.

        Parameters
        ----------
        left_index : int
            Index of the left bank entry.
        right_index : int
            Index of the right bank entry.

        Returns
        -------
        float
            Validated pairwise distance between the two entries.

        Raises
        ------
        IndexError
            If either index lies outside the workspace entry snapshot.
        """
        if (
            left_index < 0
            or left_index >= len(self.entries)
            or right_index < 0
            or right_index >= len(self.entries)
        ):
            msg = "distance indices must reference workspace entries"
            raise IndexError(msg)
        if left_index == right_index:
            return 0.0

        key = (
            (left_index, right_index)
            if left_index < right_index
            else (right_index, left_index)
        )
        distance = self.distances.get(key)
        if distance is not None:
            return distance

        compiled_distance_view = self.compiled_distance_view
        if compiled_distance_view is None:
            left_entry = self.entries[key[0]]
            right_entry = self.entries[key[1]]
            distance = require_valid_distance(
                validated_candidate_distance(
                    self.diversity_metric,
                    left_entry.candidate,
                    right_entry.candidate,
                ),
            )
        else:
            distance = require_valid_distance(
                compiled_distance_view.distance(key[0], key[1])
            )
        self.distances[key] = distance
        return distance

    def distances_to_candidate(self, candidate: CandidateT) -> tuple[float, ...]:
        """Return validated distances from one candidate to every bank entry.

        Parameters
        ----------
        candidate : CandidateT
            Canonical candidate already admitted through the CSA proposal
            validation boundary.

        Returns
        -------
        tuple[float, ...]
            Distances aligned one-to-one with :attr:`entries`.
        """
        compiled_distance_view = self.compiled_distance_view
        if compiled_distance_view is not None:
            return tuple(
                require_valid_distance(distance)
                for distance in compiled_distance_view.distances_to(candidate)
            )
        return tuple(
            require_valid_distance(
                validated_candidate_distance(
                    self.diversity_metric,
                    candidate,
                    entry.candidate,
                )
            )
            for entry in self.entries
        )

    def is_aligned_with_entries(
        self,
        entries: Sequence[CandidateEntry[CandidateT]],
    ) -> bool:
        """Return whether entries carry the represented candidates in order.

        Parameters
        ----------
        entries : Sequence[CandidateEntry[CandidateT]]
            Entry snapshot whose candidate alignment is checked by identity.

        Returns
        -------
        bool
            Whether every entry carries the candidate represented at the same
            workspace index.
        """
        return len(entries) == len(self.entries) and all(
            entry.candidate is self.entries[index].candidate
            for index, entry in enumerate(entries)
        )

    def average_pairwise_distance(self) -> float:
        """Return the mean distance across distinct bank-entry pairs.

        Returns
        -------
        float
            Mean pairwise distance, or ``0.0`` when fewer than two entries are
            represented.
        """
        entry_count = len(self.entries)
        if entry_count < 2:
            return 0.0

        distance_sum = 0.0
        pair_count = 0
        for left_index in range(entry_count - 1):
            for right_index in range(left_index + 1, entry_count):
                distance_sum += self.distance(left_index, right_index)
                pair_count += 1
        return distance_sum / float(pair_count)

    def seed_entry_distances(
        self,
        *,
        entry_index: int,
        distances: Sequence[float],
    ) -> None:
        """Cache known distances between one entry and every aligned entry.

        Parameters
        ----------
        entry_index : int
            Entry index whose distances are being cached.
        distances : collections.abc.Sequence[float]
            Distance vector aligned to :attr:`entries`. The value at
            ``entry_index`` is ignored because self-distance is always
            represented by :meth:`distance` as ``0.0``.

        Raises
        ------
        IndexError
            Raised when ``entry_index`` is outside the workspace entry range.
        ValueError
            Raised when ``distances`` is not aligned to :attr:`entries`, or when
            a supplied non-self distance is invalid.
        """
        if entry_index < 0 or entry_index >= len(self.entries):
            msg = "entry_index must be a valid entry index"
            raise IndexError(msg)

        if len(distances) != len(self.entries):
            msg = "distances must align one-to-one with entries"
            raise ValueError(msg)

        for other_index, distance in enumerate(distances):
            if other_index == entry_index:
                continue

            key = (
                (entry_index, other_index)
                if entry_index < other_index
                else (other_index, entry_index)
            )
            self.distances[key] = require_valid_distance(distance)

    def crowding_counts(self, *, distance_cutoff: float) -> tuple[int, ...]:
        """Count near neighbors for each entry using cached pair distances.

        Parameters
        ----------
        distance_cutoff : float
            Distance threshold below which two entries are considered
            neighbors.

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

        counts = [0] * len(self.entries)
        for left_index in range(len(self.entries) - 1):
            for right_index in range(left_index + 1, len(self.entries)):
                if self.distance(left_index, right_index) < distance_cutoff:
                    counts[left_index] += 1
                    counts[right_index] += 1

        return tuple(counts)


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
            validated_candidate_distance(
                diversity_metric,
                candidate,
                entry.candidate,
            )
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
    distance_workspace: BankDistanceWorkspace[CandidateT] | None = None,
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
    distance_workspace : BankDistanceWorkspace[CandidateT] | None, default=None
        Optional operation-local pairwise distance workspace aligned to
        ``entries``.

    Returns
    -------
    frozenset[int]
        Entry indices that have at least one near neighbor.
    """
    counts = (
        crowding_counts(
            entries=entries,
            diversity_metric=diversity_metric,
            distance_cutoff=distance_cutoff,
        )
        if distance_workspace is None
        else distance_workspace.crowding_counts(distance_cutoff=distance_cutoff)
    )
    return frozenset(index for index, count in enumerate(counts) if count > 0)


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
        for right_index, right_entry in enumerate(
            entries[left_index + 1 :], start=left_index + 1
        ):
            distance = require_valid_distance(
                validated_candidate_distance(
                    diversity_metric,
                    left_entry.candidate,
                    right_entry.candidate,
                ),
            )
            if distance < distance_cutoff:
                counts[left_index] += 1
                counts[right_index] += 1

    return tuple(counts)


def validated_candidate_distance(
    diversity_metric: DiversityMetric[CandidateT],
    left: CandidateT,
    right: CandidateT,
) -> float:
    """Return distance for candidates already admitted through CSA validation.

    CSA bank candidates originate from pending proposals that were validated
    when emitted or restored. Structured metrics can therefore use their
    validated-candidate geometry path here; non-structured metrics keep the
    public ``DiversityMetric`` contract. This function intentionally does not
    revalidate structured candidate shape; callers own the bank-admission
    validation boundary.
    """
    if supports_candidate_typed_structured_distance(diversity_metric):
        return structured_distance_between_validated_candidates(
            diversity_metric,
            left,
            right,
        )
    return diversity_metric.distance(left, right)


def crowding_aware_scores(
    *,
    base_scores: Sequence[float],
    entries: Sequence[CandidateEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
    penalty_ratio: float,
    niche_quality_policy: CSANicheQualityPolicy,
    distance_workspace: BankDistanceWorkspace[CandidateT] | None = None,
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
    distance_workspace : BankDistanceWorkspace[CandidateT] | None, default=None
        Optional operation-local pairwise distance workspace aligned to
        ``entries``.

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

    niche_quality_enabled = (
        niche_quality_policy.mode != "disabled" and niche_quality_policy.ratio != 0.0
    )
    if penalty_ratio == 0.0 and not niche_quality_enabled:
        return tuple(base_scores)

    if distance_workspace is None and niche_quality_enabled:
        distance_workspace = BankDistanceWorkspace(
            entries=entries,
            diversity_metric=diversity_metric,
        )
    if distance_workspace is None:
        counts = crowding_counts(
            entries=entries,
            diversity_metric=diversity_metric,
            distance_cutoff=distance_cutoff,
        )
    else:
        counts = distance_workspace.crowding_counts(distance_cutoff=distance_cutoff)
    maximum_count = max(counts, default=0)
    if maximum_count == 0:
        return tuple(base_scores)

    score_span = max(base_scores) - min(base_scores)
    score_scale = max(score_span, 1.0)
    crowding_penalties = tuple(
        penalty_ratio * score_scale * (count / maximum_count) for count in counts
    )
    quality_penalties = niche_quality_penalties(
        base_scores=base_scores,
        entries=entries,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
        counts=counts,
        score_scale=score_scale,
        policy=niche_quality_policy,
        distance_workspace=distance_workspace,
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
    distance_workspace: BankDistanceWorkspace[CandidateT] | None = None,
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
    distance_workspace : BankDistanceWorkspace[CandidateT] | None, default=None
        Optional operation-local pairwise distance workspace shared with
        crowding counts.

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

    if distance_workspace is None:
        distance_workspace = BankDistanceWorkspace(
            entries=entries,
            diversity_metric=diversity_metric,
        )

    if policy.mode == "mean":
        niche_scores = mean_niche_scores(
            base_scores=base_scores,
            entries=entries,
            diversity_metric=diversity_metric,
            distance_cutoff=distance_cutoff,
            distance_workspace=distance_workspace,
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
            distance_workspace=distance_workspace,
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
    distance_workspace: BankDistanceWorkspace[CandidateT] | None = None,
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
    distance_workspace : BankDistanceWorkspace[CandidateT] | None, default=None
        Optional operation-local pairwise distance workspace.

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

    if distance_workspace is None:
        distance_workspace = BankDistanceWorkspace(
            entries=entries,
            diversity_metric=diversity_metric,
        )

    sums = list(base_scores)
    counts = [1] * len(entries)
    for left_index in range(len(entries) - 1):
        for right_index in range(left_index + 1, len(entries)):
            if distance_workspace.distance(left_index, right_index) < distance_cutoff:
                sums[left_index] += base_scores[right_index]
                sums[right_index] += base_scores[left_index]
                counts[left_index] += 1
                counts[right_index] += 1

    return tuple(total / count for total, count in zip(sums, counts, strict=True))


def best_mean_niche_scores(
    *,
    base_scores: Sequence[float],
    entries: Sequence[CandidateEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
    distance_workspace: BankDistanceWorkspace[CandidateT] | None = None,
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
    distance_workspace : BankDistanceWorkspace[CandidateT] | None, default=None
        Optional operation-local pairwise distance workspace.

    Returns
    -------
    tuple[float, ...]
        Mixed niche score for each entry.
    """
    if distance_workspace is None:
        distance_workspace = BankDistanceWorkspace(
            entries=entries,
            diversity_metric=diversity_metric,
        )

    mean_scores = mean_niche_scores(
        base_scores=base_scores,
        entries=entries,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
        distance_workspace=distance_workspace,
    )
    best_scores = best_niche_scores(
        base_scores=base_scores,
        entries=entries,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
        distance_workspace=distance_workspace,
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
    distance_workspace: BankDistanceWorkspace[CandidateT] | None = None,
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
    distance_workspace : BankDistanceWorkspace[CandidateT] | None, default=None
        Optional operation-local pairwise distance workspace.

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

    if distance_workspace is None:
        distance_workspace = BankDistanceWorkspace(
            entries=entries,
            diversity_metric=diversity_metric,
        )

    best_scores = list(base_scores)
    for left_index in range(len(entries) - 1):
        for right_index in range(left_index + 1, len(entries)):
            if distance_workspace.distance(left_index, right_index) < distance_cutoff:
                left_score = base_scores[left_index]
                right_score = base_scores[right_index]
                best_scores[left_index] = min(best_scores[left_index], right_score)
                best_scores[right_index] = min(best_scores[right_index], left_score)

    return tuple(best_scores)
