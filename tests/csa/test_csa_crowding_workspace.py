"""Incremental neighborhood counts over immutable CSA bank snapshots."""

import gc
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from weakref import ref

import numpy as np
import pytest
from typing_extensions import override

from variopt.algorithms.population.csa.banking.bank import BankEntry
from variopt.algorithms.population.csa.banking.queries import (
    BankDistanceWorkspace,
    CandidateEntry,
    crowding_aware_scores,
)
from variopt.algorithms.population.csa.banking.update.policy import (
    CSANicheQualityPolicy,
)
from variopt.diversity import DiversityMetric
from variopt.diversity.space_metric import CompiledStructuredDistanceView


@dataclass
class RecordingDistance(DiversityMetric[int]):
    calls: list[tuple[int, int]] = field(default_factory=list)
    fail_at: int | None = None

    @override
    def distance(self, left: int, right: int) -> float:
        self.calls.append((left, right))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("metric failure")
        return float(abs(left - right))


class RecordingWorkspace(BankDistanceWorkspace[int]):
    def __init__(
        self,
        *,
        entries: Sequence[CandidateEntry[int]],
        diversity_metric: DiversityMetric[int],
        compiled_distance_view: CompiledStructuredDistanceView[int] | None = None,
    ) -> None:
        super().__init__(
            entries=entries,
            diversity_metric=diversity_metric,
            compiled_distance_view=compiled_distance_view,
        )
        self.pair_queries: list[tuple[int, int]] = []

    @override
    def distance(self, left_index: int, right_index: int) -> float:
        self.pair_queries.append((left_index, right_index))
        return super().distance(left_index, right_index)


def _workspace(*values: int) -> RecordingWorkspace:
    return RecordingWorkspace(
        entries=tuple(BankEntry(candidate=value, value=0.0) for value in values),
        diversity_metric=RecordingDistance(),
    )


def _full_counts(
    workspace: BankDistanceWorkspace[int], cutoff: float
) -> tuple[int, ...]:
    values = tuple(entry.candidate for entry in workspace.entries)
    return tuple(
        sum(
            index != other and abs(value - neighbor) < cutoff
            for other, neighbor in enumerate(values)
        )
        for index, value in enumerate(values)
    )


def test_unchanged_queries_reuse_counts_without_visiting_pairs() -> None:
    workspace = _workspace(0, 1, 3, 6, 10)
    expected = _full_counts(workspace, 4.0)
    assert workspace.crowding_counts(distance_cutoff=4.0) == expected
    assert workspace.pair_queries == list(combinations(range(5), 2))
    workspace.pair_queries.clear()

    assert workspace.crowding_counts(distance_cutoff=4.0) == expected
    assert workspace.pair_queries == []


@pytest.mark.parametrize("slot", [0, 2, 4])
def test_seeded_replacement_visits_only_changed_pairs(slot: int) -> None:
    original = _workspace(0, 1, 3, 6, 10)
    expected_original = original.crowding_counts(distance_cutoff=4.0)
    entries = list(original.entries)
    entries[slot] = BankEntry(candidate=5, value=-1.0)
    changed = original.rebase(entries=entries, invalidated_indices=frozenset({slot}))
    changed.seed_entry_distances(
        entry_index=slot,
        distances=tuple(float(abs(5 - entry.candidate)) for entry in entries),
    )

    assert changed.crowding_counts(distance_cutoff=4.0) == _full_counts(changed, 4.0)
    assert isinstance(changed, RecordingWorkspace)
    assert changed.pair_queries == [
        pair for pair in combinations(range(5), 2) if slot in pair
    ]
    assert original.crowding_counts(distance_cutoff=4.0) == expected_original


def test_deferred_rebases_visit_each_changed_pair_once_in_canonical_order() -> None:
    metric = RecordingDistance()
    original = RecordingWorkspace(
        entries=tuple(BankEntry(candidate=value, value=0.0) for value in (9, 1, 7, 3)),
        diversity_metric=metric,
    )
    original.crowding_counts(distance_cutoff=4.0)
    metric.calls.clear()
    current = original
    for slot, value in ((2, 2), (0, 5), (2, 8)):
        entries = list(current.entries)
        entries[slot] = BankEntry(candidate=value, value=0.0)
        current = current.rebase(entries=entries, invalidated_indices=frozenset())

    assert metric.calls == []
    assert current.crowding_counts(distance_cutoff=4.0) == _full_counts(current, 4.0)
    assert isinstance(current, RecordingWorkspace)
    pairs = [pair for pair in combinations(range(4), 2) if 0 in pair or 2 in pair]
    assert current.pair_queries == pairs
    assert metric.calls == [
        (current.entries[left].candidate, current.entries[right].candidate)
        for left, right in pairs
    ]


def test_score_only_rebase_keeps_counts_but_not_niche_score_summaries() -> None:
    original = _workspace(0, 1, 3, 6)
    expected = original.crowding_counts(distance_cutoff=4.0)
    changed = original.rebase(
        entries=tuple(
            BankEntry(candidate=entry.candidate, value=99.0)
            for entry in original.entries
        ),
        invalidated_indices=frozenset(),
    )
    assert changed.crowding_counts(distance_cutoff=4.0) == expected
    assert isinstance(changed, RecordingWorkspace)
    assert changed.pair_queries == []

    for policy in (
        CSANicheQualityPolicy(mode="mean", ratio=0.5),
        CSANicheQualityPolicy(mode="best_mean", ratio=0.5),
    ):
        for scores in ((0.0, 1.0, 2.0, 3.0), (10.0, -20.0, 8.0, 1.0)):
            cached_scores = crowding_aware_scores(
                base_scores=scores,
                entries=changed.entries,
                diversity_metric=changed.diversity_metric,
                distance_cutoff=4.0,
                penalty_ratio=0.2,
                niche_quality_policy=policy,
                distance_workspace=changed,
            )
            assert cached_scores == crowding_aware_scores(
                base_scores=scores,
                entries=changed.entries,
                diversity_metric=changed.diversity_metric,
                distance_cutoff=4.0,
                penalty_ratio=0.2,
                niche_quality_policy=policy,
            )


@pytest.mark.parametrize("cutoff", [0.0, 1.0, 4.0, float("inf"), float("nan")])
def test_cutoff_changes_and_recovery_rebuild_counts(cutoff: float) -> None:
    workspace = _workspace(0, 0, 1, 4)
    workspace.crowding_counts(distance_cutoff=2.0)
    assert workspace.crowding_counts(distance_cutoff=cutoff) == _full_counts(
        workspace, cutoff
    )
    assert workspace.crowding_counts(distance_cutoff=2.0) == _full_counts(
        workspace, 2.0
    )
    with pytest.raises(ValueError, match="non-negative"):
        workspace.crowding_counts(distance_cutoff=-1.0)
    assert workspace.crowding_counts(distance_cutoff=2.0) == _full_counts(
        workspace, 2.0
    )


def test_reseeding_and_equal_revision_branches_do_not_mutate_old_counts() -> None:
    original = _workspace(0, 10, 30)
    assert original.crowding_counts(distance_cutoff=5.0) == (0, 0, 0)
    left = original.rebase(
        entries=tuple(original.entries), invalidated_indices=frozenset({1})
    )
    right = original.rebase(
        entries=tuple(original.entries), invalidated_indices=frozenset({2})
    )
    left.seed_entry_distances(entry_index=1, distances=(1.0, 0.0, 2.0))
    right.seed_entry_distances(entry_index=2, distances=(3.0, 10.0, 0.0))
    assert left.crowding_counts(distance_cutoff=5.0) == (1, 2, 1)
    assert right.crowding_counts(distance_cutoff=5.0) == (1, 0, 1)
    assert original.crowding_counts(distance_cutoff=5.0) == (0, 0, 0)

    original.seed_entry_distances(entry_index=0, distances=(0.0, 4.0, 1.0))
    assert original.crowding_counts(distance_cutoff=5.0) == (2, 1, 1)
    assert left.crowding_counts(distance_cutoff=5.0) == (1, 2, 1)
    assert right.crowding_counts(distance_cutoff=5.0) == (1, 0, 1)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_failed_seed_does_not_invalidate_or_change_complete_counts(
    invalid: float,
) -> None:
    workspace = _workspace(0, 1, 10)
    expected = workspace.crowding_counts(distance_cutoff=2.0)
    workspace.pair_queries.clear()
    with pytest.raises(ValueError):
        workspace.seed_entry_distances(entry_index=0, distances=(0.0, 999.0, invalid))
    assert workspace.crowding_counts(distance_cutoff=2.0) == expected
    assert workspace.pair_queries == []


def test_failed_incremental_query_is_retryable_without_publishing_partial_counts() -> (
    None
):
    metric = RecordingDistance()
    original = RecordingWorkspace(
        entries=_workspace(0, 1, 3, 6).entries, diversity_metric=metric
    )
    expected_original = original.crowding_counts(distance_cutoff=4.0)
    changed = original.rebase(
        entries=(
            *original.entries[:2],
            BankEntry(candidate=2, value=0.0),
            original.entries[3],
        ),
        invalidated_indices=frozenset({2}),
    )
    metric.fail_at = len(metric.calls) + 2
    with pytest.raises(RuntimeError, match="metric failure"):
        changed.crowding_counts(distance_cutoff=4.0)
    assert original.crowding_counts(distance_cutoff=4.0) == expected_original
    assert changed.crowding_counts(distance_cutoff=4.0) == _full_counts(changed, 4.0)
    assert changed.crowding_counts(distance_cutoff=4.0) == _full_counts(changed, 4.0)


def test_append_then_remove_remaps_without_reusing_shifted_counts() -> None:
    original = _workspace(0, 1, 10)
    original.crowding_counts(distance_cutoff=3.0)
    grown = original.rebase(
        entries=(*original.entries, BankEntry(candidate=2, value=0.0)),
        invalidated_indices=frozenset({3}),
    )
    assert grown.crowding_counts(distance_cutoff=3.0) == (2, 2, 0, 2)
    assert isinstance(grown, RecordingWorkspace)
    assert grown.pair_queries == [(0, 3), (1, 3), (2, 3)]
    removed = grown.rebase(
        entries=(grown.entries[3], grown.entries[2]), invalidated_indices=frozenset()
    )
    assert removed.crowding_counts(distance_cutoff=3.0) == (0, 0)
    assert original.crowding_counts(distance_cutoff=3.0) == (1, 1, 0)


def test_empty_singleton_and_complete_reorder_match_fresh_counts() -> None:
    current = _workspace()
    for values in ((), (3,), (3, 4), (8, 3, 4), (4, 8, 3), ()):
        current = current.rebase(
            entries=tuple(BankEntry(candidate=value, value=0.0) for value in values),
            invalidated_indices=frozenset(),
        )
        assert current.crowding_counts(distance_cutoff=2.0) == _full_counts(
            current, 2.0
        )


def test_explicit_invalidation_discards_seeded_neighborhood_contributions() -> None:
    original = _workspace(0, 10, 30)
    original.seed_entry_distances(entry_index=1, distances=(1.0, 0.0, 2.0))
    assert original.crowding_counts(distance_cutoff=5.0) == (1, 2, 1)
    changed = original.rebase(
        entries=original.entries, invalidated_indices=frozenset({1})
    )
    assert changed.crowding_counts(distance_cutoff=5.0) == (0, 0, 0)
    assert original.crowding_counts(distance_cutoff=5.0) == (1, 2, 1)


def test_long_unqueried_replacement_chain_retains_only_one_count_row_alignment() -> (
    None
):
    current = _workspace(*range(8))
    current.crowding_counts(distance_cutoff=3.0)
    for step in range(256):
        entries = list(current.entries)
        slot = step % len(entries)
        entries[slot] = BankEntry(candidate=step, value=0.0)
        current = current.rebase(entries=entries, invalidated_indices=frozenset({slot}))
        current.seed_entry_distances(
            entry_index=slot,
            distances=tuple(float(abs(step - entry.candidate)) for entry in entries),
        )
        snapshot = current._crowding_snapshot
        assert snapshot is not None
        retained_rows = {id(row): row for row in (*current._rows, *snapshot.rows)}
        assert len(retained_rows) <= 2 * len(entries)
        assert sum(len(row.distances) for row in retained_rows.values()) <= (
            2 * len(entries) * (len(entries) - 1)
        )
    assert current.crowding_counts(distance_cutoff=3.0) == _full_counts(current, 3.0)


@pytest.mark.parametrize("seed", [0, 11, 47])
def test_long_mutation_walk_preserves_counts_across_retained_branches(
    seed: int,
) -> None:
    rng = np.random.RandomState(seed)
    current = _workspace(0, 0, 3, 5, 10)
    retained: list[BankDistanceWorkspace[int]] = []
    for step in range(180):
        cutoff = 4.0 if step % 7 else float(step % 6)
        assert current.crowding_counts(distance_cutoff=cutoff) == _full_counts(
            current, cutoff
        )
        if step % 11 == 0:
            retained.append(current)
        if step % 13 == 0 and retained:
            current = retained[int(rng.randint(len(retained)))]
        entries = list(current.entries)
        if step % 5 == 0 and len(entries) > 1:
            del entries[int(rng.randint(len(entries)))]
        elif step % 3 == 0:
            entries.append(BankEntry(candidate=int(rng.randint(12)), value=0.0))
        else:
            slot = int(rng.randint(len(entries)))
            entries[slot] = BankEntry(candidate=int(rng.randint(12)), value=0.0)
        current = current.rebase(entries=entries, invalidated_indices=frozenset())
        if step % 2 == 0:
            slot = int(rng.randint(len(entries)))
            current.seed_entry_distances(
                entry_index=slot,
                distances=tuple(
                    float(abs(entries[slot].candidate - entry.candidate))
                    for entry in entries
                ),
            )
        for snapshot in retained:
            assert snapshot.crowding_counts(distance_cutoff=4.0) == _full_counts(
                snapshot, 4.0
            )


@dataclass(eq=False)
class EqualityHostileCandidate:
    value: int

    def __eq__(self, other: object) -> bool:
        raise AssertionError("candidate equality must not be used")


class HostileDistance(DiversityMetric[EqualityHostileCandidate]):
    @override
    def distance(
        self, left: EqualityHostileCandidate, right: EqualityHostileCandidate
    ) -> float:
        return float(abs(left.value - right.value))


def test_deferred_count_snapshot_retains_no_retired_candidates() -> None:
    retired = EqualityHostileCandidate(0)
    reference = ref(retired)
    current = BankDistanceWorkspace(
        entries=(
            BankEntry(candidate=retired, value=0.0),
            BankEntry(candidate=EqualityHostileCandidate(10), value=0.0),
        ),
        diversity_metric=HostileDistance(),
    )
    assert current.crowding_counts(distance_cutoff=5.0) == (0, 0)
    del retired
    for value in range(100):
        current = current.rebase(
            entries=(
                BankEntry(candidate=EqualityHostileCandidate(value), value=0.0),
                current.entries[1],
            ),
            invalidated_indices=frozenset(),
        )
    gc.collect()
    assert reference() is None
    assert current.crowding_counts(distance_cutoff=5.0) == (0, 0)
