"""Snapshot and mutation contracts for operation-local CSA distance caches."""

import gc
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from weakref import ref

import numpy as np
import pytest
from typing_extensions import override

from variopt import ArraySpace, RealSpace
from variopt.algorithms.population.csa.banking.bank import BankEntry
from variopt.algorithms.population.csa.banking.queries import (
    BankDistanceWorkspace,
    distances_to_candidate,
)
from variopt.diversity import DiversityMetric, StructuredSpaceDiversityMetric


@dataclass
class RecordingDistance(DiversityMetric[int]):
    calls: list[tuple[int, int]] = field(default_factory=list)
    fail_next: bool = False

    @override
    def distance(self, left: int, right: int) -> float:
        self.calls.append((left, right))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("metric failure")
        return float(abs(left - right))


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


class FixedStructuredDistance(
    StructuredSpaceDiversityMetric[Sequence[float | int], tuple[float, ...]]
):
    @override
    def distance(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        return 0.75


@dataclass
class FixedInvalidDistance(DiversityMetric[int]):
    result: float
    calls: list[tuple[int, int]] = field(default_factory=list)

    @override
    def distance(self, left: int, right: int) -> float:
        self.calls.append((left, right))
        return self.result if right == 2 else float(abs(left - right))


def _workspace(*candidates: int) -> BankDistanceWorkspace[int]:
    return BankDistanceWorkspace(
        entries=tuple(
            BankEntry(candidate=value, value=float(value)) for value in candidates
        ),
        diversity_metric=RecordingDistance(),
    )


def _assert_distances(workspace: BankDistanceWorkspace[int]) -> None:
    for left, right in combinations(range(len(workspace.entries)), 2):
        expected = float(
            abs(workspace.entries[left].candidate - workspace.entries[right].candidate)
        )
        assert workspace.distance(left, right) == expected
        assert workspace.distance(right, left) == expected


def test_compiled_candidate_batch_preserves_ties_cutoffs_and_rebase() -> None:
    space = ArraySpace(RealSpace(0.0, 1.0), length=4)
    metric = StructuredSpaceDiversityMetric(space=space)
    entries = tuple(
        BankEntry(candidate=(value,) * 4, value=float(index))
        for index, value in enumerate((0.0, 0.5, 1.0, 0.5) * 32)
    )
    workspace = BankDistanceWorkspace(entries=entries, diversity_metric=metric)
    candidate = (0.25,) * 4
    expected = tuple(metric.distance(candidate, entry.candidate) for entry in entries)

    distances = workspace.distances_to_candidate(candidate)
    assert distances == expected
    assert min(range(len(distances)), key=distances.__getitem__) == 0
    assert tuple(distance < 0.25 for distance in distances) == (False,) * 128
    assert (
        sum(distance < float(np.nextafter(0.25, 1.0)) for distance in distances) == 96
    )

    original_counts = workspace.crowding_counts(distance_cutoff=0.25)
    updated_entries = (BankEntry(candidate=candidate, value=-1.0), *entries[1:])
    updated = workspace.rebase(
        entries=updated_entries, invalidated_indices=frozenset({0})
    )
    updated.seed_entry_distances(entry_index=0, distances=distances)
    assert tuple(updated.distance(0, index) for index in range(1, 128)) == distances[1:]
    for cutoff in (0.25, float(np.nextafter(0.25, 1.0))):
        assert updated.crowding_counts(distance_cutoff=cutoff) == tuple(
            sum(
                metric.distance(entry.candidate, other.candidate) < cutoff
                for other_index, other in enumerate(updated_entries)
                if index != other_index
            )
            for index, entry in enumerate(updated_entries)
        )
    assert workspace.distances_to_candidate(candidate) == expected
    assert workspace.crowding_counts(distance_cutoff=0.25) == original_counts
    assert workspace.distance(0, 1) == 0.5


def test_candidate_batch_preserves_custom_structured_metric_override() -> None:
    space = ArraySpace(RealSpace(0.0, 128.0), length=4)
    metric = FixedStructuredDistance(space=space)
    workspace = BankDistanceWorkspace(
        entries=tuple(
            BankEntry(candidate=(float(index),) * 4, value=0.0) for index in range(128)
        ),
        diversity_metric=metric,
    )

    assert workspace.distances_to_candidate((64.0,) * 4) == (0.75,) * 128
    assert (
        distances_to_candidate(
            entries=workspace.entries,
            diversity_metric=metric,
            candidate=(64.0,) * 4,
        )
        == (0.75,) * 128
    )


def test_plain_candidate_batch_preserves_order_duplicates_and_empty_bank() -> None:
    metric = RecordingDistance()
    entries = tuple(BankEntry(candidate=value, value=0.0) for value in (3, 1, 3, 0))

    assert distances_to_candidate(
        entries=entries, diversity_metric=metric, candidate=1
    ) == (2.0, 0.0, 2.0, 1.0)
    assert metric.calls == [(1, 3), (1, 1), (1, 3), (1, 0)]
    metric.calls.clear()
    assert (
        distances_to_candidate(entries=(), diversity_metric=metric, candidate=1) == ()
    )
    assert metric.calls == []


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_candidate_batch_rejects_invalid_custom_distance_after_valid_prefix(
    invalid: float,
) -> None:
    metric = FixedInvalidDistance(result=invalid)
    entries = tuple(BankEntry(candidate=value, value=0.0) for value in (1, 2, 3))
    workspace = BankDistanceWorkspace(entries=entries, diversity_metric=metric)

    for query in (
        lambda: distances_to_candidate(
            entries=entries, diversity_metric=metric, candidate=0
        ),
        lambda: workspace.distances_to_candidate(0),
    ):
        metric.calls.clear()
        with pytest.raises(ValueError, match="distance must be"):
            query()
        assert metric.calls == [(0, 1), (0, 2)]


def test_candidate_batch_failure_does_not_poison_retry_or_cached_pairs() -> None:
    metric = RecordingDistance()
    entries = tuple(BankEntry(candidate=value, value=0.0) for value in (3, 1, 5))
    workspace = BankDistanceWorkspace(entries=entries, diversity_metric=metric)
    assert workspace.distance(0, 1) == 2.0
    metric.calls.clear()
    metric.fail_next = True

    with pytest.raises(RuntimeError, match="metric failure"):
        workspace.distances_to_candidate(2)
    assert metric.calls == [(2, 3)]
    assert workspace.distances_to_candidate(2) == (1.0, 1.0, 3.0)
    assert workspace.distance(0, 1) == 2.0
    assert metric.calls == [(2, 3), (2, 3), (2, 1), (2, 5)]


def test_candidate_batch_never_compares_candidate_equality() -> None:
    candidate = EqualityHostileCandidate(2)
    entries = tuple(
        BankEntry(candidate=value, value=0.0)
        for value in (
            EqualityHostileCandidate(1),
            candidate,
            EqualityHostileCandidate(1),
        )
    )
    metric = HostileDistance()
    workspace = BankDistanceWorkspace(entries=entries, diversity_metric=metric)

    assert distances_to_candidate(
        entries=entries, diversity_metric=metric, candidate=candidate
    ) == (1.0, 0.0, 1.0)
    assert workspace.distances_to_candidate(candidate) == (1.0, 0.0, 1.0)


def test_scalar_candidate_batch_preserves_cutoff_ulps_and_retained_workspace() -> None:
    metric = StructuredSpaceDiversityMetric(space=RealSpace(0.0, 1.0))
    entries = tuple(BankEntry(candidate=value, value=0.0) for value in (0.0, 0.5, 0.0))
    workspace = BankDistanceWorkspace(entries=entries, diversity_metric=metric)
    candidate = 0.25
    distances = workspace.distances_to_candidate(candidate)

    assert distances == (0.25, 0.25, 0.25)
    assert min(range(len(distances)), key=distances.__getitem__) == 0
    assert not any(distance < 0.25 for distance in distances)
    assert all(distance < float(np.nextafter(0.25, 1.0)) for distance in distances)
    updated = workspace.rebase(
        entries=(BankEntry(candidate=candidate, value=-1.0), *entries[1:]),
        invalidated_indices=frozenset({0}),
    )
    updated.seed_entry_distances(entry_index=0, distances=distances)
    assert updated.distance(0, 1) == 0.25
    assert workspace.distance(0, 1) == 0.5
    assert workspace.distances_to_candidate(candidate) == distances


def test_rebase_preserves_old_and_new_answers_after_late_cache_misses() -> None:
    original = _workspace(0, 10, 30, 50)
    assert original.distance(0, 1) == 10.0
    changed = original.rebase(
        entries=(
            *original.entries[:2],
            BankEntry(candidate=40, value=1.0),
            original.entries[3],
        ),
        invalidated_indices=frozenset({2}),
    )

    _assert_distances(changed)
    _assert_distances(original)
    _assert_distances(changed)


def test_sibling_branches_and_repeated_slot_changes_remain_independent() -> None:
    original = _workspace(0, 10, 30, 50)
    _assert_distances(original)
    branches = [original, original]
    retained = [original]
    for step in range(30):
        branch = step % 2
        slot = (step // 2) % 4
        entries = list(branches[branch].entries)
        entries[slot] = BankEntry(candidate=100 + step, value=float(step))
        branches[branch] = branches[branch].rebase(
            entries=entries, invalidated_indices=frozenset({slot})
        )
        retained.append(branches[branch])
        for workspace in reversed(retained):
            _assert_distances(workspace)


def test_seeding_a_branch_does_not_overwrite_retained_rows() -> None:
    original = _workspace(0, 10, 30)
    _assert_distances(original)
    branch = original.rebase(
        entries=tuple(
            BankEntry(candidate=entry.candidate, value=-1.0)
            for entry in original.entries
        ),
        invalidated_indices=frozenset(),
    )
    branch.seed_entry_distances(entry_index=1, distances=(7.0, float("nan"), 9.0))

    assert branch.distance(0, 1) == 7.0
    assert branch.distance(1, 2) == 9.0
    assert branch.distance(0, 2) == 30.0
    _assert_distances(original)

    original.seed_entry_distances(entry_index=2, distances=(3.0, 2.0, 0.0))
    assert branch.distance(0, 2) == 30.0
    assert branch.distance(1, 2) == 9.0


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_invalid_seed_is_atomic_even_after_valid_prefix(invalid: float) -> None:
    workspace = _workspace(0, 10, 30)
    _assert_distances(workspace)

    with pytest.raises(ValueError):
        workspace.seed_entry_distances(entry_index=0, distances=(0.0, 777.0, invalid))

    _assert_distances(workspace)


def test_append_remove_and_empty_rebuild_keep_retained_snapshots_valid() -> None:
    original = _workspace(0, 10, 30)
    _assert_distances(original)
    appended = original.rebase(
        entries=(*original.entries, BankEntry(candidate=70, value=0.0)),
        invalidated_indices=frozenset({3}),
    )
    appended.seed_entry_distances(entry_index=3, distances=(70.0, 60.0, 40.0, 0.0))
    removed = appended.rebase(
        entries=(appended.entries[3], appended.entries[1]),
        invalidated_indices=frozenset(),
    )
    empty = removed.rebase(entries=(), invalidated_indices=frozenset())
    rebuilt = empty.rebase(entries=original.entries, invalidated_indices=frozenset())

    for workspace in (rebuilt, removed, appended, original):
        _assert_distances(workspace)
    assert empty.crowding_counts(distance_cutoff=1.0) == ()
    assert empty.average_pairwise_distance() == 0.0


def test_explicit_invalidation_recomputes_same_identity_only_in_new_snapshot() -> None:
    original = _workspace(0, 10, 30)
    original.seed_entry_distances(entry_index=1, distances=(7.0, 0.0, 9.0))
    changed = original.rebase(
        entries=original.entries, invalidated_indices=frozenset({1})
    )

    _assert_distances(changed)
    assert original.distance(0, 1) == 7.0
    assert original.distance(1, 2) == 9.0


def test_same_size_reorder_detects_identity_changes_without_hints() -> None:
    original = _workspace(0, 10, 30, 80)
    _assert_distances(original)
    reordered = original.rebase(
        entries=(
            original.entries[3],
            original.entries[1],
            original.entries[0],
            original.entries[2],
        ),
        invalidated_indices=frozenset({-1, 99}),
    )
    _assert_distances(reordered)
    _assert_distances(original)


def test_unchanged_pair_is_not_recomputed_after_replacement_or_score_update() -> None:
    metric = RecordingDistance()
    original = BankDistanceWorkspace(
        entries=tuple(BankEntry(candidate=value, value=0.0) for value in (0, 10, 30)),
        diversity_metric=metric,
    )
    _assert_distances(original)
    changed = original.rebase(
        entries=(
            original.entries[0],
            original.entries[1],
            BankEntry(candidate=40, value=0.0),
        ),
        invalidated_indices=frozenset({2}),
    )
    rescored = changed.rebase(
        entries=tuple(
            BankEntry(candidate=entry.candidate, value=99.0)
            for entry in changed.entries
        ),
        invalidated_indices=frozenset(),
    )
    assert rescored.distance(1, 0) == 10.0
    assert len(metric.calls) == 3
    _assert_distances(rescored)
    assert len(metric.calls) == 5
    _assert_distances(original)
    assert len(metric.calls) == 5


def test_pair_argument_order_does_not_follow_row_revision() -> None:
    metric = RecordingDistance()
    original = BankDistanceWorkspace(
        entries=(BankEntry(candidate=9, value=0.0), BankEntry(candidate=2, value=0.0)),
        diversity_metric=metric,
    )
    assert original.distance(1, 0) == 7.0
    changed = original.rebase(
        entries=(original.entries[0], BankEntry(candidate=3, value=0.0)),
        invalidated_indices=frozenset({1}),
    )
    assert changed.distance(1, 0) == 6.0
    assert metric.calls == [(9, 2), (9, 3)]


def test_failed_distance_is_retryable_and_does_not_corrupt_other_pairs() -> None:
    metric = RecordingDistance()
    workspace = BankDistanceWorkspace(
        entries=tuple(BankEntry(candidate=value, value=0.0) for value in (0, 10, 30)),
        diversity_metric=metric,
    )
    assert workspace.distance(0, 1) == 10.0
    metric.fail_next = True
    with pytest.raises(RuntimeError, match="metric failure"):
        workspace.distance(2, 0)
    assert workspace.distance(0, 1) == 10.0
    assert workspace.distance(0, 2) == 30.0
    assert metric.calls == [(0, 10), (0, 30), (0, 30)]


@pytest.mark.parametrize("seed", [0, 11, 47])
def test_seeded_mutation_walk_matches_fresh_workspace(seed: int) -> None:
    rng = np.random.RandomState(seed)
    current = _workspace(0, 0, 10, 30, 90)
    retained = []
    for step in range(120):
        retained.append(current)
        entries = list(current.entries)
        if step % 7 == 0 and len(entries) > 1:
            del entries[int(rng.randint(len(entries)))]
        elif step % 5 == 0 or not entries:
            entries.append(BankEntry(candidate=int(rng.randint(20)), value=0.0))
        else:
            slot = int(rng.randint(len(entries)))
            entries[slot] = BankEntry(candidate=int(rng.randint(20)), value=0.0)
        current = current.rebase(entries=entries, invalidated_indices=frozenset())
        if entries and step % 3 == 0:
            slot = int(rng.randint(len(entries)))
            current.seed_entry_distances(
                entry_index=slot,
                distances=tuple(
                    float(abs(entries[slot].candidate - entry.candidate))
                    for entry in entries
                ),
            )
        _assert_distances(current)
        fresh = BankDistanceWorkspace(
            entries=entries, diversity_metric=RecordingDistance()
        )
        cutoff = float(step % 10)
        assert current.crowding_counts(distance_cutoff=cutoff) == fresh.crowding_counts(
            distance_cutoff=cutoff
        )
        assert current.average_pairwise_distance() == fresh.average_pairwise_distance()
        _assert_distances(retained[int(rng.randint(len(retained)))])


def test_replacement_releases_candidate_history_without_using_equality() -> None:
    retired = EqualityHostileCandidate(0)
    reference = ref(retired)
    current = BankDistanceWorkspace(
        entries=(
            BankEntry(candidate=retired, value=0.0),
            BankEntry(candidate=EqualityHostileCandidate(10), value=0.0),
        ),
        diversity_metric=HostileDistance(),
    )
    assert current.distance(0, 1) == 10.0
    del retired
    for value in range(1, 101):
        current = current.rebase(
            entries=(
                BankEntry(candidate=EqualityHostileCandidate(value), value=0.0),
                current.entries[1],
            ),
            invalidated_indices=frozenset(),
        )
        assert current.distance(0, 1) == float(abs(value - 10))
    gc.collect()
    assert reference() is None


def test_zero_and_cutoff_equality_after_seeding_and_rebase() -> None:
    original = _workspace(0, 0, 2)
    original.seed_entry_distances(entry_index=1, distances=(0.0, 0.0, 2.0))
    rebased = original.rebase(
        entries=(*original.entries,), invalidated_indices=frozenset({2})
    )
    assert rebased.crowding_counts(distance_cutoff=0.0) == (0, 0, 0)
    assert rebased.crowding_counts(distance_cutoff=2.0) == (1, 1, 0)
    assert rebased.distance(1, 0) == 0.0


def test_seeded_distances_survive_append_to_singleton_and_multiple_invalidations() -> (
    None
):
    original = _workspace(10)
    original.seed_entry_distances(entry_index=0, distances=(float("nan"),))
    grown = original.rebase(
        entries=(*original.entries, BankEntry(candidate=30, value=0.0)),
        invalidated_indices=frozenset({1}),
    )
    grown.seed_entry_distances(entry_index=1, distances=(20.0, 0.0))
    changed = grown.rebase(
        entries=(
            BankEntry(candidate=50, value=0.0),
            BankEntry(candidate=40, value=0.0),
        ),
        invalidated_indices=frozenset({0, 1}),
    )
    assert changed.distance(0, 1) == 10.0
    assert grown.distance(0, 1) == 20.0
    assert original.distance(0, 0) == 0.0
