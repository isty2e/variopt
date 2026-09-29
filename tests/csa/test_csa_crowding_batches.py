"""Exact cold crowding batches and their cache/customization boundaries."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import TypeVar

import numpy as np
import pytest
from typing_extensions import override

from variopt import (
    ArraySpace,
    CategoricalSpace,
    IntegerSpace,
    PermutationSpace,
    RealSpace,
    RecordSpace,
)
from variopt.algorithms.population.csa.banking.bank import BankEntry
from variopt.algorithms.population.csa.banking.queries import BankDistanceWorkspace
from variopt.diversity import StructuredSpaceDiversityMetric
from variopt.diversity.space_metric import CompiledStructuredDistanceView
from variopt.spaces.geometry.plan import (
    BuiltinStructuredGeometryPlan,
    EncodedStructuredCandidate,
)
from variopt.spaces.structured import StructuredSearchSpace
from variopt.spaces.types import SpaceCandidateValue

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)
RealCandidate = tuple[float, ...]


def _workspace(count: int = 64) -> BankDistanceWorkspace[RealCandidate]:
    metric = StructuredSpaceDiversityMetric(
        space=ArraySpace(RealSpace(0.0, 1.0), length=8)
    )
    return BankDistanceWorkspace(
        entries=tuple(
            BankEntry(candidate=(float(index % 5) / 4,) * 8, value=0.0)
            for index in range(count)
        ),
        diversity_metric=metric,
    )


def _scalar_counts(
    workspace: BankDistanceWorkspace[CandidateT], cutoff: float
) -> tuple[int, ...]:
    counts = [0] * len(workspace.entries)
    for left, right in combinations(range(len(counts)), 2):
        distance = workspace.diversity_metric.distance(
            workspace.entries[left].candidate, workspace.entries[right].candidate
        )
        if distance < cutoff:
            counts[left] += 1
            counts[right] += 1
    return tuple(counts)


@pytest.mark.parametrize("count", [0, 1, 8, 24, 32, 33, 64, 129])
@pytest.mark.parametrize(
    "cutoff", [0.0, 0.25, float(np.nextafter(0.25, 1)), float("inf"), float("nan")]
)
def test_cold_counts_match_scalar_across_batch_crossover(
    count: int, cutoff: float
) -> None:
    workspace = _workspace(count)
    assert workspace.crowding_counts(distance_cutoff=cutoff) == _scalar_counts(
        workspace, cutoff
    )


def test_cold_scan_batches_long_rows_instead_of_scalar_pair_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace()
    expected = _scalar_counts(workspace, 0.4)
    calls: list[tuple[EncodedStructuredCandidate, EncodedStructuredCandidate]] = []
    original = BuiltinStructuredGeometryPlan.squared_distance

    def record(
        self: BuiltinStructuredGeometryPlan[Sequence[float | int], RealCandidate],
        left: EncodedStructuredCandidate,
        right: EncodedStructuredCandidate,
    ) -> float:
        calls.append((left, right))
        return original(self, left, right)

    monkeypatch.setattr(BuiltinStructuredGeometryPlan, "squared_distance", record)
    assert workspace.crowding_counts(distance_cutoff=0.4) == expected
    assert len(calls) == 32 * 31 // 2


def test_missing_rows_preserve_seeded_zero_and_newer_endpoint_facts() -> None:
    workspace = _workspace()
    assert workspace.distance(0, 4) == 1.0
    workspace.seed_entry_distances(entry_index=2, distances=(0.0,) * 64)
    workspace.seed_entry_distances(entry_index=45, distances=(0.9,) * 64)
    expected = list(_scalar_counts(workspace, 0.5))
    for left, right in combinations(range(64), 2):
        if left not in (2, 45) and right not in (2, 45):
            continue
        ordinary = workspace.diversity_metric.distance(
            workspace.entries[left].candidate, workspace.entries[right].candidate
        )
        seeded = 0.9 if 45 in (left, right) else 0.0
        delta = int(seeded < 0.5) - int(ordinary < 0.5)
        expected[left] += delta
        expected[right] += delta
    assert workspace.crowding_counts(distance_cutoff=0.5) == tuple(expected)
    assert workspace.distance(0, 2) == 0.0
    assert workspace.distance(2, 45) == 0.9
    assert workspace.distance(0, 4) == 1.0


def test_warm_and_fully_seeded_scans_do_not_batch_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm = _workspace()
    warm.crowding_counts(distance_cutoff=0.5)
    seeded = _workspace()
    for index in range(64):
        seeded.seed_entry_distances(entry_index=index, distances=(0.0,) * 64)

    def unexpected(
        self: CompiledStructuredDistanceView[RealCandidate],
        left_index: int,
        right_indices: Sequence[int],
    ) -> tuple[float, ...]:
        pytest.fail("complete cached distances must not be recomputed")

    monkeypatch.setattr(CompiledStructuredDistanceView, "distances_from", unexpected)
    for cutoff in (0.5, 0.75, 0.0):
        assert warm.crowding_counts(distance_cutoff=cutoff) == _scalar_counts(
            warm, cutoff
        )
    assert seeded.crowding_counts(distance_cutoff=0.5) == (63,) * 64


class FixedWorkspace(BankDistanceWorkspace[RealCandidate]):
    @override
    def distance(self, left_index: int, right_index: int) -> float:
        return 0.1


def test_compiled_workspace_subclass_retains_its_distance_override() -> None:
    base = _workspace()
    workspace = FixedWorkspace(
        entries=base.entries, diversity_metric=base.diversity_metric
    )
    assert workspace.compiled_distance_view is not None
    assert workspace.crowding_counts(distance_cutoff=0.2) == (63,) * 64


def test_failed_batch_can_retry_without_publishing_partial_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace()
    expected = _scalar_counts(workspace, 0.5)
    original = CompiledStructuredDistanceView[RealCandidate].distances_from
    calls: list[int] = []

    def fail_second(
        self: CompiledStructuredDistanceView[RealCandidate],
        left_index: int,
        right_indices: Sequence[int],
    ) -> tuple[float, ...]:
        calls.append(left_index)
        if len(calls) == 2:
            raise RuntimeError("batch failure")
        return original(self, left_index, right_indices)

    monkeypatch.setattr(CompiledStructuredDistanceView, "distances_from", fail_second)
    with pytest.raises(RuntimeError, match="batch failure"):
        workspace.crowding_counts(distance_cutoff=0.5)
    assert workspace.crowding_counts(distance_cutoff=0.5) == expected
    assert workspace.crowding_counts(distance_cutoff=0.5) == expected


def test_partial_cold_branches_keep_old_and_new_distances_independent() -> None:
    original = _workspace()
    original.distance(0, 1)
    left = original.rebase(
        entries=(BankEntry(candidate=(0.1,) * 8, value=0.0), *original.entries[1:]),
        invalidated_indices=frozenset({0}),
    )
    right = original.rebase(
        entries=(*original.entries[:-1], BankEntry(candidate=(0.2,) * 8, value=0.0)),
        invalidated_indices=frozenset({63}),
    )
    for workspace in (left, original, right, original, left):
        assert workspace.crowding_counts(distance_cutoff=0.25) == _scalar_counts(
            workspace, 0.25
        )


def test_batched_counts_survive_growth_removal_reordering_and_score_only_updates() -> (
    None
):
    current = _workspace()
    retained: list[BankDistanceWorkspace[RealCandidate]] = []
    for step in range(15):
        assert current.crowding_counts(distance_cutoff=0.4) == _scalar_counts(
            current, 0.4
        )
        retained.append(current)
        entries = list(current.entries)
        operation = step % 5
        if operation == 0:
            entries[3] = BankEntry(candidate=(0.1 + step / 100,) * 8, value=0.0)
        elif operation == 1:
            entries.append(BankEntry(candidate=(0.8,) * 8, value=0.0))
        elif operation == 2:
            del entries[1]
        elif operation == 3:
            entries.reverse()
        else:
            entries = [
                BankEntry(candidate=entry.candidate, value=-1.0) for entry in entries
            ]
        current = current.rebase(entries=entries, invalidated_indices=frozenset())

    for workspace in (*retained, current):
        for cutoff in (0.4, 0.25):
            assert workspace.crowding_counts(distance_cutoff=cutoff) == _scalar_counts(
                workspace, cutoff
            )


@dataclass(frozen=True)
class RecordingMetric(
    StructuredSpaceDiversityMetric[Sequence[float | int], RealCandidate]
):
    calls: list[tuple[RealCandidate, RealCandidate]] = field(default_factory=list)

    @override
    def distance(self, left: RealCandidate, right: RealCandidate) -> float:
        self.calls.append((left, right))
        if len(self.calls) == 2:
            raise RuntimeError("custom metric failure")
        return abs(left[0] - right[0])


def test_large_custom_metric_retains_callback_order_and_retry_prefix() -> None:
    base = _workspace()
    metric = RecordingMetric(space=ArraySpace(RealSpace(0.0, 1.0), length=8))
    workspace = BankDistanceWorkspace(entries=base.entries, diversity_metric=metric)
    assert workspace.compiled_distance_view is None
    expected = _scalar_counts(base, 0.4)
    with pytest.raises(RuntimeError, match="custom metric failure"):
        workspace.crowding_counts(distance_cutoff=0.4)
    assert workspace.crowding_counts(distance_cutoff=0.4) == expected
    pairs = [
        (base.entries[left].candidate, base.entries[right].candidate)
        for left, right in combinations(range(64), 2)
    ]
    assert metric.calls == [*pairs[:2], *pairs[1:]]


def test_separate_workspaces_can_batch_shared_encodings_concurrently() -> None:
    base = _workspace()
    view = base.compiled_distance_view
    assert view is not None
    expected = _scalar_counts(base, 0.4)
    workspaces = tuple(
        BankDistanceWorkspace(
            entries=base.entries,
            diversity_metric=base.diversity_metric,
            compiled_distance_view=view,
        )
        for _ in range(8)
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(workspace.crowding_counts, distance_cutoff=0.4)
            for workspace in workspaces
        ]
        assert [future.result() for future in futures] == [expected] * len(workspaces)
    assert base.crowding_counts(distance_cutoff=0.4) == expected


def test_cold_branches_can_memoize_shared_unchanged_pairs_concurrently() -> None:
    base = _workspace()
    base.distance(0, 1)
    workspaces: list[BankDistanceWorkspace[RealCandidate]] = []
    for index in range(8):
        entries = list(base.entries)
        entries[index] = BankEntry(candidate=(0.1 + index / 20,) * 8, value=0.0)
        workspaces.append(
            base.rebase(entries=entries, invalidated_indices=frozenset({index}))
        )
    expected = [_scalar_counts(workspace, 0.4) for workspace in workspaces]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(workspace.crowding_counts, distance_cutoff=0.4)
            for workspace in workspaces
        ]
        assert [future.result() for future in futures] == expected
    assert base.crowding_counts(distance_cutoff=0.4) == _scalar_counts(base, 0.4)


def _assert_sampled_counts(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
) -> None:
    random_state = np.random.RandomState(17)
    metric = StructuredSpaceDiversityMetric(space=space)
    workspace = BankDistanceWorkspace(
        entries=tuple(
            BankEntry(candidate=space.sample(random_state), value=0.0)
            for _ in range(72)
        ),
        diversity_metric=metric,
    )
    expected = _scalar_counts(workspace, 0.3)
    with np.errstate(all="raise"):
        assert workspace.crowding_counts(distance_cutoff=0.3) == expected


def test_cold_counts_cover_mixed_logarithmic_and_discrete_geometry() -> None:
    _assert_sampled_counts(ArraySpace(IntegerSpace(-100, 100), length=8))
    _assert_sampled_counts(ArraySpace(RealSpace(0.001, 10.0, scale="log"), length=8))
    _assert_sampled_counts(ArraySpace(CategoricalSpace(("a", "b", "c")), length=8))
    _assert_sampled_counts(
        ArraySpace(
            RecordSpace(
                real=RealSpace(-10.0, 10.0),
                integer=IntegerSpace(-10, 10),
                category=CategoricalSpace((1, 2, 3)),
                permutation=PermutationSpace(4),
            ),
            length=8,
        )
    )


@pytest.mark.skipif(np.iinfo("l").bits < 64, reason="requires a 64-bit C-long sampler")
def test_large_integer_geometry_keeps_its_scalar_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(
        self: CompiledStructuredDistanceView[tuple[int, ...]],
        left_index: int,
        right_indices: Sequence[int],
    ) -> tuple[float, ...]:
        pytest.fail("integer geometry outside int64 batching must remain scalar")

    monkeypatch.setattr(CompiledStructuredDistanceView, "distances_from", unexpected)
    space = ArraySpace(IntegerSpace(-(2**63), 2**63 - 1), length=8)
    workspace = BankDistanceWorkspace(
        entries=tuple(
            BankEntry(candidate=space.normalize((value,) * 8), value=0.0)
            for value in (-(2**63), -1, 0, 2**63 - 1) * 16
        ),
        diversity_metric=StructuredSpaceDiversityMetric(space=space),
    )
    assert workspace.crowding_counts(distance_cutoff=0.3) == _scalar_counts(
        workspace, 0.3
    )


@dataclass(frozen=True)
class NonfiniteMetric(
    StructuredSpaceDiversityMetric[Sequence[float | int], RealCandidate]
):
    @override
    def distance(self, left: RealCandidate, right: RealCandidate) -> float:
        return float("inf") if 1.0 in (left[0], right[0]) else 0.0


def test_nonfinite_metric_rejects_counts_and_allows_rebased_recovery() -> None:
    metric = NonfiniteMetric(space=ArraySpace(RealSpace(0.0, 1.0), length=8))
    workspace = BankDistanceWorkspace(
        entries=tuple(
            BankEntry(candidate=(value,) * 8, value=0.0)
            for value in (0.0, 1.0, *((0.0,) * 62))
        ),
        diversity_metric=metric,
    )
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="finite"):
            workspace.distance(0, 1)
        for _ in range(2):
            with pytest.raises(ValueError, match="finite"):
                workspace.crowding_counts(distance_cutoff=0.1)

        recovered = workspace.rebase(
            entries=(
                workspace.entries[0],
                BankEntry(candidate=(0.0,) * 8, value=0.0),
                *workspace.entries[2:],
            ),
            invalidated_indices=frozenset({1}),
        )
        assert recovered.crowding_counts(distance_cutoff=0.1) == (63,) * 64
        with pytest.raises(ValueError, match="finite"):
            workspace.crowding_counts(distance_cutoff=0.1)
