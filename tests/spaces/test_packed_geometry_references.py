"""Numerical and snapshot-ownership contracts for reusable packed references."""

import gc
import pickle
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from typing import TypeVar, cast
from unittest.mock import patch
from weakref import ref

import numpy as np
import pytest

from variopt import (
    ArraySpace,
    CategoricalSpace,
    IntegerSpace,
    PermutationSpace,
    RealSpace,
    RecordSpace,
    TupleSpace,
)
from variopt.diversity.space_metric import CompiledStructuredDistanceView
from variopt.spaces.geometry.plan import (
    PackedStructuredReferences,
    compile_builtin_geometry_plan,
)
from variopt.spaces.structured import StructuredSearchSpace
from variopt.spaces.types import SpaceCandidateValue

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)


def _assert_packed_parity(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
) -> None:
    plan = compile_builtin_geometry_plan(space)
    assert plan is not None
    random_state = np.random.RandomState(29)
    encodings = plan.encode_many(tuple(space.sample(random_state) for _ in range(513)))
    packed = plan.pack_references(encodings)
    assert packed is not None
    assert packed.encodings is encodings

    expected = tuple(plan.squared_distance(encodings[0], item) for item in encodings)
    assert plan.squared_distances_to_packed(encodings[0], packed) == expected
    for indices in (
        (),
        (0,),
        (1, 1),
        tuple(range(17)),
        tuple(range(31)),
        tuple(range(32)),
        tuple(range(513)),
        tuple(range(512, -1, -1)),
        tuple(range(0, 513, 2)),
        (5, 1, 5, 512) * 129,
    ):
        assert plan.squared_distances_to_packed(
            encodings[0], packed, reference_indices=deque(indices)
        ) == tuple(expected[index] for index in indices)


def test_packed_real_and_log_arithmetic() -> None:
    _assert_packed_parity(ArraySpace(RealSpace(-5.0, 5.0), length=32))
    _assert_packed_parity(ArraySpace(RealSpace(1e-100, 1e100, scale="log"), length=16))


def test_packed_integer_and_discrete_arithmetic() -> None:
    _assert_packed_parity(ArraySpace(IntegerSpace(2**53, 2**53 + 100), length=4))
    _assert_packed_parity(ArraySpace(IntegerSpace(1, 1000, scale="log"), length=16))
    _assert_packed_parity(ArraySpace(IntegerSpace(0, 1), length=32))
    _assert_packed_parity(ArraySpace(CategoricalSpace((b"a", b"b")), length=8))
    _assert_packed_parity(PermutationSpace(32))


def test_packed_nested_mixed_geometry() -> None:
    _assert_packed_parity(
        ArraySpace(
            RecordSpace(
                scale=RealSpace(0.1, 10.0, scale="log"),
                values=TupleSpace(
                    IntegerSpace(-8, 8),
                    CategoricalSpace((True, False)),
                    RealSpace(-3.0, 3.0),
                ),
                order=PermutationSpace(7),
                fixed=IntegerSpace(5, 5),
            ),
            length=8,
        )
    )


@pytest.mark.parametrize("reference_count", [0, 1, 31, 32, 63, 64, 128])
def test_small_reference_snapshot_keeps_scalar_crossover(reference_count: int) -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(IntegerSpace(0, 10), length=4))
    assert plan is not None
    encoding = plan.encode((0, 1, 2, 3))
    packed = plan.pack_references(deque((encoding,) * reference_count))
    if reference_count < 64:
        assert packed is None
    else:
        assert packed is not None
        assert (
            plan.squared_distances_to_packed(encoding, packed)
            == (0.0,) * reference_count
        )


def _assert_readonly(packed: PackedStructuredReferences) -> None:
    for array in (
        packed._batch.real_values,
        packed._batch.integer_values,
        packed._batch.discrete_values,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[0, 0] = 9
        assert isinstance(array.base, np.ndarray)
        with pytest.raises(ValueError):
            array.base.setflags(write=True)


@pytest.mark.parametrize("protocol", [4, 5])
def test_packed_arrays_stay_immutable_across_pickle(protocol: int) -> None:
    plan = compile_builtin_geometry_plan(
        TupleSpace(
            RealSpace(0.0, 1.0), IntegerSpace(0, 8), CategoricalSpace(("a", "b"))
        )
    )
    assert plan is not None
    candidate = plan.encode((0.5, 3, "a"))
    references = plan.pack_references((candidate,) * 128)
    assert references is not None
    _assert_readonly(references)

    restored_plan, restored_candidate, restored_references = pickle.loads(
        pickle.dumps((plan, candidate, references), protocol=protocol)
    )
    _assert_readonly(restored_references)
    assert (
        restored_plan.squared_distances_to_packed(
            restored_candidate, restored_references
        )
        == (0.0,) * 128
    )
    with pytest.raises(ValueError, match="different geometry plan"):
        plan.squared_distances_to_packed(candidate, restored_references)


@pytest.mark.parametrize("indices", [(-1,), (128,), tuple(range(127)) + (128,)])
def test_packed_query_rejects_invalid_rows_before_evaluation(
    indices: tuple[int, ...],
) -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=4))
    assert plan is not None
    candidate = plan.encode((0.0,) * 4)
    packed = plan.pack_references((candidate,) * 128)
    assert packed is not None
    with pytest.raises(IndexError, match="packed snapshot"):
        plan.squared_distances_to_packed(candidate, packed, reference_indices=indices)


def test_packed_query_rejects_foreign_plans_even_with_empty_selection() -> None:
    space = ArraySpace(RealSpace(0.0, 1.0), length=4)
    plan = compile_builtin_geometry_plan(space)
    other = compile_builtin_geometry_plan(space)
    assert plan is not None and other is not None
    candidate = plan.encode((0.0,) * 4)
    foreign = other.encode((0.0,) * 4)
    packed = plan.pack_references((candidate,) * 128)
    assert packed is not None
    with pytest.raises(ValueError, match="different geometry plan"):
        plan.squared_distances_to_packed(foreign, packed, reference_indices=())
    with pytest.raises(ValueError, match="different geometry plan"):
        other.squared_distances_to_packed(foreign, packed, reference_indices=())
    for references in ((foreign,), (candidate,) * 257 + (foreign,)):
        with pytest.raises(ValueError, match="different geometry plan"):
            plan.pack_references(references)
    with pytest.raises(ValueError, match="one geometry plan"):
        PackedStructuredReferences((candidate, foreign))
    with pytest.raises(ValueError, match="at least one"):
        PackedStructuredReferences(())


def test_packed_view_does_not_truncate_noninteger_indices() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=4))
    assert plan is not None
    view = CompiledStructuredDistanceView.from_candidates(
        plan=plan, candidates=((0.0,) * 4,) * 128
    )
    invalid_indices = cast(Sequence[int], (1.5,) * 128)
    with pytest.raises(TypeError):
        view.distances_from(0, invalid_indices)


def test_packed_view_queries_do_not_repack_and_preserve_self_rows() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 128.0), length=4))
    assert plan is not None
    candidates = tuple((float(index),) * 4 for index in range(128))
    view = CompiledStructuredDistanceView.from_candidates(
        plan=plan, candidates=candidates
    )
    assert view._packed_references is not None
    indices = tuple(range(128)) + (0, 0, 127)
    expected = tuple(index / 128.0 for index in range(128))

    with patch(
        "variopt.spaces.geometry.plan._EncodedCandidateBatch.from_encodings",
        side_effect=AssertionError("unexpected repacking"),
    ):
        assert view.distances_to(candidates[0]) == expected
        assert view.distances_from(0, indices) == expected + (0.0, 0.0, expected[-1])
        assert view.distances_from(0, (0,) * 128) == (0.0,) * 128
        assert view.distances_from(0, ()) == ()
        assert (
            view.rebase(candidates=candidates, invalidated_indices=frozenset()) is view
        )


def test_packed_view_rebase_keeps_retained_branches_independent() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 128.0), length=4))
    assert plan is not None
    candidates = tuple((float(index),) * 4 for index in range(128))
    view = CompiledStructuredDistanceView.from_candidates(
        plan=plan, candidates=candidates
    )
    baseline = view.distances_to(candidates[0])
    for next_candidates, invalidated in (
        ((candidates[-1],) + candidates[1:], frozenset({0})),
        (candidates, frozenset({17})),
        (tuple((*candidate,) for candidate in candidates), frozenset()),
        (candidates + (candidates[0],), frozenset()),
        (candidates[1:], frozenset()),
        (candidates[::-1], frozenset()),
        (candidates[:7], frozenset()),
        ((), frozenset()),
    ):
        branch = view.rebase(
            candidates=next_candidates, invalidated_indices=invalidated
        )
        fresh = CompiledStructuredDistanceView.from_candidates(
            plan=plan, candidates=next_candidates
        )
        assert branch is not view
        assert branch.is_aligned_with(next_candidates)
        assert branch.distances_to(candidates[0]) == fresh.distances_to(candidates[0])
        assert view.distances_to(candidates[0]) == baseline
        if branch._packed_references is not None:
            assert view._packed_references is not None
            assert not np.shares_memory(
                branch._packed_references._batch.real_values,
                view._packed_references._batch.real_values,
            )


def test_packed_rebase_failure_leaves_original_view_usable() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=4))
    assert plan is not None
    candidates = ((0.0,) * 4,) * 128
    view = CompiledStructuredDistanceView.from_candidates(
        plan=plan, candidates=candidates
    )
    with patch(
        "variopt.spaces.geometry.plan._EncodedCandidateBatch.from_encodings",
        side_effect=MemoryError("packing failed"),
    ):
        with pytest.raises(MemoryError, match="packing failed"):
            view.rebase(candidates=candidates, invalidated_indices=frozenset({0}))
        assert view.distances_to(candidates[0]) == (0.0,) * 128


def test_packed_view_pickle_and_replace_preserve_alignment() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 128.0), length=4))
    assert plan is not None
    view = CompiledStructuredDistanceView.from_candidates(
        plan=plan, candidates=tuple((float(index),) * 4 for index in range(128))
    )
    restored = pickle.loads(pickle.dumps(view))
    assert restored.distances_from(0, tuple(range(128))) == view.distances_from(
        0, tuple(range(128))
    )
    assert replace(view).distances_to((0.0,) * 4) == restored.distances_to((0.0,) * 4)
    with pytest.raises(ValueError, match="encodings must align"):
        replace(view, candidates=())


def test_packed_view_concurrent_queries_do_not_mutate_arrays() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=32))
    assert plan is not None
    rng = np.random.RandomState(47)
    view = CompiledStructuredDistanceView.from_candidates(
        plan=plan, candidates=tuple(plan.space.sample(rng) for _ in range(128))
    )
    expected = tuple(view.distances_to(candidate) for candidate in view.candidates[:16])
    assert view._packed_references is not None
    before = view._packed_references._batch.real_values.tobytes()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert tuple(pool.map(view.distances_to, view.candidates[:16])) == expected
    assert view._packed_references._batch.real_values.tobytes() == before


def test_packed_view_can_cross_a_spawned_process_boundary() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=4))
    assert plan is not None
    view = CompiledStructuredDistanceView.from_candidates(
        plan=plan, candidates=((0.0,) * 4,) * 128
    )
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        assert (
            pool.submit(view.distances_to, (1.0,) * 4).result(timeout=30)
            == (1.0,) * 128
        )


def test_rebased_packed_arrays_do_not_retain_history() -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(RealSpace(0.0, 1.0), length=4))
    assert plan is not None
    view = CompiledStructuredDistanceView.from_candidates(
        plan=plan, candidates=((0.0,) * 4,) * 128
    )
    assert view._packed_references is not None
    original_array = ref(view._packed_references._batch.real_values)
    for _ in range(16):
        view = view.rebase(
            candidates=view.candidates, invalidated_indices=frozenset({0})
        )
    gc.collect()
    assert original_array() is None
    assert view._packed_references is not None
    assert view._packed_references._batch.real_values.nbytes == 128 * 4 * 8
    assert view.distances_to((1.0,) * 4) == (1.0,) * 128


@pytest.mark.parametrize("dimension", [1, 17, 8193])
def test_packed_rounding_and_numpy_error_policy(dimension: int) -> None:
    plan = compile_builtin_geometry_plan(
        ArraySpace(RealSpace(0.0, 1.0), length=dimension)
    )
    assert plan is not None
    left = plan.encode((1.0,) + (1e-8,) * (dimension - 1))
    right = plan.encode((0.0,) * dimension)
    packed = plan.pack_references((right,) * 257)
    assert packed is not None
    with np.errstate(all="raise"):
        previous = np.geterr()
        assert plan.squared_distances_to_packed(left, packed) == (1.0,) * 257
        tiny = plan.encode((1e-200,) * dimension)
        assert plan.squared_distances_to_packed(tiny, packed) == (0.0,) * 257
        assert np.geterr() == previous


def test_packed_nested_subtotals_are_not_flattened() -> None:
    plan = compile_builtin_geometry_plan(
        TupleSpace(
            RealSpace(0.0, 1.0), TupleSpace(RealSpace(0.0, 1.0), RealSpace(0.0, 1.0))
        )
    )
    assert plan is not None
    left = plan.encode((1.0, (1e-8, 1e-8)))
    right = plan.encode((0.0, (0.0, 0.0)))
    packed = plan.pack_references((right,) * 128)
    assert packed is not None
    expected = float.fromhex("0x1.0000000000001p+0")
    assert plan.squared_distances_to_packed(left, packed) == (expected,) * 128


@pytest.mark.parametrize("low,high", [(-(2**63), 2**63 - 1), (10**30, 10**30 + 100)])
def test_packing_keeps_unsafe_integer_ranges_on_scalar_path(
    low: int, high: int
) -> None:
    plan = compile_builtin_geometry_plan(ArraySpace(IntegerSpace(low, high), length=4))
    assert plan is not None
    left = plan.encode((low,) * 4)
    right = plan.encode((high,) * 4)
    assert plan.pack_references((right,) * 128) is None
    assert plan.squared_distances_to_many(left, (right,) * 128) == (4.0,) * 128


def test_packing_declines_overflowed_real_spans() -> None:
    plan = compile_builtin_geometry_plan(RealSpace(-1e308, 1e308))
    assert plan is not None
    encoding = plan.encode(-1e308)
    assert plan.pack_references((encoding,) * 128) is None


def test_packing_declines_degenerate_integer_plans() -> None:
    for space in (IntegerSpace(2**53, 2**53 + 1, scale="log"), IntegerSpace(5, 5)):
        plan = compile_builtin_geometry_plan(space)
        assert plan is not None
        encoding = plan.encode(space.low)
        assert plan.pack_references((encoding,) * 128) is None
