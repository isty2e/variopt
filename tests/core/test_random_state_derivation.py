"""Exact stream and ownership contracts for batched child derivation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from hashlib import sha256
from json import dumps, loads

import numpy as np
import pytest

from variopt.randomness import (
    RandomStateSnapshot,
    derive_random_state_snapshot,
    derive_random_state_snapshots,
)


@pytest.mark.parametrize("count", [0, 1, 2, 16, 257])
@pytest.mark.parametrize("normal_count", [0, 1, 2, 3, 625])
def test_batch_matches_scalar_snapshots_and_draws(
    count: int, normal_count: int
) -> None:
    random_state = np.random.RandomState(17)
    random_state.normal(size=normal_count)
    parent = RandomStateSnapshot.from_random_state(random_state)
    key_groups = tuple((f"p-{index}", "generation") for index in range(count))

    actual = derive_random_state_snapshots(
        parent, namespace="test.child", key_groups=key_groups
    )
    expected = tuple(
        derive_random_state_snapshot(parent, namespace="test.child", keys=keys)
        for keys in key_groups
    )

    assert actual == expected
    assert parent == RandomStateSnapshot.from_random_state(random_state)
    for child, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            child.materialize().normal(size=9), reference.materialize().normal(size=9)
        )


def test_batch_preserves_seed_encoding_fixtures() -> None:
    random_state = np.random.RandomState(17)
    random_state.random_sample(size=6)
    parent = RandomStateSnapshot.from_random_state(random_state)
    parent = replace(
        parent,
        has_gaussian=1,
        cached_gaussian=1.1453112895720903,
        key_bytes=np.frombuffer(parent.key_bytes, dtype=np.uint32)
        .astype("<u4")
        .tobytes(),
    )
    children = derive_random_state_snapshots(
        parent,
        namespace="test.child",
        key_groups=((), ("ab", "c"), ("a", "bc"), ("", "\u03bb\0")),
    )

    # Pin the seed encoding independently of the shared derivation helpers.
    assert tuple(
        sha256(
            np.frombuffer(child.key_bytes, dtype=np.uint32).astype("<u4").tobytes()
        ).hexdigest()
        for child in children
    ) == (
        "0bd06dc971c00d78fb7ae84aff18abbce01732d757cd810c70a991e03ce988e3",
        "d55025992d4cbee56d536005909a999643e3e2f8a5c782f2154920aeac5fdf97",
        "6c88aa0df688cbc1504745f9dc03ed3f41cbf7790d6fd477ad00abdeb10830b8",
        "42f6d9e637fdec47421f7df55f84112e22af81880347a6d783c6d501f24ee132",
    )
    assert all(child.position == 624 for child in children)
    assert all(child.has_gaussian == 0 for child in children)
    assert all(child.cached_gaussian == 0.0 for child in children)


def test_batch_order_duplicates_and_chunking_do_not_change_streams() -> None:
    parent = RandomStateSnapshot.from_seed(7)
    keys = (("ab", "c"), ("a", "bc"), (), ("ab", "c"), ("\0", "\u03bb"))
    derive = partial(derive_random_state_snapshots, parent, namespace="test.child")

    children = derive(key_groups=keys)

    assert children == tuple(reversed(derive(key_groups=tuple(reversed(keys)))))
    assert children == derive(key_groups=keys[:2]) + derive(key_groups=keys[2:])
    assert children[0] == children[3]
    assert children[0] != children[1]
    assert len(set(children)) == 4
    assert children != derive_random_state_snapshots(
        parent, namespace="test.other", key_groups=keys
    )


@pytest.mark.parametrize("count", [0, 1, 16])
def test_batch_rejects_empty_namespace(count: int) -> None:
    with pytest.raises(ValueError, match="namespace must not be empty"):
        derive_random_state_snapshots(
            RandomStateSnapshot.from_seed(7), namespace="", key_groups=(("p",),) * count
        )


def test_empty_batch_does_not_construct_a_random_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = RandomStateSnapshot.from_seed(7)

    def fail_if_constructed(seed: int) -> np.random.RandomState:
        raise AssertionError("empty derivation must not construct an RNG")

    monkeypatch.setattr(np.random, "RandomState", fail_if_constructed)

    assert (
        derive_random_state_snapshots(parent, namespace="test.child", key_groups=())
        == ()
    )


@pytest.mark.parametrize("count", [1, 2, 16])
def test_batch_constructs_only_one_local_generator(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    parent = RandomStateSnapshot.from_seed(7)
    constructor = np.random.RandomState
    construction_seeds: list[int] = []
    groups = tuple((str(index),) for index in range(count))
    expected = tuple(
        derive_random_state_snapshot(parent, namespace="test.child", keys=keys)
        for keys in groups
    )

    def counted_constructor(seed: int) -> np.random.RandomState:
        construction_seeds.append(seed)
        return constructor(seed)

    monkeypatch.setattr(np.random, "RandomState", counted_constructor)

    actual = derive_random_state_snapshots(
        parent, namespace="test.child", key_groups=groups
    )

    assert actual == expected
    assert len(construction_seeds) == 1


def test_batch_children_are_independent_after_materialization_and_json() -> None:
    parent = RandomStateSnapshot.from_seed(11)
    children = derive_random_state_snapshots(
        parent, namespace="test.child", key_groups=(("p",), ("q",), ("p",))
    )
    before = tuple(child.to_dict() for child in children)
    fork = children[0].materialize()
    fork.normal(size=3)
    fork.seed(99)
    fork.uniform(size=1024)

    assert tuple(child.to_dict() for child in children) == before
    assert (
        tuple(RandomStateSnapshot.from_dict(loads(dumps(data))) for data in before)
        == children
    )
    assert children[0] == children[2]


def test_batch_does_not_touch_global_random_state() -> None:
    before = np.random.get_state(legacy=True)
    derive_random_state_snapshots(
        RandomStateSnapshot.from_seed(17),
        namespace="test.child",
        key_groups=tuple((str(index),) for index in range(40)),
    )
    after = np.random.get_state(legacy=True)

    assert isinstance(before, tuple)
    assert isinstance(after, tuple)

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_batch_concurrent_calls_own_their_hash_and_rng() -> None:
    parent = RandomStateSnapshot.from_seed(17)
    derive = partial(derive_random_state_snapshots, namespace="test.child")
    parents = tuple(replace(parent, position=index) for index in range(24))
    key_groups = tuple((f"p-{index}",) for index in range(32))
    operation = partial(derive, key_groups=key_groups)
    expected = tuple(operation(snapshot) for snapshot in parents)

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = tuple(executor.map(operation, parents))

    assert actual == expected
    assert all(len(set(children)) == len(key_groups) for children in actual)


def test_failed_batch_does_not_contaminate_retry() -> None:
    parent = RandomStateSnapshot.from_seed(7)
    derive = partial(derive_random_state_snapshots, parent, namespace="test.child")
    expected = derive(key_groups=(("p",), ("q",)))

    with pytest.raises(UnicodeEncodeError):
        derive(key_groups=(("p",), ("\ud800",), ("q",)))

    assert derive(key_groups=(("p",), ("q",))) == expected


def test_batch_hashes_all_parent_metadata_including_signed_zero() -> None:
    parent = RandomStateSnapshot.from_seed(7)
    parents = (
        parent,
        replace(parent, position=0),
        replace(parent, cached_gaussian=-0.0),
        replace(parent, has_gaussian=1),
        replace(parent, cached_gaussian=2.0),
    )
    children = tuple(
        derive_random_state_snapshots(
            snapshot, namespace="test.child", key_groups=(("p",),)
        )[0]
        for snapshot in parents
    )

    assert len(set(children)) == len(parents)
    assert children == tuple(
        derive_random_state_snapshot(snapshot, namespace="test.child", keys=("p",))
        for snapshot in parents
    )
