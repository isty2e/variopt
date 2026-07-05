"""Shared randomness utilities for variopt."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from math import isfinite
from typing import Protocol, TypeAlias, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .json_types import (
    JSONDict,
    JSONValue,
    require_json_finite_float,
    require_json_int,
    require_json_str,
)

RandomSeed: TypeAlias = int | None
ResultT = TypeVar("ResultT")
_RANDOM_STATE_ALGORITHM = "MT19937"
_MT19937_KEY_COUNT = 624
_UINT32_BYTE_COUNT = np.dtype(np.uint32).itemsize


class TypedRandomState(Protocol):
    """Typed subset of the NumPy ``RandomState`` API used by variopt.

    Notes
    -----
    This protocol exists to recover concrete return types from NumPy methods
    whose stub surface is still broad enough to force casts at call sites.
    """

    def uniform(self, low: float, high: float) -> float:
        """Draw one floating-point sample from a uniform interval.

        Parameters
        ----------
        low : float
            Inclusive lower bound of the interval.
        high : float
            Exclusive upper bound of the interval.

        Returns
        -------
        float
            Sampled floating-point value.
        """
        ...

    @overload
    def randint(self, low: int, high: int | None = None) -> int:
        """Draw one scalar integer sample.

        Parameters
        ----------
        low : int
            Inclusive lower bound when ``high`` is provided, or exclusive upper
            bound when ``high`` is ``None``.
        high : int | None, default=None
            Exclusive upper bound when provided.

        Returns
        -------
        int
            Sampled integer value.
        """
        ...

    @overload
    def randint(
        self,
        low: int,
        high: int | None,
        size: int,
    ) -> npt.NDArray[np.int_]:
        """Draw a vector of integer samples.

        Parameters
        ----------
        low : int
            Inclusive lower bound when ``high`` is provided, or exclusive upper
            bound when ``high`` is ``None``.
        high : int | None
            Exclusive upper bound when provided.
        size : int
            Number of integers to sample.

        Returns
        -------
        numpy.typing.NDArray[numpy.int_]
            Sampled integer vector.
        """
        ...

    @overload
    def choice(
        self,
        a: int,
        size: int,
        replace: bool = True,
        p: Sequence[float] | None = None,
    ) -> npt.NDArray[np.int_]:
        """Draw a vector of integer choice indices.

        Parameters
        ----------
        a : int
            Population size from which indices are drawn.
        size : int
            Number of indices to sample.
        replace : bool, default=True
            Whether the sample may contain repeated indices.
        p : Sequence[float] | None, default=None
            Optional probability weights.

        Returns
        -------
        numpy.typing.NDArray[numpy.int_]
            Sampled index vector.
        """
        ...

    @overload
    def choice(
        self,
        a: int,
        size: None = None,
        replace: bool = True,
        p: Sequence[float] | None = None,
    ) -> int:
        """Draw one integer choice index.

        Parameters
        ----------
        a : int
            Population size from which the index is drawn.
        size : None, default=None
            Scalar form marker for the overload.
        replace : bool, default=True
            Whether sampling with replacement is allowed.
        p : Sequence[float] | None, default=None
            Optional probability weights.

        Returns
        -------
        int
            Sampled index.
        """
        ...

    def permutation(self, x: int) -> npt.NDArray[np.int_]:
        """Return a random permutation of ``range(x)``.

        Parameters
        ----------
        x : int
            Permutation size.

        Returns
        -------
        numpy.typing.NDArray[numpy.int_]
            Permuted integer indices.
        """
        ...


@dataclass(frozen=True, slots=True)
class RandomStateSnapshot:
    """Immutable snapshot of a NumPy ``RandomState``.

    Parameters
    ----------
    algorithm : str
        NumPy random-state algorithm name.
    key_bytes : bytes
        Serialized uint32 key buffer returned by ``RandomState.get_state``.
    position : int
        Internal generator position inside the key stream.
    has_gaussian : int
        Flag indicating whether a cached Gaussian sample is stored.
    cached_gaussian : float
        Cached Gaussian sample carried by the NumPy random-state state tuple.
    """

    algorithm: str
    key_bytes: bytes
    position: int
    has_gaussian: int
    cached_gaussian: float

    def __post_init__(self) -> None:
        """Validate snapshot fields.

        Raises
        ------
        ValueError
            Raised when the serialized random-state payload is inconsistent.
        """
        if self.algorithm == "":
            msg = "algorithm must not be empty"
            raise ValueError(msg)

        if self.algorithm != _RANDOM_STATE_ALGORITHM:
            msg = "algorithm must be MT19937"
            raise ValueError(msg)

        if type(self.position) is not int:
            msg = "position must be an integer"
            raise TypeError(msg)

        if type(self.has_gaussian) is not int:
            msg = "has_gaussian must be an integer"
            raise TypeError(msg)

        if type(self.cached_gaussian) not in {int, float}:
            msg = "cached_gaussian must be numeric"
            raise TypeError(msg)
        cached_gaussian = float(self.cached_gaussian)
        if not isfinite(cached_gaussian):
            msg = "cached_gaussian must be finite"
            raise ValueError(msg)
        object.__setattr__(self, "cached_gaussian", cached_gaussian)

        if len(self.key_bytes) == 0:
            msg = "key_bytes must not be empty"
            raise ValueError(msg)

        if len(self.key_bytes) % _UINT32_BYTE_COUNT != 0:
            msg = "key_bytes must encode a whole number of uint32 keys"
            raise ValueError(msg)

        if len(self.key_bytes) != _MT19937_KEY_COUNT * _UINT32_BYTE_COUNT:
            msg = "key_bytes must encode exactly 624 MT19937 uint32 keys"
            raise ValueError(msg)

        if self.position < 0:
            msg = "position must be non-negative"
            raise ValueError(msg)

        if self.position > _MT19937_KEY_COUNT:
            msg = "position must be at most 624 for MT19937"
            raise ValueError(msg)

        if self.has_gaussian not in {0, 1}:
            msg = "has_gaussian must be 0 or 1"
            raise ValueError(msg)

    @classmethod
    def from_random_state(cls, random_state: np.random.RandomState) -> Self:
        """Capture an immutable snapshot from a NumPy random state.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state instance to snapshot.

        Returns
        -------
        Self
            Immutable snapshot carrying the full NumPy random-state payload.
        """
        algorithm, keys, position, has_gaussian, cached_gaussian = random_state.get_state()
        key_array = np.asarray(keys, dtype=np.uint32)
        return cls(
            algorithm=algorithm,
            key_bytes=key_array.tobytes(),
            position=int(position),
            has_gaussian=int(has_gaussian),
            cached_gaussian=float(cached_gaussian),
        )

    @classmethod
    def from_seed(cls, random_state: RandomSeed = None) -> Self:
        """Build a snapshot from a public seed ingress.

        Parameters
        ----------
        random_state : RandomSeed, optional
            Integer seed or ``None`` for non-deterministic initialization.

        Returns
        -------
        Self
            Immutable snapshot initialized from the requested seed ingress.
        """
        return cls.from_random_state(normalize_random_state(random_state))

    def materialize(self) -> np.random.RandomState:
        """Materialize a local random state from the snapshot.

        Returns
        -------
        numpy.random.RandomState
            New random-state instance initialized from the stored payload.
        """
        random_state = np.random.RandomState()
        key_array = np.array(memoryview(self.key_bytes).cast("I"), dtype=np.uint32)
        random_state.set_state(
            (
                self.algorithm,
                key_array,
                self.position,
                self.has_gaussian,
                self.cached_gaussian,
            ),
        )
        return random_state

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the random-state snapshot.

        Returns
        -------
        JSONDict
            JSON-safe mapping that preserves the full NumPy random-state
            payload.
        """
        return {
            "algorithm": self.algorithm,
            "key_hex": self.key_bytes.hex(),
            "position": self.position,
            "has_gaussian": self.has_gaussian,
            "cached_gaussian": self.cached_gaussian,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, JSONValue]) -> Self:
        """Build a snapshot from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe snapshot mapping produced by :meth:`to_dict`.

        Returns
        -------
        Self
            Reconstructed random-state snapshot.

        Raises
        ------
        TypeError
            If the supplied mapping carries invalid field types.
        """
        algorithm = require_json_str(data.get("algorithm"), field_name="algorithm")
        key_hex = require_json_str(data.get("key_hex"), field_name="key_hex")
        position = require_json_int(data.get("position"), field_name="position")
        has_gaussian = require_json_int(
            data.get("has_gaussian"),
            field_name="has_gaussian",
        )
        cached_gaussian = require_json_finite_float(
            data.get("cached_gaussian"),
            field_name="cached_gaussian",
        )
        try:
            key_bytes = bytes.fromhex(key_hex)
        except ValueError as exc:
            msg = "key_hex must encode hexadecimal bytes"
            raise ValueError(msg) from exc

        return cls(
            algorithm=algorithm,
            key_bytes=key_bytes,
            position=position,
            has_gaussian=has_gaussian,
            cached_gaussian=cached_gaussian,
        )

    def advance(
        self,
        operation: Callable[[np.random.RandomState], ResultT],
    ) -> tuple[ResultT, Self]:
        """Run one RNG-consuming operation and capture the advanced snapshot.

        Parameters
        ----------
        operation : Callable[[numpy.random.RandomState], ResultT]
            Callback that consumes randomness from a materialized local random
            state.

        Returns
        -------
        tuple[ResultT, Self]
            Callback result together with the advanced immutable snapshot.
        """
        random_state = self.materialize()
        result = operation(random_state)
        return result, type(self).from_random_state(random_state)

    def spawn_seeds(
        self,
        count: int,
    ) -> tuple[tuple[int, ...], Self]:
        """Draw integer child seeds and return the advanced snapshot.

        Parameters
        ----------
        count : int
            Number of integer seeds to draw.

        Returns
        -------
        tuple[tuple[int, ...], Self]
            Drawn integer seeds together with the advanced immutable snapshot.

        Raises
        ------
        ValueError
            Raised when ``count`` is negative.
        """
        if count < 0:
            msg = "count must be non-negative"
            raise ValueError(msg)

        if count == 0:
            return (), self

        random_state = self.materialize()
        seed_array = random_state_randints(
            random_state,
            low=0,
            high=int(np.iinfo(np.int32).max),
            size=count,
        )
        next_snapshot = type(self).from_random_state(random_state)
        return (
            tuple(
                int(cast(np.int_, seed_array[index]))
                for index in range(int(seed_array.size))
            ),
            next_snapshot,
        )


def derive_random_state_snapshot(
    snapshot: RandomStateSnapshot,
    *,
    namespace: str,
    keys: Sequence[str],
) -> RandomStateSnapshot:
    """Derive a deterministic child random-state snapshot from stable keys.

    Parameters
    ----------
    snapshot : RandomStateSnapshot
        Parent random-state snapshot that anchors the derived stream.
    namespace : str
        Domain separator for the child stream family.
    keys : Sequence[str]
        Stable key components that identify the child stream.

    Returns
    -------
    RandomStateSnapshot
        Child snapshot initialized from a deterministic seed.

    Raises
    ------
    ValueError
        Raised when ``namespace`` is empty.
    """
    if namespace == "":
        msg = "namespace must not be empty"
        raise ValueError(msg)

    hasher = blake2b(digest_size=8)

    def update_text(value: str) -> None:
        encoded_value = value.encode("utf-8")
        hasher.update(len(encoded_value).to_bytes(8, "big"))
        hasher.update(encoded_value)

    update_text(namespace)
    update_text(snapshot.algorithm)
    hasher.update(len(snapshot.key_bytes).to_bytes(8, "big"))
    hasher.update(snapshot.key_bytes)
    hasher.update(snapshot.position.to_bytes(8, "big"))
    hasher.update(snapshot.has_gaussian.to_bytes(1, "big"))
    update_text(repr(snapshot.cached_gaussian))
    for key in keys:
        update_text(key)

    seed_limit = int(np.iinfo(np.int32).max)
    seed = int.from_bytes(hasher.digest(), "big") % seed_limit
    return RandomStateSnapshot.from_seed(seed)


@overload
def normalize_random_state(random_state: None = None) -> np.random.RandomState:
    """Normalize ``None`` into a fresh NumPy random state.

    Parameters
    ----------
    random_state : None, default=None
        Marker requesting non-deterministic initialization.

    Returns
    -------
    numpy.random.RandomState
        Fresh local random-state instance.
    """
    ...


@overload
def normalize_random_state(random_state: int) -> np.random.RandomState:
    """Normalize an integer seed into a deterministic NumPy random state.

    Parameters
    ----------
    random_state : int
        Deterministic integer seed.

    Returns
    -------
    numpy.random.RandomState
        Deterministically initialized local random-state instance.
    """
    ...


def normalize_random_state(random_state: RandomSeed = None) -> np.random.RandomState:
    """Return a canonical NumPy ``RandomState`` from a public seed input.

    Parameters
    ----------
    random_state : RandomSeed, optional
        Public v1 randomness ingress. ``None`` requests a fresh non-deterministic
        generator. An ``int`` requests deterministic construction of a local
        ``RandomState`` instance.

    Returns
    -------
    numpy.random.RandomState
        A local RNG instance suitable for internal stochastic execution.

    Raises
    ------
    TypeError
        Raised when ``random_state`` is not ``None`` or an integer seed.
    """

    if random_state is None:
        return np.random.RandomState()

    if type(random_state) is not int:
        msg = "random_state must be an int or None"
        raise TypeError(msg)

    return np.random.RandomState(random_state)


def random_state_randint(
    random_state: np.random.RandomState,
    low: int,
    high: int | None = None,
) -> int:
    """Draw one scalar integer sample from a canonical random state.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        Random-state instance used for sampling.
    low : int
        Inclusive lower bound when ``high`` is provided, or exclusive upper
        bound when ``high`` is ``None``.
    high : int | None, default=None
        Exclusive upper bound when provided.

    Returns
    -------
    int
        Sampled integer.
    """
    typed_state = cast(TypedRandomState, random_state)
    return typed_state.randint(low, high)


def random_state_randints(
    random_state: np.random.RandomState,
    low: int,
    high: int | None,
    size: int,
) -> npt.NDArray[np.int_]:
    """Draw a vector of integer samples from a canonical random state.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        Random-state instance used for sampling.
    low : int
        Inclusive lower bound when ``high`` is provided, or exclusive upper
        bound when ``high`` is ``None``.
    high : int | None
        Exclusive upper bound when provided.
    size : int
        Number of integers to sample.

    Returns
    -------
    numpy.typing.NDArray[numpy.int_]
        Sampled integer vector.
    """
    typed_state = cast(TypedRandomState, random_state)
    return typed_state.randint(low, high, size)


def random_state_choice_index(
    random_state: np.random.RandomState,
    population_size: int,
    weights: Sequence[float] | None = None,
) -> int:
    """Draw one scalar choice index from a population.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        Random-state instance used for sampling.
    population_size : int
        Population size defining the half-open interval ``[0, population_size)``.
    weights : Sequence[float] | None, default=None
        Optional probability weights.

    Returns
    -------
    int
        Sampled population index.
    """
    typed_state = cast(TypedRandomState, random_state)
    if weights is None:
        return typed_state.choice(population_size)

    return typed_state.choice(population_size, p=weights)


def random_state_choice_indices_without_replacement(
    random_state: np.random.RandomState,
    population_size: int,
    count: int,
    weights: Sequence[float] | None = None,
) -> tuple[int, ...]:
    """Draw distinct choice indices without replacement.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        Random-state instance used for sampling.
    population_size : int
        Population size defining the half-open interval ``[0, population_size)``.
    count : int
        Number of distinct indices to sample.
    weights : Sequence[float] | None, default=None
        Optional probability weights.

    Returns
    -------
    tuple[int, ...]
        Distinct sampled indices.
    """
    typed_state = cast(TypedRandomState, random_state)
    selected_indices = typed_state.choice(
        population_size,
        size=count,
        replace=False,
        p=weights,
    )
    selected_index_list = cast(
        list[int],
        np.asarray(selected_indices, dtype=np.int_).tolist(),
    )
    return tuple(selected_index_list)


def random_state_permutation_indices(
    random_state: np.random.RandomState,
    size: int,
) -> tuple[int, ...]:
    """Draw a permutation of integer indices from a canonical random state.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        Random-state instance used for sampling.
    size : int
        Number of indices in the half-open interval ``[0, size)``.

    Returns
    -------
    tuple[int, ...]
        Permuted indices.
    """
    typed_state = cast(TypedRandomState, random_state)
    index_list = cast(
        list[int],
        np.asarray(typed_state.permutation(size), dtype=np.int_).tolist(),
    )
    return tuple(index_list)
