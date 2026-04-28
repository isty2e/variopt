"""Shared randomness utilities for variopt."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .json_types import JSONDict, JSONValue

RandomSeed: TypeAlias = int | None
ResultT = TypeVar("ResultT")


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

        if len(self.key_bytes) == 0:
            msg = "key_bytes must not be empty"
            raise ValueError(msg)

        if len(self.key_bytes) % np.dtype(np.uint32).itemsize != 0:
            msg = "key_bytes must encode a whole number of uint32 keys"
            raise ValueError(msg)

        if self.position < 0:
            msg = "position must be non-negative"
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
        random_state.set_state(
            (
                self.algorithm,
                np.frombuffer(self.key_bytes, dtype=np.uint32),
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
        algorithm = data.get("algorithm")
        key_hex = data.get("key_hex")
        position = data.get("position")
        has_gaussian = data.get("has_gaussian")
        cached_gaussian = data.get("cached_gaussian")

        if not isinstance(algorithm, str):
            msg = "random-state snapshot requires string algorithm"
            raise TypeError(msg)

        if not isinstance(key_hex, str):
            msg = "random-state snapshot requires string key_hex"
            raise TypeError(msg)

        if not isinstance(position, int):
            msg = "random-state snapshot requires integer position"
            raise TypeError(msg)

        if not isinstance(has_gaussian, int):
            msg = "random-state snapshot requires integer has_gaussian"
            raise TypeError(msg)

        if not isinstance(cached_gaussian, (int, float)):
            msg = "random-state snapshot requires numeric cached_gaussian"
            raise TypeError(msg)

        return cls(
            algorithm=algorithm,
            key_bytes=bytes.fromhex(key_hex),
            position=position,
            has_gaussian=has_gaussian,
            cached_gaussian=float(cached_gaussian),
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
