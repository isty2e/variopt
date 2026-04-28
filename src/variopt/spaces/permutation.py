"""Permutation search space."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import override

from .scalar import IntegerSpace
from .structured import LeafPath, StructuredLeafSpace, StructuredSearchSpace
from .types import SpaceCandidateValue


def normalize_permutation_values(
    values: Sequence[SpaceCandidateValue],
    *,
    size: int,
) -> tuple[int, ...]:
    """Validate and canonicalize a permutation candidate.

    Parameters
    ----------
    values : Sequence[SpaceCandidateValue]
        Raw permutation-like value sequence.
    size : int
        Required permutation length and domain size.

    Returns
    -------
    tuple[int, ...]
        Canonical permutation tuple over ``0..size-1``.

    Raises
    ------
    TypeError
        If ``values`` is string-like or contains non-integer entries.
    ValueError
        If the candidate length, domain, or uniqueness constraint is violated.
    """
    if isinstance(values, (bytes, bytearray, str)):
        msg = "permutation candidate must be a non-string sequence"
        raise TypeError(msg)
    if len(values) != size:
        msg = "permutation candidate length does not match the declared size"
        raise ValueError(msg)

    normalized_values: list[int] = []
    seen = [False] * size
    for raw_value in values:
        if type(raw_value) is not int:
            msg = "permutation candidate entries must be canonical integers"
            raise TypeError(msg)
        if raw_value < 0 or raw_value >= size:
            msg = "permutation candidate entry is outside the declared domain"
            raise ValueError(msg)
        if seen[raw_value]:
            msg = "permutation candidate must not contain duplicate entries"
            raise ValueError(msg)
        seen[raw_value] = True
        normalized_values.append(raw_value)

    return tuple(normalized_values)


@dataclass(frozen=True)
class PermutationSpace(
    StructuredSearchSpace[Sequence[int], tuple[int, ...]],
):
    """Fixed-domain permutation search space over ``0..size-1``.

    Parameters
    ----------
    size : int
        Permutation length and domain size. Canonical candidates are tuples
        containing each integer in ``0..size-1`` exactly once.
    """

    size: int
    _index_space: IntegerSpace = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate permutation-space metadata.

        Raises
        ------
        TypeError
            If ``size`` is not a canonical integer.
        ValueError
            If ``size`` is not positive.
        """
        if type(self.size) is not int:
            msg = "PermutationSpace size must be a canonical integer"
            raise TypeError(msg)
        if self.size <= 0:
            msg = "PermutationSpace size must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "_index_space", IntegerSpace(0, self.size - 1))

    @override
    def normalize(self, raw_candidate: Sequence[int]) -> tuple[int, ...]:
        """Normalize a boundary permutation candidate.

        Parameters
        ----------
        raw_candidate : Sequence[int]
            Boundary-level permutation candidate.

        Returns
        -------
        tuple[int, ...]
            Canonical permutation tuple.
        """
        return normalize_permutation_values(raw_candidate, size=self.size)

    @override
    def validate(self, candidate: tuple[int, ...]) -> None:
        """Validate a canonical permutation candidate.

        Parameters
        ----------
        candidate : tuple[int, ...]
            Canonical permutation tuple to validate.
        """
        if type(candidate) is not tuple:
            msg = "permutation candidate must be canonical tuple"
            raise TypeError(msg)
        _ = normalize_permutation_values(candidate, size=self.size)

    @override
    def sample(self, random_state: np.random.RandomState) -> tuple[int, ...]:
        """Sample a canonical permutation.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state object that owns all stochasticity for the sample.

        Returns
        -------
        tuple[int, ...]
            Canonical sampled permutation.
        """
        values = list(range(self.size))
        random_state.shuffle(values)
        return tuple(values)

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return editable permutation leaf paths.

        Returns
        -------
        tuple[LeafPath, ...]
            One singleton index path per permutation position.
        """
        return tuple((index,) for index in range(self.size))

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the leaf space at a permutation position.

        Parameters
        ----------
        path : LeafPath
            Singleton index path identifying a permutation position.

        Returns
        -------
        StructuredLeafSpace
            Integer leaf space for the indexed permutation position.
        """
        if len(path) != 1 or not isinstance(path[0], int):
            msg = f"path {path!r} is invalid for permutation traversal"
            raise TypeError(msg)
        if path[0] < 0 or path[0] >= self.size:
            msg = f"path {path!r} references an out-of-bounds permutation index"
            raise TypeError(msg)
        return self._index_space

    @override
    def leaf_value_at_path(
        self,
        candidate: tuple[int, ...],
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the value stored at a permutation position.

        Parameters
        ----------
        candidate : tuple[int, ...]
            Canonical permutation candidate.
        path : LeafPath
            Singleton index path identifying a permutation position.

        Returns
        -------
        SpaceCandidateValue
            Integer value stored at ``path``.
        """
        self.validate(candidate)
        if len(path) != 1 or not isinstance(path[0], int):
            msg = f"path {path!r} is invalid for permutation candidate traversal"
            raise TypeError(msg)
        if path[0] < 0 or path[0] >= self.size:
            msg = f"path {path!r} references an out-of-bounds permutation index"
            raise TypeError(msg)
        return candidate[path[0]]

    @override
    def replace_leaf_values(
        self,
        candidate: tuple[int, ...],
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> tuple[int, ...]:
        """Return a permutation with selected positions replaced.

        Parameters
        ----------
        candidate : tuple[int, ...]
            Canonical permutation candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by singleton index paths.

        Returns
        -------
        tuple[int, ...]
            Canonical permutation after applying the replacements.
        """
        self.validate(candidate)
        values = list(candidate)
        for path, replacement in replacements.items():
            if len(path) != 1 or not isinstance(path[0], int):
                msg = f"path {path!r} is invalid for permutation replacement"
                raise TypeError(msg)
            if path[0] < 0 or path[0] >= self.size:
                msg = f"path {path!r} references an out-of-bounds permutation index"
                raise TypeError(msg)
            if type(replacement) is not int:
                msg = "permutation leaf replacement must be a canonical integer"
                raise TypeError(msg)
            values[path[0]] = self._index_space.normalize(replacement)
        return self.normalize(values)
