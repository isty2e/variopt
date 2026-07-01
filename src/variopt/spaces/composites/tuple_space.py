"""Tuple-shaped composite search space."""

from collections.abc import Mapping, Sequence

import numpy as np
from typing_extensions import override

from ..structured import LeafPath, StructuredLeafSpace, StructuredSearchSpace
from ..types import SpaceBoundaryValue, SpaceCandidateValue
from .adapters import (
    CompositeChildSpace,
    child_candidates_equal,
    group_child_replacements,
    leaf_value_at_child_space,
    normalize_child_space,
    replace_leaf_values_in_child_space,
    sample_child_space,
    validate_child_space,
)


class TupleSpace(
    StructuredSearchSpace[
        Sequence[SpaceBoundaryValue],
        tuple[SpaceCandidateValue, ...],
    ]
):
    """Fixed-arity heterogeneous tuple search space.

    Parameters
    ----------
    *spaces : CompositeChildSpace
        Child spaces in positional tuple order.
    """

    _spaces: tuple[CompositeChildSpace, ...]

    def __init__(self, *spaces: CompositeChildSpace) -> None:
        """Create a tuple search space from positional child spaces.

        Parameters
        ----------
        *spaces : CompositeChildSpace
            Child spaces in positional tuple order.

        Raises
        ------
        ValueError
            If no child spaces are supplied.
        """
        if len(spaces) == 0:
            msg = "TupleSpace requires at least one child space"
            raise ValueError(msg)

        self._spaces = tuple(spaces)

    @property
    def child_spaces(self) -> tuple[CompositeChildSpace, ...]:
        """Return the declared child spaces.

        Returns
        -------
        tuple[CompositeChildSpace, ...]
            Child spaces in positional tuple order.
        """
        return self._spaces

    @override
    def normalize(
        self,
        raw_candidate: Sequence[SpaceBoundaryValue],
    ) -> tuple[SpaceCandidateValue, ...]:
        """Normalize a tuple-shaped boundary candidate.

        Parameters
        ----------
        raw_candidate : Sequence[SpaceBoundaryValue]
            Boundary-level candidate expected to be a non-string sequence.

        Returns
        -------
        tuple[SpaceCandidateValue, ...]
            Canonical tuple candidate.
        """
        if type(raw_candidate) in {bytes, bytearray, str}:
            msg = "tuple candidate must be a non-string sequence"
            raise TypeError(msg)

        if len(raw_candidate) != len(self._spaces):
            msg = "tuple candidate length does not match the declared arity"
            raise ValueError(msg)

        return tuple(
            normalize_child_space(space, raw_candidate[index])
            for index, space in enumerate(self._spaces)
        )

    @override
    def validate(self, candidate: tuple[SpaceCandidateValue, ...]) -> None:
        """Validate a canonical tuple candidate.

        Parameters
        ----------
        candidate : tuple[SpaceCandidateValue, ...]
            Candidate expected to be a canonical tuple whose elements align
            with the declared child spaces.
        """
        if type(candidate) is not tuple:
            msg = "tuple candidate must be canonical tuple"
            raise TypeError(msg)

        if len(candidate) != len(self._spaces):
            msg = "tuple candidate length does not match the declared arity"
            raise ValueError(msg)

        for value, space in zip(candidate, self._spaces, strict=True):
            validate_child_space(space, value)

    @override
    def candidates_equal(
        self,
        left_candidate: tuple[SpaceCandidateValue, ...],
        right_candidate: tuple[SpaceCandidateValue, ...],
    ) -> bool:
        """Return whether two tuples denote the same space point.

        Parameters
        ----------
        left_candidate : tuple[SpaceCandidateValue, ...]
            Left canonical tuple candidate.
        right_candidate : tuple[SpaceCandidateValue, ...]
            Right canonical tuple candidate.

        Returns
        -------
        bool
            Whether every aligned child matches under its child-space equality
            contract.
        """
        self.validate(left_candidate)
        self.validate(right_candidate)
        return all(
            child_candidates_equal(space, left_value, right_value)
            for space, left_value, right_value in zip(
                self._spaces,
                left_candidate,
                right_candidate,
                strict=True,
            )
        )

    @override
    def sample(self, random_state: np.random.RandomState) -> tuple[SpaceCandidateValue, ...]:
        """Sample a canonical tuple candidate.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state object that owns all stochasticity for the sample.

        Returns
        -------
        tuple[SpaceCandidateValue, ...]
            Canonical sampled tuple candidate.
        """
        return tuple(sample_child_space(space, random_state) for space in self._spaces)

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return editable tuple leaf paths.

        Returns
        -------
        tuple[LeafPath, ...]
            Canonical leaf paths prefixed by tuple index.
        """
        paths: list[LeafPath] = []
        for index, child_space in enumerate(self._spaces):
            for child_path in child_space.leaf_paths():
                paths.append((index,) + child_path)
        return tuple(paths)

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the leaf space at a tuple path.

        Parameters
        ----------
        path : LeafPath
            Leaf path whose first segment identifies the tuple position.

        Returns
        -------
        StructuredLeafSpace
            Leaf space declared for ``path``.
        """
        if len(path) == 0:
            msg = "tuple paths must include at least one segment"
            raise TypeError(msg)

        segment = path[0]
        if type(segment) is not int:
            msg = f"path {path!r} is invalid for tuple child traversal"
            raise TypeError(msg)

        if segment < 0 or segment >= len(self._spaces):
            msg = f"path {path!r} references an out-of-bounds tuple index"
            raise TypeError(msg)
        return self._spaces[segment].leaf_space_at_path(path[1:])

    @override
    def leaf_value_at_path(
        self,
        candidate: tuple[SpaceCandidateValue, ...],
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the leaf value stored at a tuple path.

        Parameters
        ----------
        candidate : tuple[SpaceCandidateValue, ...]
            Canonical tuple candidate.
        path : LeafPath
            Leaf path whose first segment identifies the tuple position.

        Returns
        -------
        SpaceCandidateValue
            Canonical leaf value stored at ``path``.
        """
        self.validate(candidate)
        if len(path) == 0:
            msg = "tuple paths must include at least one segment"
            raise TypeError(msg)

        segment = path[0]
        if type(segment) is not int:
            msg = f"path {path!r} is invalid for tuple candidate traversal"
            raise TypeError(msg)

        if segment < 0 or segment >= len(self._spaces):
            msg = f"path {path!r} references an out-of-bounds tuple index"
            raise TypeError(msg)
        return leaf_value_at_child_space(
            self._spaces[segment],
            candidate[segment],
            path[1:],
        )

    @override
    def replace_leaf_values(
        self,
        candidate: tuple[SpaceCandidateValue, ...],
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> tuple[SpaceCandidateValue, ...]:
        """Return a tuple candidate with selected leaves replaced.

        Parameters
        ----------
        candidate : tuple[SpaceCandidateValue, ...]
            Canonical tuple candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by tuple-prefixed leaf paths.

        Returns
        -------
        tuple[SpaceCandidateValue, ...]
            Updated canonical tuple candidate.
        """
        self.validate(candidate)
        grouped_replacements = group_child_replacements(replacements)
        if len(grouped_replacements) == 0:
            return candidate

        for segment in grouped_replacements:
            if type(segment) is not int:
                msg = f"replacement path references an invalid tuple index: {segment!r}"
                raise TypeError(msg)
            if segment < 0 or segment >= len(self._spaces):
                msg = f"replacement path references an out-of-bounds tuple index: {segment!r}"
                raise TypeError(msg)

        replaced_children = list(candidate)
        for index, child_space in enumerate(self._spaces):
            child_replacements = grouped_replacements.get(index)
            if child_replacements is None:
                continue
            replaced_children[index] = replace_leaf_values_in_child_space(
                child_space,
                candidate[index],
                child_replacements,
            )
        return tuple(replaced_children)
