"""Array-shaped composite search space."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from typing_extensions import override

from ..base import SearchSpace
from ..structured import LeafPath, StructuredLeafSpace, StructuredSearchSpace
from ..types import SpaceCandidateValue
from .adapters import group_child_replacements

ElementInputT = TypeVar("ElementInputT")
ElementCandidateT = TypeVar("ElementCandidateT", bound=SpaceCandidateValue)


def require_structured_element_space(
    space: SearchSpace[ElementInputT, ElementCandidateT],
    *,
    operation: str,
) -> StructuredSearchSpace[ElementInputT, ElementCandidateT]:
    """Return the structured element-space view or raise a domain error.

    Parameters
    ----------
    space : SearchSpace[ElementInputT, ElementCandidateT]
        Element space to validate.
    operation : str
        Human-readable operation name used in the error message.

    Returns
    -------
    StructuredSearchSpace[ElementInputT, ElementCandidateT]
        Structured element-space view.

    Raises
    ------
    TypeError
        If ``space`` does not implement ``StructuredSearchSpace``.
    """
    if not isinstance(space, StructuredSearchSpace):
        msg = f"ArraySpace {operation} requires a structured element space"
        raise TypeError(msg)
    return space


@dataclass(frozen=True)
class ArraySpace(
    StructuredSearchSpace[Sequence[ElementInputT], tuple[ElementCandidateT, ...]],
    Generic[ElementInputT, ElementCandidateT],
):
    """Fixed-length homogeneous array search space.

    Parameters
    ----------
    element_space : SearchSpace[ElementInputT, ElementCandidateT]
        Search space shared by every array position.
    length : int
        Fixed array length.
    """

    element_space: SearchSpace[ElementInputT, ElementCandidateT]
    length: int

    def __post_init__(self) -> None:
        """Validate array-space metadata.

        Raises
        ------
        TypeError
            If ``length`` is not a canonical integer.
        ValueError
            If ``length`` is negative.
        """
        if type(self.length) is not int:
            msg = "ArraySpace length must be a canonical integer"
            raise TypeError(msg)

        if self.length < 0:
            msg = "ArraySpace length must be non-negative"
            raise ValueError(msg)

    @override
    def normalize(
        self,
        raw_candidate: Sequence[ElementInputT],
    ) -> tuple[ElementCandidateT, ...]:
        """Normalize an array-shaped boundary candidate.

        Parameters
        ----------
        raw_candidate : Sequence[ElementInputT]
            Boundary-level candidate expected to be a non-string sequence.

        Returns
        -------
        tuple[ElementCandidateT, ...]
            Canonical array candidate.
        """
        if type(raw_candidate) in {bytes, bytearray, str}:
            msg = "array candidate must be a non-string sequence"
            raise TypeError(msg)

        if len(raw_candidate) != self.length:
            msg = "array candidate length does not match the declared length"
            raise ValueError(msg)

        return tuple(self.element_space.normalize(value) for value in raw_candidate)

    @override
    def validate(self, candidate: tuple[ElementCandidateT, ...]) -> None:
        """Validate a canonical array candidate.

        Parameters
        ----------
        candidate : tuple[ElementCandidateT, ...]
            Candidate expected to be a canonical tuple aligned with
            ``element_space`` and ``length``.
        """
        if type(candidate) is not tuple:
            msg = "array candidate must be canonical tuple"
            raise TypeError(msg)

        if len(candidate) != self.length:
            msg = "array candidate length does not match the declared length"
            raise ValueError(msg)

        for value in candidate:
            self.element_space.validate(value)

    @override
    def candidates_equal(
        self,
        left_candidate: tuple[ElementCandidateT, ...],
        right_candidate: tuple[ElementCandidateT, ...],
    ) -> bool:
        """Return whether two arrays denote the same space point.

        Parameters
        ----------
        left_candidate : tuple[ElementCandidateT, ...]
            Left canonical array candidate.
        right_candidate : tuple[ElementCandidateT, ...]
            Right canonical array candidate.

        Returns
        -------
        bool
            Whether every aligned element matches under ``element_space``.
        """
        self.validate(left_candidate)
        self.validate(right_candidate)
        return all(
            self.element_space.candidates_equal(left_value, right_value)
            for left_value, right_value in zip(
                left_candidate,
                right_candidate,
                strict=True,
            )
        )

    @override
    def sample(self, random_state: np.random.RandomState) -> tuple[ElementCandidateT, ...]:
        """Sample a canonical array candidate.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state object that owns all stochasticity for the sample.

        Returns
        -------
        tuple[ElementCandidateT, ...]
            Canonical sampled array candidate.
        """
        return tuple(self.element_space.sample(random_state) for _ in range(self.length))

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return editable array leaf paths.

        Returns
        -------
        tuple[LeafPath, ...]
            Canonical leaf paths prefixed by array index.
        """
        structured_element_space = require_structured_element_space(
            self.element_space,
            operation="leaf traversal",
        )

        paths: list[LeafPath] = []
        for index in range(self.length):
            for child_path in structured_element_space.leaf_paths():
                paths.append((index,) + child_path)
        return tuple(paths)

    @override
    def active_leaf_paths_for_validated_candidate(
        self,
        candidate: tuple[ElementCandidateT, ...],
    ) -> tuple[LeafPath, ...]:
        """Return all array leaves for an already validated candidate.

        Parameters
        ----------
        candidate : tuple[ElementCandidateT, ...]
            Canonical array candidate already validated by the current operation.

        Returns
        -------
        tuple[LeafPath, ...]
            Canonical leaf paths prefixed by array index.
        """
        return self.leaf_paths()

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the leaf space at an array path.

        Parameters
        ----------
        path : LeafPath
            Leaf path whose first segment identifies the array index.

        Returns
        -------
        StructuredLeafSpace
            Leaf space declared for ``path``.
        """
        structured_element_space = require_structured_element_space(
            self.element_space,
            operation="leaf traversal",
        )
        if len(path) == 0:
            msg = "array paths must include at least one segment"
            raise TypeError(msg)

        segment = path[0]
        if type(segment) is not int:
            msg = f"path {path!r} is invalid for array child traversal"
            raise TypeError(msg)

        if segment < 0 or segment >= self.length:
            msg = f"path {path!r} references an out-of-bounds array index"
            raise TypeError(msg)
        return structured_element_space.leaf_space_at_path(path[1:])

    @override
    def leaf_value_at_path(
        self,
        candidate: tuple[ElementCandidateT, ...],
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the leaf value stored at an array path.

        Parameters
        ----------
        candidate : tuple[ElementCandidateT, ...]
            Canonical array candidate.
        path : LeafPath
            Leaf path whose first segment identifies the array index.

        Returns
        -------
        SpaceCandidateValue
            Canonical leaf value stored at ``path``.
        """
        self.validate(candidate)
        return self.leaf_value_at_validated_path(candidate, path)

    @override
    def leaf_value_at_validated_path(
        self,
        candidate: tuple[ElementCandidateT, ...],
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return one array leaf for an already validated candidate.

        Parameters
        ----------
        candidate : tuple[ElementCandidateT, ...]
            Canonical array candidate already validated by the current operation.
        path : LeafPath
            Leaf path whose first segment identifies the array index.

        Returns
        -------
        SpaceCandidateValue
            Canonical leaf value stored at ``path``.
        """
        structured_element_space = require_structured_element_space(
            self.element_space,
            operation="leaf traversal",
        )
        if len(path) == 0:
            msg = "array paths must include at least one segment"
            raise TypeError(msg)

        segment = path[0]
        if type(segment) is not int:
            msg = f"path {path!r} is invalid for array candidate traversal"
            raise TypeError(msg)

        if segment < 0 or segment >= self.length:
            msg = f"path {path!r} references an out-of-bounds array index"
            raise TypeError(msg)
        return structured_element_space.leaf_value_at_validated_path(
            candidate[segment],
            path[1:],
        )

    @override
    def replace_leaf_values(
        self,
        candidate: tuple[ElementCandidateT, ...],
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> tuple[ElementCandidateT, ...]:
        """Return an array candidate with selected leaves replaced.

        Parameters
        ----------
        candidate : tuple[ElementCandidateT, ...]
            Canonical array candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by index-prefixed leaf paths.

        Returns
        -------
        tuple[ElementCandidateT, ...]
            Updated canonical array candidate.
        """
        self.validate(candidate)
        return self.replace_leaf_values_in_validated_candidate(
            candidate,
            replacements,
        )

    @override
    def replace_leaf_values_in_validated_candidate(
        self,
        candidate: tuple[ElementCandidateT, ...],
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> tuple[ElementCandidateT, ...]:
        """Return replacements for an already validated array candidate.

        Parameters
        ----------
        candidate : tuple[ElementCandidateT, ...]
            Canonical array candidate already validated by the current operation.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by index-prefixed leaf paths.

        Returns
        -------
        tuple[ElementCandidateT, ...]
            Updated canonical array candidate.
        """
        structured_element_space = require_structured_element_space(
            self.element_space,
            operation="leaf replacement",
        )

        grouped_replacements = group_child_replacements(replacements)
        if len(grouped_replacements) == 0:
            return candidate

        for segment in grouped_replacements:
            if type(segment) is not int or segment < 0 or segment >= self.length:
                msg = f"replacement path references an invalid array index: {segment!r}"
                raise TypeError(msg)

        replaced_children = list(candidate)
        for index, child_candidate in enumerate(candidate):
            child_replacements = grouped_replacements.get(index)
            if child_replacements is None:
                continue
            replaced_children[index] = (
                structured_element_space.replace_leaf_values_in_validated_candidate(
                    child_candidate,
                    child_replacements,
                )
            )
        return tuple(replaced_children)
