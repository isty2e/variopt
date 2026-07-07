"""Record-shaped composite search space."""

from collections.abc import Mapping

import numpy as np
from typing_extensions import override

from ..structured import LeafPath, StructuredLeafSpace, StructuredSearchSpace
from ..types import SpaceBoundaryValue, SpaceCandidateValue
from .adapters import (
    CompositeChildSpace,
    child_candidates_equal,
    group_child_replacements,
    leaf_value_at_validated_child_space,
    normalize_child_space,
    replace_leaf_values_in_validated_child_space,
    sample_child_space,
    validate_child_space,
)
from .records import RecordCandidate


class RecordSpace(
    StructuredSearchSpace[
        Mapping[str, SpaceBoundaryValue] | RecordCandidate,
        RecordCandidate,
    ]
):
    """Named-field heterogeneous record search space.

    Parameters
    ----------
    **fields : CompositeChildSpace
        Mapping from field name to child space.
    """

    _fields: tuple[tuple[str, CompositeChildSpace], ...]
    _field_names: tuple[str, ...]
    _field_indices: dict[str, int]
    _child_spaces_by_name: dict[str, CompositeChildSpace]

    def __init__(self, **fields: CompositeChildSpace) -> None:
        """Create a record search space from named child spaces.

        Parameters
        ----------
        **fields : CompositeChildSpace
            Mapping from field name to child space.

        Raises
        ------
        ValueError
            If no fields are supplied.
        """
        if len(fields) == 0:
            msg = "RecordSpace requires at least one named field"
            raise ValueError(msg)

        self._fields = tuple(fields.items())
        self._field_names = tuple(fields)
        self._field_indices = {
            name: index
            for index, name in enumerate(self._field_names)
        }
        self._child_spaces_by_name = dict(self._fields)

    @property
    def fields(self) -> tuple[tuple[str, CompositeChildSpace], ...]:
        """Return the declared record fields.

        Returns
        -------
        tuple[tuple[str, CompositeChildSpace], ...]
            Field-name and child-space pairs in canonical order.
        """
        return self._fields

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two record spaces declare the same ordered fields."""
        if type(self) is not RecordSpace:
            return self is other
        if type(other) is not RecordSpace:
            return False
        return self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Return a hash derived from the record-space declaration."""
        if type(self) is not RecordSpace:
            return object.__hash__(self)
        return hash((RecordSpace, self._fields))

    @override
    def normalize(
        self,
        raw_candidate: Mapping[str, SpaceBoundaryValue] | RecordCandidate,
    ) -> RecordCandidate:
        """Normalize a record-shaped boundary candidate.

        Parameters
        ----------
        raw_candidate : Mapping[str, SpaceBoundaryValue] | RecordCandidate
            Boundary-level candidate expected to be a mapping or
            :class:`RecordCandidate`.

        Returns
        -------
        RecordCandidate
            Canonical record candidate.
        """
        if isinstance(raw_candidate, RecordCandidate):
            self.validate(raw_candidate)
            return raw_candidate
        raw_mapping = raw_candidate

        actual_names = tuple(raw_mapping.keys())

        if (
            set(actual_names) != set(self._field_names)
            or len(actual_names) != len(self._field_names)
        ):
            msg = "record candidate keys must exactly match the declared fields"
            raise ValueError(msg)

        entries = tuple(
            (
                name,
                normalize_child_space(space, raw_mapping[name]),
            )
            for name, space in self._fields
        )
        return RecordCandidate(entries=entries)

    @override
    def validate(self, candidate: RecordCandidate) -> None:
        """Validate a canonical record candidate.

        Parameters
        ----------
        candidate : RecordCandidate
            Candidate expected to be a canonical :class:`RecordCandidate`.
        """
        if type(candidate) is not RecordCandidate:
            msg = "record candidate must be canonical RecordCandidate"
            raise TypeError(msg)

        actual_names = tuple(candidate.keys())

        if actual_names != self._field_names:
            msg = "record candidate keys must exactly match the declared fields"
            raise ValueError(msg)

        for index, (_name, space) in enumerate(self._fields):
            validate_child_space(space, candidate.entries[index][1])

    @override
    def candidates_equal(
        self,
        left_candidate: RecordCandidate,
        right_candidate: RecordCandidate,
    ) -> bool:
        """Return whether two records denote the same space point.

        Parameters
        ----------
        left_candidate : RecordCandidate
            Left canonical record candidate.
        right_candidate : RecordCandidate
            Right canonical record candidate.

        Returns
        -------
        bool
            Whether every aligned field matches under its child-space equality
            contract.
        """
        self.validate(left_candidate)
        self.validate(right_candidate)
        return all(
            child_candidates_equal(
                space,
                left_candidate.entries[index][1],
                right_candidate.entries[index][1],
            )
            for index, (_name, space) in enumerate(self._fields)
        )

    @override
    def sample(self, random_state: np.random.RandomState) -> RecordCandidate:
        """Sample a canonical record candidate.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state object that owns all stochasticity for the sample.

        Returns
        -------
        RecordCandidate
            Canonical sampled record candidate.
        """
        entries = tuple(
            (name, sample_child_space(space, random_state))
            for name, space in self._fields
        )
        return RecordCandidate(entries=entries)

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return editable record leaf paths.

        Returns
        -------
        tuple[LeafPath, ...]
            Canonical leaf paths prefixed by record field name.
        """
        paths: list[LeafPath] = []
        for name, child_space in self._fields:
            for child_path in child_space.leaf_paths():
                paths.append((name,) + child_path)
        return tuple(paths)

    @override
    def active_leaf_paths_for_validated_candidate(
        self,
        candidate: RecordCandidate,
    ) -> tuple[LeafPath, ...]:
        """Return all record leaves for an already validated candidate.

        Parameters
        ----------
        candidate : RecordCandidate
            Canonical record candidate already validated by the current operation.

        Returns
        -------
        tuple[LeafPath, ...]
            Canonical leaf paths prefixed by record field name.
        """
        return self.leaf_paths()

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the leaf space at a record path.

        Parameters
        ----------
        path : LeafPath
            Leaf path whose first segment identifies the record field.

        Returns
        -------
        StructuredLeafSpace
            Leaf space declared for ``path``.
        """
        if len(path) == 0:
            msg = "record paths must include at least one segment"
            raise TypeError(msg)

        segment = path[0]
        if not isinstance(segment, str):
            msg = f"path {path!r} is invalid for record child traversal"
            raise TypeError(msg)

        child_space = self._child_spaces_by_name.get(segment)
        if child_space is None:
            msg = f"path {path!r} references an unknown record field"
            raise TypeError(msg)
        return child_space.leaf_space_at_path(path[1:])

    @override
    def leaf_value_at_path(
        self,
        candidate: RecordCandidate,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the leaf value stored at a record path.

        Parameters
        ----------
        candidate : RecordCandidate
            Canonical record candidate.
        path : LeafPath
            Leaf path whose first segment identifies the record field.

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
        candidate: RecordCandidate,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return one record leaf for an already validated candidate.

        Parameters
        ----------
        candidate : RecordCandidate
            Canonical record candidate already validated by the current operation.
        path : LeafPath
            Leaf path whose first segment identifies the record field.

        Returns
        -------
        SpaceCandidateValue
            Canonical leaf value stored at ``path``.
        """
        if len(path) == 0:
            msg = "record paths must include at least one segment"
            raise TypeError(msg)

        segment = path[0]
        if not isinstance(segment, str):
            msg = f"path {path!r} is invalid for record candidate traversal"
            raise TypeError(msg)

        field_index = self._field_indices.get(segment)
        if field_index is None:
            msg = f"path {path!r} references an unknown record field"
            raise TypeError(msg)
        child_space = self._fields[field_index][1]
        return leaf_value_at_validated_child_space(
            child_space,
            candidate.entries[field_index][1],
            path[1:],
        )

    @override
    def replace_leaf_values(
        self,
        candidate: RecordCandidate,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> RecordCandidate:
        """Return a record candidate with selected leaves replaced.

        Parameters
        ----------
        candidate : RecordCandidate
            Canonical record candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by field-prefixed leaf paths.

        Returns
        -------
        RecordCandidate
            Updated canonical record candidate.
        """
        self.validate(candidate)
        return self.replace_leaf_values_in_validated_candidate(
            candidate,
            replacements,
        )

    @override
    def replace_leaf_values_in_validated_candidate(
        self,
        candidate: RecordCandidate,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> RecordCandidate:
        """Return replacements for an already validated record candidate.

        Parameters
        ----------
        candidate : RecordCandidate
            Canonical record candidate already validated by the current operation.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by field-prefixed leaf paths.

        Returns
        -------
        RecordCandidate
            Updated canonical record candidate.
        """
        grouped_replacements = group_child_replacements(replacements)
        if len(grouped_replacements) == 0:
            return candidate

        for segment in grouped_replacements:
            if not isinstance(segment, str) or segment not in self._child_spaces_by_name:
                msg = f"replacement path references an unknown record field: {segment!r}"
                raise TypeError(msg)

        replaced_entries: list[tuple[str, SpaceCandidateValue]] = []
        for index, (name, child_space) in enumerate(self._fields):
            child_replacements = grouped_replacements.get(name)
            if child_replacements is None:
                replaced_entries.append((name, candidate.entries[index][1]))
                continue
            replaced_entries.append(
                (
                    name,
                    replace_leaf_values_in_validated_child_space(
                        child_space,
                        candidate.entries[index][1],
                        child_replacements,
                    ),
                )
            )
        return RecordCandidate(entries=tuple(replaced_entries))
