"""Composite child-space adapters and replacement routing helpers."""

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias, TypeGuard

import numpy as np

from ..permutation import PermutationSpace
from ..scalar import CategoricalSpace, IntegerSpace, RealSpace
from ..structured import LeafPath, StructuredLeafSpace
from ..types import SpaceBoundaryValue, SpaceCandidateValue, SpaceScalarValue
from .records import RecordCandidate


class CompositeSequenceSpace(Protocol):
    """Protocol for tuple- or array-shaped composite child spaces.

    Notes
    -----
    This protocol captures the operations that heterogeneous composite adapters
    need from sequence-shaped child spaces. Concrete implementations are
    typically :class:`variopt.spaces.composites.TupleSpace` or
    :class:`variopt.spaces.composites.ArraySpace`.
    """

    def normalize(
        self,
        raw_candidate: Sequence[SpaceBoundaryValue],
    ) -> tuple[SpaceCandidateValue, ...]:
        """Normalize one raw sequence-shaped child candidate.

        Parameters
        ----------
        raw_candidate : Sequence[SpaceBoundaryValue]
            Raw child candidate supplied through a composite boundary. Elements
            may still be boundary values instead of canonical candidate values.

        Returns
        -------
        tuple[SpaceCandidateValue, ...]
            Canonical child candidate for the sequence-shaped composite.
        """
        ...

    def validate(self, candidate: tuple[SpaceCandidateValue, ...]) -> None:
        """Validate one canonical sequence-shaped child candidate.

        Parameters
        ----------
        candidate : tuple[SpaceCandidateValue, ...]
            Canonical child candidate to validate.

        Raises
        ------
        TypeError
            If the candidate shape or element types do not match the child
            space contract.
        ValueError
            If the candidate values violate the child space domain.
        """
        ...

    def candidates_equal(
        self,
        left_candidate: tuple[SpaceCandidateValue, ...],
        right_candidate: tuple[SpaceCandidateValue, ...],
    ) -> bool:
        """Return whether two canonical sequence candidates denote one point."""
        ...

    def sample(
        self,
        random_state: np.random.RandomState,
    ) -> tuple[SpaceCandidateValue, ...]:
        """Sample one canonical sequence-shaped child candidate.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random generator used for deterministic sampling.

        Returns
        -------
        tuple[SpaceCandidateValue, ...]
            Fresh canonical child candidate.
        """
        ...

    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return all declared leaf paths for this composite child.

        Returns
        -------
        tuple[LeafPath, ...]
            Canonical leaf paths relative to this child space.
        """
        ...

    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the leaf space at one child path.

        Parameters
        ----------
        path : LeafPath
            Child-relative leaf path.

        Returns
        -------
        StructuredLeafSpace
            Leaf space declared at ``path``.
        """
        ...

    def leaf_value_at_path(
        self,
        candidate: tuple[SpaceCandidateValue, ...],
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the leaf value stored at one path.

        Parameters
        ----------
        candidate : tuple[SpaceCandidateValue, ...]
            Canonical child candidate to inspect.
        path : LeafPath
            Child-relative leaf path to read.

        Returns
        -------
        SpaceCandidateValue
            Canonical leaf value stored at ``path``.
        """
        ...

    def replace_leaf_values(
        self,
        candidate: tuple[SpaceCandidateValue, ...],
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> tuple[SpaceCandidateValue, ...]:
        """Return one candidate with leaf replacements applied.

        Parameters
        ----------
        candidate : tuple[SpaceCandidateValue, ...]
            Canonical child candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Leaf-relative replacement mapping keyed by child-relative paths.

        Returns
        -------
        tuple[SpaceCandidateValue, ...]
            Updated canonical child candidate.
        """
        ...


class CompositeRecordSpace(Protocol):
    """Protocol for record-shaped composite child spaces.

    Notes
    -----
    This protocol captures the record-oriented operations required by composite
    adapters when they route normalization, validation, sampling, and leaf
    replacement through heterogeneous child spaces.
    """

    @property
    def fields(self) -> tuple[tuple[str, "CompositeChildSpace"], ...]:
        """Return the declared record fields in canonical order.

        Returns
        -------
        tuple[tuple[str, CompositeChildSpace], ...]
            Field declarations as ``(name, child_space)`` pairs.
        """
        ...

    def normalize(
        self,
        raw_candidate: Mapping[str, SpaceBoundaryValue] | RecordCandidate,
    ) -> RecordCandidate:
        """Normalize one raw record-shaped child candidate.

        Parameters
        ----------
        raw_candidate : Mapping[str, SpaceBoundaryValue] | RecordCandidate
            Raw child candidate supplied through a composite boundary.

        Returns
        -------
        RecordCandidate
            Canonical record-shaped child candidate.
        """
        ...

    def validate(self, candidate: RecordCandidate) -> None:
        """Validate one canonical record-shaped child candidate.

        Parameters
        ----------
        candidate : RecordCandidate
            Canonical child candidate to validate.

        Raises
        ------
        TypeError
            If the candidate layout or field types are invalid.
        ValueError
            If any field value lies outside the declared child space.
        """
        ...

    def candidates_equal(
        self,
        left_candidate: RecordCandidate,
        right_candidate: RecordCandidate,
    ) -> bool:
        """Return whether two canonical record candidates denote one point."""
        ...

    def sample(self, random_state: np.random.RandomState) -> RecordCandidate:
        """Sample one canonical record-shaped child candidate.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random generator used for deterministic sampling.

        Returns
        -------
        RecordCandidate
            Fresh canonical record-shaped child candidate.
        """
        ...

    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return all declared leaf paths for this composite child.

        Returns
        -------
        tuple[LeafPath, ...]
            Canonical leaf paths relative to this child space.
        """
        ...

    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the leaf space at one child path.

        Parameters
        ----------
        path : LeafPath
            Child-relative leaf path.

        Returns
        -------
        StructuredLeafSpace
            Leaf space declared at ``path``.
        """
        ...

    def leaf_value_at_path(
        self,
        candidate: RecordCandidate,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the leaf value stored at one path.

        Parameters
        ----------
        candidate : RecordCandidate
            Canonical child candidate to inspect.
        path : LeafPath
            Child-relative leaf path to read.

        Returns
        -------
        SpaceCandidateValue
            Canonical leaf value stored at ``path``.
        """
        ...

    def replace_leaf_values(
        self,
        candidate: RecordCandidate,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> RecordCandidate:
        """Return one candidate with leaf replacements applied.

        Parameters
        ----------
        candidate : RecordCandidate
            Canonical child candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Leaf-relative replacement mapping keyed by child-relative paths.

        Returns
        -------
        RecordCandidate
            Updated canonical child candidate.
        """
        ...


CategoricalScalarSpace: TypeAlias = CategoricalSpace[SpaceScalarValue]
CompositeLeafChildSpace: TypeAlias = (
    RealSpace
    | IntegerSpace
    | CategoricalScalarSpace
    | PermutationSpace
)
CompositeNestedChildSpace: TypeAlias = (
    CompositeSequenceSpace
    | CompositeRecordSpace
)
CompositeChildSpace: TypeAlias = CompositeLeafChildSpace | CompositeNestedChildSpace


def is_categorical_child_space(
    space: CompositeChildSpace,
) -> TypeGuard[CategoricalScalarSpace]:
    """Return whether one composite child space is scalar categorical.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space to classify.

    Returns
    -------
    TypeGuard[CategoricalScalarSpace]
        ``True`` when ``space`` is a scalar categorical leaf space.
    """
    return isinstance(space, CategoricalSpace)


def is_record_composite_space(space: CompositeChildSpace) -> TypeGuard[CompositeRecordSpace]:
    """Return whether one composite child space is record-shaped.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space to classify.

    Returns
    -------
    TypeGuard[CompositeRecordSpace]
        ``True`` when ``space`` exposes the record-shaped composite protocol.
    """
    return hasattr(space, "fields")


def is_sequence_composite_space(
    space: CompositeChildSpace,
) -> TypeGuard[CompositeSequenceSpace]:
    """Return whether one composite child space is tuple- or array-shaped.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space to classify.

    Returns
    -------
    TypeGuard[CompositeSequenceSpace]
        ``True`` when ``space`` exposes the sequence-shaped composite
        protocol.
    """
    return hasattr(space, "child_spaces") or hasattr(space, "element_space")


def require_record_composite_space(space: CompositeChildSpace) -> CompositeRecordSpace:
    """Return one record-shaped composite child space or raise.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space to validate.

    Returns
    -------
    CompositeRecordSpace
        Record-shaped composite child space.

    Raises
    ------
    TypeError
        If ``space`` is not record-shaped.
    """
    if not is_record_composite_space(space):
        msg = "expected a record-shaped composite child space"
        raise TypeError(msg)
    return space


def require_sequence_composite_space(
    space: CompositeChildSpace,
) -> CompositeSequenceSpace:
    """Return one sequence-shaped composite child space or raise.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space to validate.

    Returns
    -------
    CompositeSequenceSpace
        Sequence-shaped composite child space.

    Raises
    ------
    TypeError
        If ``space`` is not sequence-shaped.
    """
    if not is_sequence_composite_space(space):
        msg = "expected a sequence-shaped composite child space"
        raise TypeError(msg)
    return space


def require_real_boundary_value(value: SpaceBoundaryValue) -> float | int:
    """Return one real-space boundary value or raise.

    Parameters
    ----------
    value : SpaceBoundaryValue
        Raw boundary value to validate.

    Returns
    -------
    float | int
        Numeric boundary value accepted by :class:`RealSpace`.

    Raises
    ------
    TypeError
        If ``value`` is not numeric or is a boolean.
    """
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        msg = "real child candidate must be numeric"
        raise TypeError(msg)
    return value


def require_integer_boundary_value(value: SpaceBoundaryValue) -> int:
    """Return one integer-space boundary value or raise.

    Parameters
    ----------
    value : SpaceBoundaryValue
        Raw boundary value to validate.

    Returns
    -------
    int
        Canonical integer boundary value.

    Raises
    ------
    TypeError
        If ``value`` is not a canonical integer.
    """
    if type(value) is not int:
        msg = "integer child candidate must be a canonical integer"
        raise TypeError(msg)
    return value


def require_scalar_boundary_value(value: SpaceBoundaryValue) -> SpaceScalarValue:
    """Return one scalar categorical value or raise.

    Parameters
    ----------
    value : SpaceBoundaryValue
        Raw boundary value to validate.

    Returns
    -------
    SpaceScalarValue
        Scalar categorical value suitable for
        :class:`CategoricalSpace` normalization.

    Raises
    ------
    TypeError
        If ``value`` is not one of the supported scalar types.
    """
    if isinstance(value, (bool, int, float, str, bytes, bytearray)):
        return value

    msg = "categorical child candidate must be scalar"
    raise TypeError(msg)


def require_permutation_boundary_value(value: SpaceBoundaryValue) -> Sequence[int]:
    """Return one permutation-space boundary value or raise.

    Parameters
    ----------
    value : SpaceBoundaryValue
        Raw boundary value to validate.

    Returns
    -------
    Sequence[int]
        Integer sequence suitable for permutation normalization.

    Raises
    ------
    TypeError
        If ``value`` is not a non-string integer sequence.
    """
    if isinstance(value, (bytes, bytearray, str)) or not isinstance(value, Sequence):
        msg = "permutation child candidate must be a non-string sequence"
        raise TypeError(msg)

    normalized_elements: list[int] = []
    for element in value:
        if type(element) is not int:
            msg = "permutation child candidate must contain only canonical integers"
            raise TypeError(msg)
        normalized_elements.append(element)
    return tuple(normalized_elements)


def require_record_boundary_value(
    value: SpaceBoundaryValue,
) -> Mapping[str, SpaceBoundaryValue] | RecordCandidate:
    """Return one record-composite boundary value or raise.

    Parameters
    ----------
    value : SpaceBoundaryValue
        Raw boundary value to validate.

    Returns
    -------
    Mapping[str, SpaceBoundaryValue] | RecordCandidate
        Record-shaped boundary payload accepted by a record child space.

    Raises
    ------
    TypeError
        If ``value`` is neither a mapping nor a :class:`RecordCandidate`.
    """
    if isinstance(value, RecordCandidate):
        return value

    if not isinstance(value, Mapping):
        msg = "record child candidate must be a mapping or RecordCandidate"
        raise TypeError(msg)
    return value


def require_sequence_boundary_value(
    value: SpaceBoundaryValue,
) -> Sequence[SpaceBoundaryValue]:
    """Return one sequence-composite boundary value or raise.

    Parameters
    ----------
    value : SpaceBoundaryValue
        Raw boundary value to validate.

    Returns
    -------
    Sequence[SpaceBoundaryValue]
        Sequence-shaped boundary payload accepted by a sequence child space.

    Raises
    ------
    TypeError
        If ``value`` is not a non-string sequence.
    """
    if isinstance(value, (bytes, bytearray, str)) or not isinstance(value, Sequence):
        msg = "sequence child candidate must be a non-string sequence"
        raise TypeError(msg)
    return value


def require_real_candidate(value: SpaceCandidateValue) -> float:
    """Return one canonical real child candidate or raise.

    Parameters
    ----------
    value : SpaceCandidateValue
        Canonical candidate value to validate.

    Returns
    -------
    float
        Canonical floating-point candidate value.

    Raises
    ------
    TypeError
        If ``value`` is not a canonical float.
    """
    if type(value) is not float:
        msg = "real child candidate must be a canonical float"
        raise TypeError(msg)
    return value


def require_integer_candidate(value: SpaceCandidateValue) -> int:
    """Return one canonical integer child candidate or raise.

    Parameters
    ----------
    value : SpaceCandidateValue
        Canonical candidate value to validate.

    Returns
    -------
    int
        Canonical integer candidate value.

    Raises
    ------
    TypeError
        If ``value`` is not a canonical integer.
    """
    if type(value) is not int:
        msg = "integer child candidate must be a canonical integer"
        raise TypeError(msg)
    return value


def require_scalar_candidate(value: SpaceCandidateValue) -> SpaceScalarValue:
    """Return one canonical scalar categorical candidate or raise.

    Parameters
    ----------
    value : SpaceCandidateValue
        Canonical candidate value to validate.

    Returns
    -------
    SpaceScalarValue
        Scalar categorical candidate value.

    Raises
    ------
    TypeError
        If ``value`` is not one of the supported scalar types.
    """
    if isinstance(value, (bool, int, float, str, bytes, bytearray)):
        return value

    msg = "categorical child candidate must be scalar"
    raise TypeError(msg)


def require_permutation_candidate(value: SpaceCandidateValue) -> tuple[int, ...]:
    """Return one canonical permutation child candidate or raise.

    Parameters
    ----------
    value : SpaceCandidateValue
        Canonical candidate value to validate.

    Returns
    -------
    tuple[int, ...]
        Canonical permutation candidate.

    Raises
    ------
    TypeError
        If ``value`` is not a tuple of canonical integers.
    """
    if not isinstance(value, tuple):
        msg = "permutation child candidate must be a canonical tuple"
        raise TypeError(msg)

    normalized_elements: list[int] = []
    for element in value:
        if type(element) is not int:
            msg = "permutation child candidate must contain only canonical integers"
            raise TypeError(msg)
        normalized_elements.append(element)
    return tuple(normalized_elements)


def require_sequence_candidate(
    value: SpaceCandidateValue,
) -> tuple[SpaceCandidateValue, ...]:
    """Return one canonical sequence-composite child candidate or raise.

    Parameters
    ----------
    value : SpaceCandidateValue
        Canonical candidate value to validate.

    Returns
    -------
    tuple[SpaceCandidateValue, ...]
        Canonical sequence-shaped child candidate.

    Raises
    ------
    TypeError
        If ``value`` is not a canonical tuple.
    """
    if not isinstance(value, tuple):
        msg = "sequence child candidate must be a canonical tuple"
        raise TypeError(msg)
    return value


def require_record_candidate(value: SpaceCandidateValue) -> RecordCandidate:
    """Return one canonical record-composite child candidate or raise.

    Parameters
    ----------
    value : SpaceCandidateValue
        Canonical candidate value to validate.

    Returns
    -------
    RecordCandidate
        Canonical record-shaped child candidate.

    Raises
    ------
    TypeError
        If ``value`` is not a :class:`RecordCandidate`.
    """
    if not isinstance(value, RecordCandidate):
        msg = "record child candidate must be a canonical RecordCandidate"
        raise TypeError(msg)
    return value


def group_child_replacements(
    replacements: Mapping[LeafPath, SpaceCandidateValue],
) -> dict[int | str, dict[LeafPath, SpaceCandidateValue]]:
    """Group non-empty replacement paths by their first child segment.

    Parameters
    ----------
    replacements : Mapping[LeafPath, SpaceCandidateValue]
        Flat mapping of replacement paths relative to a composite child space.

    Returns
    -------
    dict[int | str, dict[LeafPath, SpaceCandidateValue]]
        Nested mapping keyed by the first path segment, with the remainder of
        each path stored relative to that child.

    Raises
    ------
    TypeError
        If a replacement path does not identify a child segment.
    """
    grouped: dict[int | str, dict[LeafPath, SpaceCandidateValue]] = {}
    for path, replacement in replacements.items():
        if len(path) == 0:
            msg = "composite replacement paths must include at least one segment"
            raise TypeError(msg)
        head = path[0]
        tail = path[1:]
        if head not in grouped:
            grouped[head] = {}
        grouped[head][tail] = replacement
    return grouped


def normalize_child_space(
    space: CompositeChildSpace,
    raw_candidate: SpaceBoundaryValue,
) -> SpaceCandidateValue:
    """Normalize one heterogeneous child candidate.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space responsible for normalization.
    raw_candidate : SpaceBoundaryValue
        Raw boundary candidate supplied for that child space.

    Returns
    -------
    SpaceCandidateValue
        Canonical child candidate produced by ``space``.

    Raises
    ------
    TypeError
        If ``raw_candidate`` cannot be coerced into the boundary contract of
        ``space``.
    """
    if isinstance(space, RealSpace):
        return space.normalize(require_real_boundary_value(raw_candidate))

    if isinstance(space, IntegerSpace):
        return space.normalize(require_integer_boundary_value(raw_candidate))

    if is_categorical_child_space(space):
        return space.normalize(require_scalar_boundary_value(raw_candidate))

    if isinstance(space, PermutationSpace):
        return space.normalize(require_permutation_boundary_value(raw_candidate))

    if is_record_composite_space(space):
        return require_record_composite_space(space).normalize(
            require_record_boundary_value(raw_candidate),
        )

    if is_sequence_composite_space(space):
        return require_sequence_composite_space(space).normalize(
            require_sequence_boundary_value(raw_candidate),
        )

    msg = "unsupported composite child space"
    raise TypeError(msg)


def validate_child_space(
    space: CompositeChildSpace,
    candidate: SpaceCandidateValue,
) -> None:
    """Validate one heterogeneous canonical child candidate.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space responsible for validation.
    candidate : SpaceCandidateValue
        Canonical child candidate to validate.

    Raises
    ------
    TypeError
        If ``candidate`` does not match the canonical value shape required by
        ``space``.
    ValueError
        If ``candidate`` violates the child space domain.
    """
    if isinstance(space, RealSpace):
        space.validate(require_real_candidate(candidate))
        return

    if isinstance(space, IntegerSpace):
        space.validate(require_integer_candidate(candidate))
        return

    if is_categorical_child_space(space):
        space.validate(require_scalar_candidate(candidate))
        return

    if isinstance(space, PermutationSpace):
        space.validate(require_permutation_candidate(candidate))
        return

    if is_record_composite_space(space):
        require_record_composite_space(space).validate(
            require_record_candidate(candidate),
        )
        return

    if is_sequence_composite_space(space):
        require_sequence_composite_space(space).validate(
            require_sequence_candidate(candidate),
        )
        return

    msg = "unsupported composite child space"
    raise TypeError(msg)


def child_candidates_equal(
    space: CompositeChildSpace,
    left_candidate: SpaceCandidateValue,
    right_candidate: SpaceCandidateValue,
) -> bool:
    """Return candidate equality through a heterogeneous child-space seam.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space responsible for candidate identity semantics.
    left_candidate : SpaceCandidateValue
        Left canonical child candidate.
    right_candidate : SpaceCandidateValue
        Right canonical child candidate.

    Returns
    -------
    bool
        Whether the child candidates denote the same child-space point.

    Raises
    ------
    TypeError
        If either candidate does not match the child-space canonical shape.
    ValueError
        If either candidate violates the child-space domain.
    """
    if isinstance(space, RealSpace):
        return space.candidates_equal(
            require_real_candidate(left_candidate),
            require_real_candidate(right_candidate),
        )

    if isinstance(space, IntegerSpace):
        return space.candidates_equal(
            require_integer_candidate(left_candidate),
            require_integer_candidate(right_candidate),
        )

    if is_categorical_child_space(space):
        return space.candidates_equal(
            require_scalar_candidate(left_candidate),
            require_scalar_candidate(right_candidate),
        )

    if isinstance(space, PermutationSpace):
        return space.candidates_equal(
            require_permutation_candidate(left_candidate),
            require_permutation_candidate(right_candidate),
        )

    if is_record_composite_space(space):
        return require_record_composite_space(space).candidates_equal(
            require_record_candidate(left_candidate),
            require_record_candidate(right_candidate),
        )

    if is_sequence_composite_space(space):
        return require_sequence_composite_space(space).candidates_equal(
            require_sequence_candidate(left_candidate),
            require_sequence_candidate(right_candidate),
        )

    msg = "unsupported composite child space"
    raise TypeError(msg)


def sample_child_space(
    space: CompositeChildSpace,
    random_state: np.random.RandomState,
) -> SpaceCandidateValue:
    """Sample one heterogeneous canonical child candidate.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space to sample from.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    SpaceCandidateValue
        Fresh canonical child candidate from ``space``.
    """
    if isinstance(space, RealSpace):
        return space.sample(random_state)

    if isinstance(space, IntegerSpace):
        return space.sample(random_state)

    if is_categorical_child_space(space):
        return space.sample(random_state)

    if isinstance(space, PermutationSpace):
        return space.sample(random_state)

    if is_record_composite_space(space):
        return require_record_composite_space(space).sample(random_state)

    if is_sequence_composite_space(space):
        return require_sequence_composite_space(space).sample(random_state)

    msg = "unsupported composite child space"
    raise TypeError(msg)


def leaf_value_at_child_space(
    space: CompositeChildSpace,
    candidate: SpaceCandidateValue,
    path: LeafPath,
) -> SpaceCandidateValue:
    """Return one leaf value through a heterogeneous child-space seam.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space that owns the leaf path.
    candidate : SpaceCandidateValue
        Canonical child candidate to inspect.
    path : LeafPath
        Leaf path relative to ``space``.

    Returns
    -------
    SpaceCandidateValue
        Canonical leaf value stored at ``path``.

    Raises
    ------
    TypeError
        If ``candidate`` is not valid for ``space``.
    """
    if isinstance(space, RealSpace):
        return space.leaf_value_at_path(require_real_candidate(candidate), path)

    if isinstance(space, IntegerSpace):
        return space.leaf_value_at_path(require_integer_candidate(candidate), path)

    if is_categorical_child_space(space):
        return space.leaf_value_at_path(
            require_scalar_candidate(candidate),
            path,
        )

    if isinstance(space, PermutationSpace):
        return space.leaf_value_at_path(require_permutation_candidate(candidate), path)

    if is_record_composite_space(space):
        return require_record_composite_space(space).leaf_value_at_path(
            require_record_candidate(candidate),
            path,
        )

    if is_sequence_composite_space(space):
        return require_sequence_composite_space(space).leaf_value_at_path(
            require_sequence_candidate(candidate),
            path,
        )

    msg = "unsupported composite child space"
    raise TypeError(msg)


def replace_leaf_values_in_child_space(
    space: CompositeChildSpace,
    candidate: SpaceCandidateValue,
    replacements: Mapping[LeafPath, SpaceCandidateValue],
) -> SpaceCandidateValue:
    """Apply leaf replacements through a heterogeneous child-space seam.

    Parameters
    ----------
    space : CompositeChildSpace
        Child space that owns the candidate.
    candidate : SpaceCandidateValue
        Canonical child candidate to update.
    replacements : Mapping[LeafPath, SpaceCandidateValue]
        Leaf-relative replacement mapping.

    Returns
    -------
    SpaceCandidateValue
        Updated canonical child candidate.

    Raises
    ------
    TypeError
        If ``candidate`` is not valid for ``space`` or if replacement values do
        not match the leaf contract.
    """
    if isinstance(space, RealSpace):
        return space.replace_leaf_values(require_real_candidate(candidate), replacements)

    if isinstance(space, IntegerSpace):
        return space.replace_leaf_values(require_integer_candidate(candidate), replacements)

    if is_categorical_child_space(space):
        return space.replace_leaf_values(
            require_scalar_candidate(candidate),
            replacements,
        )

    if isinstance(space, PermutationSpace):
        return space.replace_leaf_values(
            require_permutation_candidate(candidate),
            replacements,
        )

    if is_record_composite_space(space):
        return require_record_composite_space(space).replace_leaf_values(
            require_record_candidate(candidate),
            replacements,
        )

    if is_sequence_composite_space(space):
        return require_sequence_composite_space(space).replace_leaf_values(
            require_sequence_candidate(candidate),
            replacements,
        )

    msg = "unsupported composite child space"
    raise TypeError(msg)
