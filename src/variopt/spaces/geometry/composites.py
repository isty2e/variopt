"""Built-in composite structured-space geometry implementations."""

from dataclasses import dataclass, field
from math import log
from typing import Protocol, TypeGuard

from ..composites import RecordCandidate
from ..scalar import IntegerSpace, RealSpace
from ..types import SpaceCandidateValue
from .contracts import StructuredSpaceGeometry
from .leaf import (
    require_geometry_candidate_tuple,
    require_geometry_real_candidate,
    require_geometry_record_candidate,
)
from .parts import StructuredDistanceParts
from .permutation import PermutationSpaceGeometry
from .scalar import CategoricalSpaceGeometry, IntegerSpaceGeometry, RealSpaceGeometry

_SCALAR_LEAF_GEOMETRY_TYPES = (
    CategoricalSpaceGeometry,
    IntegerSpaceGeometry,
    RealSpaceGeometry,
)


class DistancePartValuesGeometry(StructuredSpaceGeometry, Protocol):
    """Internal geometry shape that exposes allocation-free distance parts."""

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values without allocating a parts object."""
        ...


class ValidatedDistancePartValuesGeometry(DistancePartValuesGeometry, Protocol):
    """Internal geometry shape for already validated canonical candidates."""

    def distance_part_values_for_validated_candidates(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values without repeating public validation."""
        ...


@dataclass(frozen=True, slots=True)
class TupleSpaceGeometry:
    """Fast geometry for one heterogeneous tuple space.

    Parameters
    ----------
    arity : int
        Declared tuple arity.
    child_geometries : tuple[StructuredSpaceGeometry, ...]
        Compiled child geometries in tuple order.
    """

    arity: int
    child_geometries: tuple[StructuredSpaceGeometry, ...]
    categorical_child_geometries: tuple[CategoricalSpaceGeometry, ...] | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Cache the closed all-categorical child geometry view, when present."""
        object.__setattr__(
            self,
            "categorical_child_geometries",
            collect_categorical_geometries(self.child_geometries),
        )

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return one structural distance decomposition for one tuple candidate.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left canonical tuple candidate.
        right : SpaceCandidateValue
            Right canonical tuple candidate.

        Returns
        -------
        StructuredDistanceParts
            Composite distance decomposition aggregated across child geometries.

        Raises
        ------
        TypeError
            If either candidate is not a canonical tuple.
        ValueError
            If a tuple length does not match ``arity``.
        """
        overlap_squared_distance, shared_leaf_count, topology_mismatch_leaf_count = (
            self.distance_part_values(left, right)
        )
        return StructuredDistanceParts(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for one tuple candidate."""
        left_tuple = require_geometry_candidate_tuple(
            value=left,
            message="tuple-space fast diversity path requires canonical tuple candidates",
        )
        right_tuple = require_geometry_candidate_tuple(
            value=right,
            message="tuple-space fast diversity path requires canonical tuple candidates",
        )
        if len(left_tuple) != self.arity or len(right_tuple) != self.arity:
            msg = "tuple candidate length does not match the declared arity"
            raise ValueError(msg)

        squared_distance = 0.0
        shared_leaf_count = 0
        topology_mismatch_leaf_count = 0
        for index, child_geometry in enumerate(self.child_geometries):
            left_value = left_tuple[index]
            right_value = right_tuple[index]
            if isinstance(child_geometry, _SCALAR_LEAF_GEOMETRY_TYPES):
                squared_distance += child_geometry.squared_distance(
                    left_value,
                    right_value,
                )
                shared_leaf_count += 1
                continue
            if geometry_has_distance_part_values(child_geometry):
                (
                    child_squared_distance,
                    child_shared_leaf_count,
                    child_topology_mismatch_leaf_count,
                ) = child_geometry.distance_part_values(
                    left_value,
                    right_value,
                )
                squared_distance += child_squared_distance
                shared_leaf_count += child_shared_leaf_count
                topology_mismatch_leaf_count += child_topology_mismatch_leaf_count
                continue
            child_parts = child_geometry.distance_parts(
                left_value,
                right_value,
            )
            squared_distance += child_parts.overlap_squared_distance
            shared_leaf_count += child_parts.shared_leaf_count
            topology_mismatch_leaf_count += child_parts.topology_mismatch_leaf_count
        return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)

    def distance_part_values_for_validated_candidates(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw part values for canonical tuple candidates."""
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            return self.distance_part_values(left, right)

        categorical_child_geometries = self.categorical_child_geometries
        if categorical_child_geometries is not None:
            mismatch_count = 0.0
            for index, child_geometry in enumerate(categorical_child_geometries):
                mismatch_count += (
                    child_geometry.squared_distance_for_validated_candidates(
                        left[index],
                        right[index],
                    )
                )
            return (mismatch_count, self.arity, 0)

        squared_distance = 0.0
        shared_leaf_count = 0
        topology_mismatch_leaf_count = 0
        for index, child_geometry in enumerate(self.child_geometries):
            left_value = left[index]
            right_value = right[index]
            if isinstance(child_geometry, _SCALAR_LEAF_GEOMETRY_TYPES):
                (
                    child_squared_distance,
                    child_shared_leaf_count,
                    child_topology_mismatch_leaf_count,
                ) = child_geometry.distance_part_values_for_validated_candidates(
                    left_value,
                    right_value,
                )
                squared_distance += child_squared_distance
                shared_leaf_count += child_shared_leaf_count
                topology_mismatch_leaf_count += child_topology_mismatch_leaf_count
                continue
            if geometry_has_validated_distance_part_values(child_geometry):
                (
                    child_squared_distance,
                    child_shared_leaf_count,
                    child_topology_mismatch_leaf_count,
                ) = child_geometry.distance_part_values_for_validated_candidates(
                    left_value,
                    right_value,
                )
                squared_distance += child_squared_distance
                shared_leaf_count += child_shared_leaf_count
                topology_mismatch_leaf_count += child_topology_mismatch_leaf_count
                continue
            child_parts = child_geometry.distance_parts(
                left_value,
                right_value,
            )
            squared_distance += child_parts.overlap_squared_distance
            shared_leaf_count += child_parts.shared_leaf_count
            topology_mismatch_leaf_count += child_parts.topology_mismatch_leaf_count
        return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)


@dataclass(frozen=True, slots=True)
class RecordSpaceGeometry:
    """Fast geometry for one heterogeneous record space.

    Parameters
    ----------
    field_geometries : tuple[tuple[str, StructuredSpaceGeometry], ...]
        Field-name and geometry pairs in canonical record order.
    """

    field_geometries: tuple[tuple[str, StructuredSpaceGeometry], ...]
    categorical_field_geometries: (
        tuple[tuple[str, CategoricalSpaceGeometry], ...] | None
    ) = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Cache the closed all-categorical field geometry view, when present."""
        object.__setattr__(
            self,
            "categorical_field_geometries",
            collect_categorical_field_geometries(self.field_geometries),
        )

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return one structural distance decomposition for one record candidate.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left canonical record candidate.
        right : SpaceCandidateValue
            Right canonical record candidate.

        Returns
        -------
        StructuredDistanceParts
            Composite distance decomposition aggregated across record fields.

        Raises
        ------
        TypeError
            If either candidate is not a canonical record.
        ValueError
            If the record keys do not match the declared field order.
        """
        overlap_squared_distance, shared_leaf_count, topology_mismatch_leaf_count = (
            self.distance_part_values(left, right)
        )
        return StructuredDistanceParts(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for one record candidate."""
        left_record = require_geometry_record_candidate(
            value=left,
            message="record-space fast diversity path requires canonical record candidates",
        )
        right_record = require_geometry_record_candidate(
            value=right,
            message="record-space fast diversity path requires canonical record candidates",
        )
        left_entries = left_record.entries
        right_entries = right_record.entries
        if len(left_entries) != len(self.field_geometries) or len(right_entries) != len(
            self.field_geometries
        ):
            msg = "record candidate keys must exactly match the declared fields"
            raise ValueError(msg)
        for index, (name, _child_geometry) in enumerate(self.field_geometries):
            left_name = left_entries[index][0]
            right_name = right_entries[index][0]
            if left_name != name or right_name != name:
                msg = "record candidate keys must exactly match the declared fields"
                raise ValueError(msg)

        squared_distance = 0.0
        shared_leaf_count = 0
        topology_mismatch_leaf_count = 0
        for index, (_name, child_geometry) in enumerate(self.field_geometries):
            left_value = left_entries[index][1]
            right_value = right_entries[index][1]
            if isinstance(child_geometry, _SCALAR_LEAF_GEOMETRY_TYPES):
                squared_distance += child_geometry.squared_distance(
                    left_value,
                    right_value,
                )
                shared_leaf_count += 1
                continue
            if geometry_has_distance_part_values(child_geometry):
                (
                    child_squared_distance,
                    child_shared_leaf_count,
                    child_topology_mismatch_leaf_count,
                ) = child_geometry.distance_part_values(
                    left_value,
                    right_value,
                )
                squared_distance += child_squared_distance
                shared_leaf_count += child_shared_leaf_count
                topology_mismatch_leaf_count += child_topology_mismatch_leaf_count
                continue
            child_parts = child_geometry.distance_parts(
                left_value,
                right_value,
            )
            squared_distance += child_parts.overlap_squared_distance
            shared_leaf_count += child_parts.shared_leaf_count
            topology_mismatch_leaf_count += child_parts.topology_mismatch_leaf_count
        return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)

    def distance_part_values_for_validated_candidates(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw part values for canonical record candidates."""
        if not isinstance(left, RecordCandidate) or not isinstance(
            right, RecordCandidate
        ):
            return self.distance_part_values(left, right)

        left_entries = left.entries
        right_entries = right.entries
        categorical_field_geometries = self.categorical_field_geometries
        if categorical_field_geometries is not None:
            mismatch_count = 0.0
            for index, (_name, child_geometry) in enumerate(
                categorical_field_geometries
            ):
                mismatch_count += (
                    child_geometry.squared_distance_for_validated_candidates(
                        left_entries[index][1],
                        right_entries[index][1],
                    )
                )
            return (mismatch_count, len(categorical_field_geometries), 0)

        squared_distance = 0.0
        shared_leaf_count = 0
        topology_mismatch_leaf_count = 0
        for index, (_name, child_geometry) in enumerate(self.field_geometries):
            left_value = left_entries[index][1]
            right_value = right_entries[index][1]
            if isinstance(child_geometry, _SCALAR_LEAF_GEOMETRY_TYPES):
                (
                    child_squared_distance,
                    child_shared_leaf_count,
                    child_topology_mismatch_leaf_count,
                ) = child_geometry.distance_part_values_for_validated_candidates(
                    left_value,
                    right_value,
                )
                squared_distance += child_squared_distance
                shared_leaf_count += child_shared_leaf_count
                topology_mismatch_leaf_count += child_topology_mismatch_leaf_count
                continue
            if geometry_has_validated_distance_part_values(child_geometry):
                (
                    child_squared_distance,
                    child_shared_leaf_count,
                    child_topology_mismatch_leaf_count,
                ) = child_geometry.distance_part_values_for_validated_candidates(
                    left_value,
                    right_value,
                )
                squared_distance += child_squared_distance
                shared_leaf_count += child_shared_leaf_count
                topology_mismatch_leaf_count += child_topology_mismatch_leaf_count
                continue
            child_parts = child_geometry.distance_parts(
                left_value,
                right_value,
            )
            squared_distance += child_parts.overlap_squared_distance
            shared_leaf_count += child_parts.shared_leaf_count
            topology_mismatch_leaf_count += child_parts.topology_mismatch_leaf_count
        return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)


@dataclass(frozen=True, slots=True)
class ArraySpaceGeometry:
    """Fast geometry for one homogeneous array space.

    Parameters
    ----------
    length : int
        Declared array length.
    element_geometry : StructuredSpaceGeometry
        Compiled geometry reused for every array element.
    """

    length: int
    element_geometry: StructuredSpaceGeometry

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return one structural distance decomposition for one homogeneous array.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left canonical array candidate.
        right : SpaceCandidateValue
            Right canonical array candidate.

        Returns
        -------
        StructuredDistanceParts
            Composite distance decomposition aggregated across array elements.

        Raises
        ------
        TypeError
            If either candidate is not a canonical tuple.
        ValueError
            If an array length does not match ``length``.
        """
        overlap_squared_distance, shared_leaf_count, topology_mismatch_leaf_count = (
            self.distance_part_values(left, right)
        )
        return StructuredDistanceParts(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for one homogeneous array."""
        left_tuple = require_geometry_candidate_tuple(
            value=left,
            message="array-space fast diversity path requires canonical tuple candidates",
        )
        right_tuple = require_geometry_candidate_tuple(
            value=right,
            message="array-space fast diversity path requires canonical tuple candidates",
        )
        if len(left_tuple) != self.length or len(right_tuple) != self.length:
            msg = "array candidate length does not match the declared length"
            raise ValueError(msg)

        squared_distance = 0.0
        shared_leaf_count = 0
        topology_mismatch_leaf_count = 0
        element_geometry = self.element_geometry
        if isinstance(element_geometry, _SCALAR_LEAF_GEOMETRY_TYPES):
            for index in range(self.length):
                squared_distance += element_geometry.squared_distance(
                    left_tuple[index],
                    right_tuple[index],
                )
                shared_leaf_count += 1
            return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)

        if geometry_has_distance_part_values(element_geometry):
            for index in range(self.length):
                (
                    child_squared_distance,
                    child_shared_leaf_count,
                    child_topology_mismatch_leaf_count,
                ) = element_geometry.distance_part_values(
                    left_tuple[index],
                    right_tuple[index],
                )
                squared_distance += child_squared_distance
                shared_leaf_count += child_shared_leaf_count
                topology_mismatch_leaf_count += child_topology_mismatch_leaf_count
            return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)

        for index in range(self.length):
            child_parts = element_geometry.distance_parts(
                left_tuple[index],
                right_tuple[index],
            )
            squared_distance += child_parts.overlap_squared_distance
            shared_leaf_count += child_parts.shared_leaf_count
            topology_mismatch_leaf_count += child_parts.topology_mismatch_leaf_count
        return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)

    def distance_part_values_for_validated_candidates(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw part values for canonical homogeneous array candidates."""
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            return self.distance_part_values(left, right)

        squared_distance = 0.0
        shared_leaf_count = 0
        topology_mismatch_leaf_count = 0
        element_geometry = self.element_geometry
        if isinstance(element_geometry, _SCALAR_LEAF_GEOMETRY_TYPES):
            for index in range(self.length):
                (
                    child_squared_distance,
                    child_shared_leaf_count,
                    child_topology_mismatch_leaf_count,
                ) = element_geometry.distance_part_values_for_validated_candidates(
                    left[index],
                    right[index],
                )
                squared_distance += child_squared_distance
                shared_leaf_count += child_shared_leaf_count
                topology_mismatch_leaf_count += child_topology_mismatch_leaf_count
            return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)

        if geometry_has_validated_distance_part_values(element_geometry):
            for index in range(self.length):
                (
                    child_squared_distance,
                    child_shared_leaf_count,
                    child_topology_mismatch_leaf_count,
                ) = element_geometry.distance_part_values_for_validated_candidates(
                    left[index],
                    right[index],
                )
                squared_distance += child_squared_distance
                shared_leaf_count += child_shared_leaf_count
                topology_mismatch_leaf_count += child_topology_mismatch_leaf_count
            return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)

        for index in range(self.length):
            child_parts = element_geometry.distance_parts(
                left[index],
                right[index],
            )
            squared_distance += child_parts.overlap_squared_distance
            shared_leaf_count += child_parts.shared_leaf_count
            topology_mismatch_leaf_count += child_parts.topology_mismatch_leaf_count
        return (squared_distance, shared_leaf_count, topology_mismatch_leaf_count)


@dataclass(frozen=True, slots=True)
class BinaryArraySpaceGeometry:
    """Fast geometry for one binary integer array space.

    Parameters
    ----------
    length : int
        Declared array length.
    element_space : IntegerSpace
        Binary integer element space used by the array.
    """

    length: int
    element_space: IntegerSpace

    def __post_init__(self) -> None:
        """Reject incompatible array-space specializations."""
        if (
            self.element_space.low != 0
            or self.element_space.high != 1
            or self.element_space.scale != "linear"
        ):
            msg = "binary array geometry requires linear IntegerSpace(0, 1)"
            raise ValueError(msg)

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return one mismatch-count distance over a binary integer array.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left canonical binary array candidate.
        right : SpaceCandidateValue
            Right canonical binary array candidate.

        Returns
        -------
        StructuredDistanceParts
            Distance decomposition whose overlap term is the mismatch count.

        Raises
        ------
        TypeError
            If either candidate is not a canonical integer tuple.
        ValueError
            If an array length does not match ``length`` or a value is outside
            ``{0, 1}``.
        """
        overlap_squared_distance, shared_leaf_count, topology_mismatch_leaf_count = (
            self.distance_part_values(left, right)
        )
        return StructuredDistanceParts(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for one binary integer array."""
        left_tuple = require_geometry_candidate_tuple(
            value=left,
            message="binary-array diversity requires canonical tuple candidates",
        )
        right_tuple = require_geometry_candidate_tuple(
            value=right,
            message="binary-array diversity requires canonical tuple candidates",
        )
        if len(left_tuple) != self.length or len(right_tuple) != self.length:
            msg = "binary array candidate length does not match the declared length"
            raise ValueError(msg)

        return self.distance_part_values_for_validated_candidates(
            left_tuple,
            right_tuple,
        )

    def distance_part_values_for_validated_candidates(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw part values for canonical binary integer arrays."""
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            return self.distance_part_values(left, right)

        mismatch_count = 0.0
        for index in range(self.length):
            left_value = left[index]
            right_value = right[index]
            if type(left_value) is not int or type(right_value) is not int:
                msg = "binary-array diversity requires canonical integer candidates"
                raise TypeError(msg)
            if left_value not in {0, 1} or right_value not in {0, 1}:
                msg = "binary-array diversity requires only 0/1 candidate values"
                raise ValueError(msg)
            if left_value != right_value:
                mismatch_count += 1.0
        return (mismatch_count, self.length, 0)


@dataclass(frozen=True, slots=True)
class IntegerArraySpaceGeometry:
    """Fast geometry for one homogeneous integer array space.

    Parameters
    ----------
    length : int
        Declared array length.
    element_space : IntegerSpace
        Integer element space used by the array.
    """

    length: int
    element_space: IntegerSpace

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return one normalized squared distance over an integer array.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left canonical integer array candidate.
        right : SpaceCandidateValue
            Right canonical integer array candidate.

        Returns
        -------
        StructuredDistanceParts
            Distance decomposition over the integer array.

        Raises
        ------
        TypeError
            If either candidate is not a canonical integer tuple.
        ValueError
            If an array length does not match ``length`` or a value is outside
            the declared integer bounds.
        """
        overlap_squared_distance, shared_leaf_count, topology_mismatch_leaf_count = (
            self.distance_part_values(left, right)
        )
        return StructuredDistanceParts(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for one integer array."""
        left_tuple = require_geometry_candidate_tuple(
            value=left,
            message="integer-array diversity requires canonical tuple candidates",
        )
        right_tuple = require_geometry_candidate_tuple(
            value=right,
            message="integer-array diversity requires canonical tuple candidates",
        )
        if len(left_tuple) != self.length or len(right_tuple) != self.length:
            msg = "integer array candidate length does not match the declared length"
            raise ValueError(msg)

        return self.distance_part_values_for_validated_candidates(
            left_tuple,
            right_tuple,
        )

    def distance_part_values_for_validated_candidates(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw part values for canonical integer arrays."""
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            return self.distance_part_values(left, right)

        element_space = self.element_space
        if element_space.low == element_space.high:
            for index in range(self.length):
                left_value = left[index]
                right_value = right[index]
                if type(left_value) is not int or type(right_value) is not int:
                    msg = (
                        "integer-array diversity requires canonical integer candidates"
                    )
                    raise TypeError(msg)
                if (
                    left_value < element_space.low
                    or left_value > element_space.high
                    or right_value < element_space.low
                    or right_value > element_space.high
                ):
                    msg = "integer candidate is outside the declared bounds"
                    raise ValueError(msg)
            return (0.0, self.length, 0)

        squared_distance = 0.0
        if element_space.scale == "log":
            coordinate_span = log(float(element_space.high)) - log(
                float(element_space.low)
            )
            for index in range(self.length):
                left_value = left[index]
                right_value = right[index]
                if type(left_value) is not int or type(right_value) is not int:
                    msg = (
                        "integer-array diversity requires canonical integer candidates"
                    )
                    raise TypeError(msg)
                if (
                    left_value < element_space.low
                    or left_value > element_space.high
                    or right_value < element_space.low
                    or right_value > element_space.high
                ):
                    msg = "integer candidate is outside the declared bounds"
                    raise ValueError(msg)
                leaf_distance = (
                    abs(log(float(left_value)) - log(float(right_value)))
                    / coordinate_span
                )
                squared_distance += leaf_distance * leaf_distance
            return (squared_distance, self.length, 0)

        coordinate_span = float(element_space.high - element_space.low)
        for index in range(self.length):
            left_value = left[index]
            right_value = right[index]
            if type(left_value) is not int or type(right_value) is not int:
                msg = "integer-array diversity requires canonical integer candidates"
                raise TypeError(msg)
            if (
                left_value < element_space.low
                or left_value > element_space.high
                or right_value < element_space.low
                or right_value > element_space.high
            ):
                msg = "integer candidate is outside the declared bounds"
                raise ValueError(msg)
            leaf_distance = abs(float(left_value - right_value)) / coordinate_span
            squared_distance += leaf_distance * leaf_distance
        return (squared_distance, self.length, 0)


@dataclass(frozen=True, slots=True)
class RealArraySpaceGeometry:
    """Fast geometry for one homogeneous real array space.

    Parameters
    ----------
    length : int
        Declared array length.
    element_space : RealSpace
        Real-valued element space used by the array.
    """

    length: int
    element_space: RealSpace

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return one normalized squared distance over a real array.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left canonical real array candidate.
        right : SpaceCandidateValue
            Right canonical real array candidate.

        Returns
        -------
        StructuredDistanceParts
            Distance decomposition over the real array.

        Raises
        ------
        TypeError
            If either candidate is not a canonical numeric tuple.
        ValueError
            If an array length does not match ``length`` or a value is outside
            the declared real bounds.
        """
        overlap_squared_distance, shared_leaf_count, topology_mismatch_leaf_count = (
            self.distance_part_values(left, right)
        )
        return StructuredDistanceParts(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for one real array."""
        left_tuple = require_geometry_candidate_tuple(
            value=left,
            message="real-array diversity requires canonical tuple candidates",
        )
        right_tuple = require_geometry_candidate_tuple(
            value=right,
            message="real-array diversity requires canonical tuple candidates",
        )
        if len(left_tuple) != self.length or len(right_tuple) != self.length:
            msg = "real array candidate length does not match the declared length"
            raise ValueError(msg)

        return self.distance_part_values_for_validated_candidates(
            left_tuple,
            right_tuple,
        )

    def distance_part_values_for_validated_candidates(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw part values for canonical real arrays."""
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            return self.distance_part_values(left, right)

        element_space = self.element_space
        if element_space.low == element_space.high:
            for index in range(self.length):
                left_value = require_geometry_real_candidate(
                    value=left[index],
                    message="real-array diversity requires numeric left leaf values",
                )
                right_value = require_geometry_real_candidate(
                    value=right[index],
                    message="real-array diversity requires numeric right leaf values",
                )
                if (
                    left_value < element_space.low
                    or left_value > element_space.high
                    or right_value < element_space.low
                    or right_value > element_space.high
                ):
                    msg = "real candidate is outside the declared bounds"
                    raise ValueError(msg)
            return (0.0, self.length, 0)

        squared_distance = 0.0
        if element_space.scale == "log":
            coordinate_span = log(element_space.high) - log(element_space.low)
            for index in range(self.length):
                left_value = require_geometry_real_candidate(
                    value=left[index],
                    message="real-array diversity requires numeric left leaf values",
                )
                right_value = require_geometry_real_candidate(
                    value=right[index],
                    message="real-array diversity requires numeric right leaf values",
                )
                if (
                    left_value < element_space.low
                    or left_value > element_space.high
                    or right_value < element_space.low
                    or right_value > element_space.high
                ):
                    msg = "real candidate is outside the declared bounds"
                    raise ValueError(msg)
                leaf_distance = (
                    abs(log(left_value) - log(right_value)) / coordinate_span
                )
                squared_distance += leaf_distance * leaf_distance
            return (squared_distance, self.length, 0)

        coordinate_span = element_space.high - element_space.low
        for index in range(self.length):
            left_value = require_geometry_real_candidate(
                value=left[index],
                message="real-array diversity requires numeric left leaf values",
            )
            right_value = require_geometry_real_candidate(
                value=right[index],
                message="real-array diversity requires numeric right leaf values",
            )
            if (
                left_value < element_space.low
                or left_value > element_space.high
                or right_value < element_space.low
                or right_value > element_space.high
            ):
                msg = "real candidate is outside the declared bounds"
                raise ValueError(msg)
            leaf_distance = abs(left_value - right_value) / coordinate_span
            squared_distance += leaf_distance * leaf_distance
        return (squared_distance, self.length, 0)


# Keep the closed built-in registry as the single concrete class list. The
# protocol above is structural so this tuple does not need a mirrored type union.
_DISTANCE_PART_VALUES_GEOMETRY_TYPES = (
    CategoricalSpaceGeometry,
    IntegerSpaceGeometry,
    RealSpaceGeometry,
    TupleSpaceGeometry,
    RecordSpaceGeometry,
    ArraySpaceGeometry,
    BinaryArraySpaceGeometry,
    IntegerArraySpaceGeometry,
    RealArraySpaceGeometry,
    PermutationSpaceGeometry,
)

_VALIDATED_DISTANCE_PART_VALUES_GEOMETRY_TYPES = _DISTANCE_PART_VALUES_GEOMETRY_TYPES


def geometry_has_distance_part_values(
    geometry: StructuredSpaceGeometry,
) -> TypeGuard[DistancePartValuesGeometry]:
    """Return whether a concrete built-in geometry exposes raw part values."""
    return isinstance(geometry, _DISTANCE_PART_VALUES_GEOMETRY_TYPES)


def geometry_has_validated_distance_part_values(
    geometry: StructuredSpaceGeometry,
) -> TypeGuard[ValidatedDistancePartValuesGeometry]:
    """Return whether a built-in geometry exposes a validated-candidate path."""
    return isinstance(geometry, _VALIDATED_DISTANCE_PART_VALUES_GEOMETRY_TYPES)


def collect_categorical_geometries(
    geometries: tuple[StructuredSpaceGeometry, ...],
) -> tuple[CategoricalSpaceGeometry, ...] | None:
    """Return categorical geometries when every child is categorical."""
    collected: list[CategoricalSpaceGeometry] = []
    for geometry in geometries:
        if not isinstance(geometry, CategoricalSpaceGeometry):
            return None
        collected.append(geometry)
    return tuple(collected)


def collect_categorical_field_geometries(
    geometries: tuple[tuple[str, StructuredSpaceGeometry], ...],
) -> tuple[tuple[str, CategoricalSpaceGeometry], ...] | None:
    """Return categorical field geometries when every record field is categorical."""
    collected: list[tuple[str, CategoricalSpaceGeometry]] = []
    for name, geometry in geometries:
        if not isinstance(geometry, CategoricalSpaceGeometry):
            return None
        collected.append((name, geometry))
    return tuple(collected)


def collect_child_geometries(
    geometries: tuple[StructuredSpaceGeometry | None, ...],
) -> tuple[StructuredSpaceGeometry, ...] | None:
    """Return one all-or-nothing tuple of compiled child geometries.

    Parameters
    ----------
    geometries : tuple[StructuredSpaceGeometry | None, ...]
        Child geometry candidates in tuple order.

    Returns
    -------
    tuple[StructuredSpaceGeometry, ...] | None
        Fully collected child geometries, or ``None`` when any child geometry
        is unavailable.
    """
    collected: list[StructuredSpaceGeometry] = []
    for geometry in geometries:
        if geometry is None:
            return None
        collected.append(geometry)
    return tuple(collected)


def collect_field_geometries(
    geometries: tuple[tuple[str, StructuredSpaceGeometry | None], ...],
) -> tuple[tuple[str, StructuredSpaceGeometry], ...] | None:
    """Return one all-or-nothing tuple of compiled field geometries.

    Parameters
    ----------
    geometries : tuple[tuple[str, StructuredSpaceGeometry | None], ...]
        Field-name and geometry pairs in record order.

    Returns
    -------
    tuple[tuple[str, StructuredSpaceGeometry], ...] | None
        Fully collected field geometries, or ``None`` when any field geometry
        is unavailable.
    """
    collected: list[tuple[str, StructuredSpaceGeometry]] = []
    for name, geometry in geometries:
        if geometry is None:
            return None
        collected.append((name, geometry))
    return tuple(collected)
