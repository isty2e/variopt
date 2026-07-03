"""Built-in composite structured-space geometry implementations."""

from dataclasses import dataclass
from math import log

from ..scalar import IntegerSpace, RealSpace
from ..types import SpaceCandidateValue
from .contracts import StructuredSpaceGeometry
from .leaf import (
    require_candidate_tuple,
    require_real_candidate,
    require_record_candidate,
)
from .parts import StructuredDistanceParts
from .scalar import CategoricalSpaceGeometry, IntegerSpaceGeometry, RealSpaceGeometry

_SCALAR_LEAF_GEOMETRY_TYPES = (
    CategoricalSpaceGeometry,
    IntegerSpaceGeometry,
    RealSpaceGeometry,
)


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
        left_tuple = require_candidate_tuple(
            value=left,
            message="tuple-space fast diversity path requires canonical tuple candidates",
        )
        right_tuple = require_candidate_tuple(
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
            if isinstance(child_geometry, _SCALAR_LEAF_GEOMETRY_TYPES):
                squared_distance += child_geometry.squared_distance(
                    left_tuple[index],
                    right_tuple[index],
                )
                shared_leaf_count += 1
                continue
            child_parts = child_geometry.distance_parts(
                left_tuple[index],
                right_tuple[index],
            )
            squared_distance += child_parts.overlap_squared_distance
            shared_leaf_count += child_parts.shared_leaf_count
            topology_mismatch_leaf_count += child_parts.topology_mismatch_leaf_count
        return StructuredDistanceParts(
            overlap_squared_distance=squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )


@dataclass(frozen=True, slots=True)
class RecordSpaceGeometry:
    """Fast geometry for one heterogeneous record space.

    Parameters
    ----------
    field_geometries : tuple[tuple[str, StructuredSpaceGeometry], ...]
        Field-name and geometry pairs in canonical record order.
    """

    field_geometries: tuple[tuple[str, StructuredSpaceGeometry], ...]

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
        left_record = require_record_candidate(
            value=left,
            message="record-space fast diversity path requires canonical record candidates",
        )
        right_record = require_record_candidate(
            value=right,
            message="record-space fast diversity path requires canonical record candidates",
        )
        left_entries = left_record.entries
        right_entries = right_record.entries
        if (
            len(left_entries) != len(self.field_geometries)
            or len(right_entries) != len(self.field_geometries)
        ):
            msg = "record candidate keys must exactly match the declared fields"
            raise ValueError(msg)

        squared_distance = 0.0
        shared_leaf_count = 0
        topology_mismatch_leaf_count = 0
        for index, (name, child_geometry) in enumerate(self.field_geometries):
            left_name, left_value = left_entries[index]
            right_name, right_value = right_entries[index]
            if left_name != name or right_name != name:
                msg = "record candidate keys must exactly match the declared fields"
                raise ValueError(msg)
            if isinstance(child_geometry, _SCALAR_LEAF_GEOMETRY_TYPES):
                squared_distance += child_geometry.squared_distance(
                    left_value,
                    right_value,
                )
                shared_leaf_count += 1
                continue
            child_parts = child_geometry.distance_parts(
                left_value,
                right_value,
            )
            squared_distance += child_parts.overlap_squared_distance
            shared_leaf_count += child_parts.shared_leaf_count
            topology_mismatch_leaf_count += child_parts.topology_mismatch_leaf_count
        return StructuredDistanceParts(
            overlap_squared_distance=squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )


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
        left_tuple = require_candidate_tuple(
            value=left,
            message="array-space fast diversity path requires canonical tuple candidates",
        )
        right_tuple = require_candidate_tuple(
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
            return StructuredDistanceParts(
                overlap_squared_distance=squared_distance,
                shared_leaf_count=shared_leaf_count,
            )

        for index in range(self.length):
            child_parts = element_geometry.distance_parts(
                left_tuple[index],
                right_tuple[index],
            )
            squared_distance += child_parts.overlap_squared_distance
            shared_leaf_count += child_parts.shared_leaf_count
            topology_mismatch_leaf_count += child_parts.topology_mismatch_leaf_count
        return StructuredDistanceParts(
            overlap_squared_distance=squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )


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
        left_tuple = require_candidate_tuple(
            value=left,
            message="binary-array diversity requires canonical tuple candidates",
        )
        right_tuple = require_candidate_tuple(
            value=right,
            message="binary-array diversity requires canonical tuple candidates",
        )
        if len(left_tuple) != self.length or len(right_tuple) != self.length:
            msg = "binary array candidate length does not match the declared length"
            raise ValueError(msg)

        mismatch_count = 0.0
        for index in range(self.length):
            left_value = left_tuple[index]
            right_value = right_tuple[index]
            if type(left_value) is not int or type(right_value) is not int:
                msg = "binary-array diversity requires canonical integer candidates"
                raise TypeError(msg)
            if left_value not in {0, 1} or right_value not in {0, 1}:
                msg = "binary-array diversity requires only 0/1 candidate values"
                raise ValueError(msg)
            if left_value != right_value:
                mismatch_count += 1.0
        return StructuredDistanceParts(
            overlap_squared_distance=mismatch_count,
            shared_leaf_count=self.length,
        )


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
        left_tuple = require_candidate_tuple(
            value=left,
            message="integer-array diversity requires canonical tuple candidates",
        )
        right_tuple = require_candidate_tuple(
            value=right,
            message="integer-array diversity requires canonical tuple candidates",
        )
        if len(left_tuple) != self.length or len(right_tuple) != self.length:
            msg = "integer array candidate length does not match the declared length"
            raise ValueError(msg)

        element_space = self.element_space
        if element_space.low == element_space.high:
            for index in range(self.length):
                left_value = left_tuple[index]
                right_value = right_tuple[index]
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
            return StructuredDistanceParts(
                overlap_squared_distance=0.0,
                shared_leaf_count=self.length,
            )

        squared_distance = 0.0
        if element_space.scale == "log":
            coordinate_span = log(float(element_space.high)) - log(float(element_space.low))
            for index in range(self.length):
                left_value = left_tuple[index]
                right_value = right_tuple[index]
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
                leaf_distance = abs(log(float(left_value)) - log(float(right_value))) / coordinate_span
                squared_distance += leaf_distance * leaf_distance
            return StructuredDistanceParts(
                overlap_squared_distance=squared_distance,
                shared_leaf_count=self.length,
            )

        coordinate_span = float(element_space.high - element_space.low)
        for index in range(self.length):
            left_value = left_tuple[index]
            right_value = right_tuple[index]
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
        return StructuredDistanceParts(
            overlap_squared_distance=squared_distance,
            shared_leaf_count=self.length,
        )


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
        left_tuple = require_candidate_tuple(
            value=left,
            message="real-array diversity requires canonical tuple candidates",
        )
        right_tuple = require_candidate_tuple(
            value=right,
            message="real-array diversity requires canonical tuple candidates",
        )
        if len(left_tuple) != self.length or len(right_tuple) != self.length:
            msg = "real array candidate length does not match the declared length"
            raise ValueError(msg)

        element_space = self.element_space
        if element_space.low == element_space.high:
            for index in range(self.length):
                left_value = require_real_candidate(
                    value=left_tuple[index],
                    message="real-array diversity requires numeric left leaf values",
                )
                right_value = require_real_candidate(
                    value=right_tuple[index],
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
            return StructuredDistanceParts(
                overlap_squared_distance=0.0,
                shared_leaf_count=self.length,
            )

        squared_distance = 0.0
        if element_space.scale == "log":
            coordinate_span = log(element_space.high) - log(element_space.low)
            for index in range(self.length):
                left_value = require_real_candidate(
                    value=left_tuple[index],
                    message="real-array diversity requires numeric left leaf values",
                )
                right_value = require_real_candidate(
                    value=right_tuple[index],
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
                leaf_distance = abs(log(left_value) - log(right_value)) / coordinate_span
                squared_distance += leaf_distance * leaf_distance
            return StructuredDistanceParts(
                overlap_squared_distance=squared_distance,
                shared_leaf_count=self.length,
            )

        coordinate_span = element_space.high - element_space.low
        for index in range(self.length):
            left_value = require_real_candidate(
                value=left_tuple[index],
                message="real-array diversity requires numeric left leaf values",
            )
            right_value = require_real_candidate(
                value=right_tuple[index],
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
        return StructuredDistanceParts(
            overlap_squared_distance=squared_distance,
            shared_leaf_count=self.length,
        )


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
