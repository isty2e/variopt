"""Reusable encoded geometry plans for exact built-in structured spaces."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import log
from typing import Generic, Protocol, TypeVar

from ..composites import CompositeChildSpace, RecordCandidate
from ..composites.array_space import ArraySpace
from ..composites.record_space import RecordSpace
from ..composites.tuple_space import TupleSpace
from ..permutation import PermutationSpace
from ..scalar import CategoricalSpace, IntegerSpace, RealSpace
from ..structured import StructuredSearchSpace
from ..types import SpaceBoundaryValue, SpaceCandidateValue, SpaceScalarValue
from .parts import StructuredDistanceParts
from .scalar import CategoricalChoiceKey, categorical_choice_key
from .taxonomy import BuiltinGeometrySpace, is_exact_builtin_geometry_space

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)


@dataclass(frozen=True, slots=True, eq=False)
class StructuredGeometryPlanIdentity:
    """Identity token that prevents mixing encodings from different plans."""


@dataclass(frozen=True, slots=True)
class EncodedStructuredCandidate:
    """Derived numeric encoding of one validated structured candidate.

    Notes
    -----
    Encodings are non-authoritative runtime projections. They remain aligned to
    the plan that created them and are not candidate or checkpoint formats.
    """

    plan_identity: StructuredGeometryPlanIdentity = field(repr=False)
    real_values: tuple[float, ...]
    integer_values: tuple[int, ...]
    discrete_values: tuple[int, ...]


@dataclass(slots=True)
class _EncodingBuilder:
    real_values: list[float] = field(default_factory=list)
    integer_values: list[int] = field(default_factory=list)
    discrete_values: list[int] = field(default_factory=list)


class _CandidateEncoder(Protocol):
    def encode(
        self,
        candidate: SpaceCandidateValue,
        builder: _EncodingBuilder,
    ) -> None:
        """Append one validated candidate projection to ``builder``."""
        ...


class _DistanceKernel(Protocol):
    def squared_distance(
        self,
        left: EncodedStructuredCandidate,
        right: EncodedStructuredCandidate,
    ) -> float:
        """Return one overlap squared-distance subtotal."""
        ...


@dataclass(frozen=True, slots=True)
class _CompiledCandidateGeometry:
    encoder: _CandidateEncoder
    kernel: _DistanceKernel


@dataclass(frozen=True, slots=True)
class _NoValueEncoder:
    def encode(
        self,
        candidate: SpaceCandidateValue,
        builder: _EncodingBuilder,
    ) -> None:
        _ = candidate, builder


@dataclass(frozen=True, slots=True)
class _RealValueEncoder:
    logarithmic: bool = False

    def encode(
        self,
        candidate: SpaceCandidateValue,
        builder: _EncodingBuilder,
    ) -> None:
        if type(candidate) is not float:
            msg = "validated real candidate must be a canonical float"
            raise TypeError(msg)
        builder.real_values.append(log(candidate) if self.logarithmic else candidate)


@dataclass(frozen=True, slots=True)
class _IntegerValueEncoder:
    logarithmic: bool = False
    discrete: bool = False

    def encode(
        self,
        candidate: SpaceCandidateValue,
        builder: _EncodingBuilder,
    ) -> None:
        if type(candidate) is not int:
            msg = "validated integer candidate must be a canonical integer"
            raise TypeError(msg)
        if self.discrete:
            builder.discrete_values.append(candidate)
        elif self.logarithmic:
            builder.real_values.append(log(float(candidate)))
        else:
            builder.integer_values.append(candidate)


@dataclass(frozen=True, slots=True)
class _CategoricalValueEncoder:
    choice_keys: tuple[CategoricalChoiceKey, ...]

    def encode(
        self,
        candidate: SpaceCandidateValue,
        builder: _EncodingBuilder,
    ) -> None:
        candidate_key = categorical_choice_key(candidate)
        if candidate_key is None:
            msg = "validated categorical candidate must have an exact scalar key"
            raise TypeError(msg)
        try:
            choice_index = self.choice_keys.index(candidate_key)
        except ValueError as exception:
            msg = "validated categorical candidate is not in the geometry plan"
            raise RuntimeError(msg) from exception
        builder.discrete_values.append(choice_index)


@dataclass(frozen=True, slots=True)
class _SequenceCandidateEncoder:
    child_encoders: tuple[_CandidateEncoder, ...]

    def encode(
        self,
        candidate: SpaceCandidateValue,
        builder: _EncodingBuilder,
    ) -> None:
        if type(candidate) is not tuple:
            msg = "validated sequence candidate must be a canonical tuple"
            raise TypeError(msg)
        if len(candidate) != len(self.child_encoders):
            msg = "validated sequence candidate has an unexpected arity"
            raise ValueError(msg)
        for index, child_encoder in enumerate(self.child_encoders):
            child_encoder.encode(candidate[index], builder)


@dataclass(frozen=True, slots=True)
class _RecordCandidateEncoder:
    field_encoders: tuple[_CandidateEncoder, ...]

    def encode(
        self,
        candidate: SpaceCandidateValue,
        builder: _EncodingBuilder,
    ) -> None:
        if not isinstance(candidate, RecordCandidate):
            msg = "validated record candidate must use RecordCandidate"
            raise TypeError(msg)
        entries = candidate.entries
        if len(entries) != len(self.field_encoders):
            msg = "validated record candidate has an unexpected field count"
            raise ValueError(msg)
        for index, field_encoder in enumerate(self.field_encoders):
            field_encoder.encode(entries[index][1], builder)


@dataclass(frozen=True, slots=True)
class _ZeroDistanceKernel:
    def squared_distance(
        self,
        left: EncodedStructuredCandidate,
        right: EncodedStructuredCandidate,
    ) -> float:
        _ = left, right
        return 0.0


@dataclass(frozen=True, slots=True)
class _RealCoordinateDistanceKernel:
    start: int
    stop: int
    coordinate_span: float

    def squared_distance(
        self,
        left: EncodedStructuredCandidate,
        right: EncodedStructuredCandidate,
    ) -> float:
        squared_distance = 0.0
        for index in range(self.start, self.stop):
            leaf_distance = (
                abs(left.real_values[index] - right.real_values[index])
                / self.coordinate_span
            )
            squared_distance += leaf_distance * leaf_distance
        return squared_distance


@dataclass(frozen=True, slots=True)
class _LinearIntegerDistanceKernel:
    start: int
    stop: int
    coordinate_span: float

    def squared_distance(
        self,
        left: EncodedStructuredCandidate,
        right: EncodedStructuredCandidate,
    ) -> float:
        squared_distance = 0.0
        for index in range(self.start, self.stop):
            leaf_distance = (
                abs(float(left.integer_values[index] - right.integer_values[index]))
                / self.coordinate_span
            )
            squared_distance += leaf_distance * leaf_distance
        return squared_distance


@dataclass(frozen=True, slots=True)
class _MismatchDistanceKernel:
    start: int
    stop: int

    def squared_distance(
        self,
        left: EncodedStructuredCandidate,
        right: EncodedStructuredCandidate,
    ) -> float:
        mismatch_count = 0.0
        for index in range(self.start, self.stop):
            if left.discrete_values[index] != right.discrete_values[index]:
                mismatch_count += 1.0
        return mismatch_count


@dataclass(frozen=True, slots=True)
class _CompositeDistanceKernel:
    child_kernels: tuple[_DistanceKernel, ...]

    def squared_distance(
        self,
        left: EncodedStructuredCandidate,
        right: EncodedStructuredCandidate,
    ) -> float:
        squared_distance = 0.0
        for child_kernel in self.child_kernels:
            squared_distance += child_kernel.squared_distance(left, right)
        return squared_distance


@dataclass(slots=True)
class _GeometryPlanBuilder:
    real_value_count: int = 0
    integer_value_count: int = 0
    discrete_value_count: int = 0
    leaf_count: int = 0

    def reserve_real_values(self, count: int) -> tuple[int, int]:
        start = self.real_value_count
        self.real_value_count += count
        self.leaf_count += count
        return start, start + count

    def reserve_integer_values(self, count: int) -> tuple[int, int]:
        start = self.integer_value_count
        self.integer_value_count += count
        self.leaf_count += count
        return start, start + count

    def reserve_discrete_values(self, count: int) -> tuple[int, int]:
        start = self.discrete_value_count
        self.discrete_value_count += count
        self.leaf_count += count
        return start, start + count

    def reserve_zero_values(self, count: int) -> None:
        self.leaf_count += count


@dataclass(frozen=True, slots=True)
class BuiltinStructuredGeometryPlan(
    Generic[BoundaryT, CandidateT],
):
    """Immutable encoded-distance plan for one exact built-in space."""

    space: StructuredSearchSpace[BoundaryT, CandidateT] = field(repr=False)
    plan_identity: StructuredGeometryPlanIdentity = field(repr=False)
    leaf_count: int
    _encoder: _CandidateEncoder = field(repr=False, compare=False)
    _kernel: _DistanceKernel = field(repr=False, compare=False)
    _real_value_count: int = field(repr=False)
    _integer_value_count: int = field(repr=False)
    _discrete_value_count: int = field(repr=False)

    def encode(self, candidate: CandidateT) -> EncodedStructuredCandidate:
        """Validate and encode one canonical candidate."""
        self.space.validate(candidate)
        builder = _EncodingBuilder()
        self._encoder.encode(candidate, builder)
        if (
            len(builder.real_values) != self._real_value_count
            or len(builder.integer_values) != self._integer_value_count
            or len(builder.discrete_values) != self._discrete_value_count
        ):
            msg = "compiled geometry encoder produced a misaligned projection"
            raise RuntimeError(msg)
        return EncodedStructuredCandidate(
            plan_identity=self.plan_identity,
            real_values=tuple(builder.real_values),
            integer_values=tuple(builder.integer_values),
            discrete_values=tuple(builder.discrete_values),
        )

    def encode_many(
        self,
        candidates: Sequence[CandidateT],
    ) -> tuple[EncodedStructuredCandidate, ...]:
        """Validate and encode candidates in input order."""
        return tuple(self.encode(candidate) for candidate in candidates)

    def distance_parts(
        self,
        left: CandidateT,
        right: CandidateT,
    ) -> StructuredDistanceParts:
        """Return distance parts after validating and encoding two candidates."""
        return StructuredDistanceParts(
            overlap_squared_distance=self.squared_distance(
                self.encode(left),
                self.encode(right),
            ),
            shared_leaf_count=self.leaf_count,
        )

    def squared_distance(
        self,
        left: EncodedStructuredCandidate,
        right: EncodedStructuredCandidate,
    ) -> float:
        """Return overlap squared distance between two aligned encodings."""
        self._validate_encoding_alignment(left)
        self._validate_encoding_alignment(right)
        return self._kernel.squared_distance(left, right)

    def squared_distances_to_many(
        self,
        candidate: EncodedStructuredCandidate,
        references: Sequence[EncodedStructuredCandidate],
    ) -> tuple[float, ...]:
        """Return squared distances from one encoding to ordered references."""
        self._validate_encoding_alignment(candidate)
        for reference in references:
            self._validate_encoding_alignment(reference)
        return tuple(
            self._kernel.squared_distance(candidate, reference)
            for reference in references
        )

    def pairwise_squared_distances(
        self,
        candidates: Sequence[EncodedStructuredCandidate],
    ) -> tuple[tuple[float, ...], ...]:
        """Return one symmetric squared-distance matrix."""
        for candidate in candidates:
            self._validate_encoding_alignment(candidate)

        candidate_count = len(candidates)
        distances = [
            [0.0 for _right_index in range(candidate_count)]
            for _left_index in range(candidate_count)
        ]
        for left_index in range(candidate_count):
            for right_index in range(left_index):
                distance = self._kernel.squared_distance(
                    candidates[left_index],
                    candidates[right_index],
                )
                distances[left_index][right_index] = distance
                distances[right_index][left_index] = distance
        return tuple(tuple(row) for row in distances)

    def _validate_encoding_alignment(
        self,
        candidate: EncodedStructuredCandidate,
    ) -> None:
        if candidate.plan_identity is not self.plan_identity:
            msg = "encoded candidate belongs to a different geometry plan"
            raise ValueError(msg)


def compile_builtin_geometry_plan(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
) -> BuiltinStructuredGeometryPlan[BoundaryT, CandidateT] | None:
    """Compile an encoded geometry plan for one exact built-in space.

    Subclasses, custom spaces, and unsupported nested children return ``None``
    so callers retain the existing generic diversity contract.
    """
    candidate_space = space
    if not is_exact_builtin_geometry_space(space):
        return None

    builder = _GeometryPlanBuilder()
    compiled_geometry = _compile_candidate_geometry(space, builder)
    if compiled_geometry is None or builder.leaf_count == 0:
        return None
    return BuiltinStructuredGeometryPlan(
        space=candidate_space,
        plan_identity=StructuredGeometryPlanIdentity(),
        leaf_count=builder.leaf_count,
        _encoder=compiled_geometry.encoder,
        _kernel=compiled_geometry.kernel,
        _real_value_count=builder.real_value_count,
        _integer_value_count=builder.integer_value_count,
        _discrete_value_count=builder.discrete_value_count,
    )


def _compile_candidate_geometry(
    space: BuiltinGeometrySpace,
    builder: _GeometryPlanBuilder,
) -> _CompiledCandidateGeometry | None:
    if isinstance(space, RealSpace):
        return _compile_real_geometry(space, builder)
    if isinstance(space, IntegerSpace):
        return _compile_integer_geometry(space, builder)
    if isinstance(space, CategoricalSpace):
        return _compile_categorical_geometry(space, builder)
    if isinstance(space, PermutationSpace):
        start, stop = builder.reserve_discrete_values(space.size)
        return _CompiledCandidateGeometry(
            encoder=_SequenceCandidateEncoder(
                tuple(
                    _IntegerValueEncoder(discrete=True) for _index in range(space.size)
                )
            ),
            kernel=_MismatchDistanceKernel(start=start, stop=stop),
        )
    if isinstance(space, TupleSpace):
        child_geometries = _compile_child_geometries(space.child_spaces, builder)
        if child_geometries is None:
            return None
        return _CompiledCandidateGeometry(
            encoder=_SequenceCandidateEncoder(
                tuple(child.encoder for child in child_geometries)
            ),
            kernel=_CompositeDistanceKernel(
                tuple(child.kernel for child in child_geometries)
            ),
        )
    if isinstance(space, RecordSpace):
        child_geometries = _compile_child_geometries(
            tuple(child_space for _field_name, child_space in space.fields),
            builder,
        )
        if child_geometries is None:
            return None
        return _CompiledCandidateGeometry(
            encoder=_RecordCandidateEncoder(
                tuple(child.encoder for child in child_geometries)
            ),
            kernel=_CompositeDistanceKernel(
                tuple(child.kernel for child in child_geometries)
            ),
        )
    return _compile_array_geometry(space, builder)


def _compile_real_geometry(
    space: RealSpace,
    builder: _GeometryPlanBuilder,
    *,
    count: int = 1,
    sequence: bool = False,
) -> _CompiledCandidateGeometry:
    encoder = _repeated_encoder(_RealValueEncoder(), count, sequence=sequence)
    if space.low == space.high:
        builder.reserve_zero_values(count)
        return _CompiledCandidateGeometry(
            encoder=_repeated_encoder(
                _NoValueEncoder(),
                count,
                sequence=sequence,
            ),
            kernel=_ZeroDistanceKernel(),
        )
    start, stop = builder.reserve_real_values(count)
    if space.scale == "log":
        return _CompiledCandidateGeometry(
            encoder=_repeated_encoder(
                _RealValueEncoder(logarithmic=True),
                count,
                sequence=sequence,
            ),
            kernel=_RealCoordinateDistanceKernel(
                start=start,
                stop=stop,
                coordinate_span=log(space.high) - log(space.low),
            ),
        )
    return _CompiledCandidateGeometry(
        encoder=encoder,
        kernel=_RealCoordinateDistanceKernel(
            start=start,
            stop=stop,
            coordinate_span=space.high - space.low,
        ),
    )


def _compile_integer_geometry(
    space: IntegerSpace,
    builder: _GeometryPlanBuilder,
    *,
    count: int = 1,
    sequence: bool = False,
) -> _CompiledCandidateGeometry:
    if space.low == space.high:
        builder.reserve_zero_values(count)
        return _CompiledCandidateGeometry(
            encoder=_repeated_encoder(
                _NoValueEncoder(),
                count,
                sequence=sequence,
            ),
            kernel=_ZeroDistanceKernel(),
        )
    if space.low == 0 and space.high == 1 and space.scale == "linear":
        start, stop = builder.reserve_discrete_values(count)
        return _CompiledCandidateGeometry(
            encoder=_repeated_encoder(
                _IntegerValueEncoder(discrete=True),
                count,
                sequence=sequence,
            ),
            kernel=_MismatchDistanceKernel(start=start, stop=stop),
        )
    if space.scale == "log":
        start, stop = builder.reserve_real_values(count)
        return _CompiledCandidateGeometry(
            encoder=_repeated_encoder(
                _IntegerValueEncoder(logarithmic=True),
                count,
                sequence=sequence,
            ),
            kernel=_RealCoordinateDistanceKernel(
                start=start,
                stop=stop,
                coordinate_span=log(float(space.high)) - log(float(space.low)),
            ),
        )
    start, stop = builder.reserve_integer_values(count)
    return _CompiledCandidateGeometry(
        encoder=_repeated_encoder(
            _IntegerValueEncoder(),
            count,
            sequence=sequence,
        ),
        kernel=_LinearIntegerDistanceKernel(
            start=start,
            stop=stop,
            coordinate_span=float(space.high - space.low),
        ),
    )


def _compile_categorical_geometry(
    space: CategoricalSpace[SpaceScalarValue],
    builder: _GeometryPlanBuilder,
    *,
    count: int = 1,
    sequence: bool = False,
) -> _CompiledCandidateGeometry | None:
    choice_keys: list[CategoricalChoiceKey] = []
    for choice in space.choices:
        choice_key = categorical_choice_key(choice)
        if choice_key is None:
            return None
        choice_keys.append(choice_key)
    if len(choice_keys) == 1:
        builder.reserve_zero_values(count)
        return _CompiledCandidateGeometry(
            encoder=_repeated_encoder(
                _NoValueEncoder(),
                count,
                sequence=sequence,
            ),
            kernel=_ZeroDistanceKernel(),
        )
    start, stop = builder.reserve_discrete_values(count)
    return _CompiledCandidateGeometry(
        encoder=_repeated_encoder(
            _CategoricalValueEncoder(tuple(choice_keys)),
            count,
            sequence=sequence,
        ),
        kernel=_MismatchDistanceKernel(start=start, stop=stop),
    )


def _compile_array_geometry(
    space: ArraySpace[SpaceBoundaryValue, SpaceCandidateValue],
    builder: _GeometryPlanBuilder,
) -> _CompiledCandidateGeometry | None:
    element_space = space.element_space
    if not is_exact_builtin_geometry_space(element_space):
        return None
    if isinstance(element_space, RealSpace):
        child_geometry = _compile_real_geometry(
            element_space,
            builder,
            count=space.length,
            sequence=True,
        )
        return child_geometry
    if isinstance(element_space, IntegerSpace):
        child_geometry = _compile_integer_geometry(
            element_space,
            builder,
            count=space.length,
            sequence=True,
        )
        return child_geometry
    if isinstance(element_space, CategoricalSpace):
        child_geometry = _compile_categorical_geometry(
            element_space,
            builder,
            count=space.length,
            sequence=True,
        )
        return child_geometry

    child_geometries = _compile_child_geometries(
        tuple(element_space for _index in range(space.length)),
        builder,
    )
    if child_geometries is None:
        return None
    return _CompiledCandidateGeometry(
        encoder=_SequenceCandidateEncoder(
            tuple(child.encoder for child in child_geometries)
        ),
        kernel=_CompositeDistanceKernel(
            tuple(child.kernel for child in child_geometries)
        ),
    )


def _compile_child_geometries(
    child_spaces: Sequence[CompositeChildSpace],
    builder: _GeometryPlanBuilder,
) -> tuple[_CompiledCandidateGeometry, ...] | None:
    child_geometries: list[_CompiledCandidateGeometry] = []
    for child_space in child_spaces:
        if not is_exact_builtin_geometry_space(child_space):
            return None
        child_geometry = _compile_candidate_geometry(child_space, builder)
        if child_geometry is None:
            return None
        child_geometries.append(child_geometry)
    return tuple(child_geometries)


def _repeated_encoder(
    encoder: _CandidateEncoder,
    count: int,
    *,
    sequence: bool,
) -> _CandidateEncoder:
    if not sequence:
        return encoder
    return _SequenceCandidateEncoder(tuple(encoder for _index in range(count)))
