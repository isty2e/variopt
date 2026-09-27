"""Reusable encoded geometry plans for exact built-in structured spaces."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite, log
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

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


@dataclass(frozen=True, slots=True)
class _EncodedCandidateBatch:
    """Call-local matrices, each shaped (reference count, coordinate count)."""

    real_values: NDArray[np.float64]
    integer_values: NDArray[np.int64]
    discrete_values: NDArray[np.int64]

    @classmethod
    def from_encodings(
        cls,
        references: Sequence[EncodedStructuredCandidate],
    ) -> "_EncodedCandidateBatch":
        return cls(
            real_values=np.asarray(
                [reference.real_values for reference in references], dtype=np.float64
            )
            if references[0].real_values
            else np.empty((len(references), 0), dtype=np.float64),
            integer_values=np.asarray(
                [reference.integer_values for reference in references], dtype=np.int64
            )
            if references[0].integer_values
            else np.empty((len(references), 0), dtype=np.int64),
            discrete_values=np.asarray(
                [reference.discrete_values for reference in references], dtype=np.int64
            )
            if references[0].discrete_values
            else np.empty((len(references), 0), dtype=np.int64),
        )

    @property
    def reference_count(self) -> int:
        return self.real_values.shape[0]


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

    def squared_distances_to_batch(
        self,
        candidate: EncodedStructuredCandidate,
        references: _EncodedCandidateBatch,
    ) -> NDArray[np.float64]:
        """Return ordered subtotals for one batch of references."""
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

    def squared_distances_to_batch(
        self,
        candidate: EncodedStructuredCandidate,
        references: _EncodedCandidateBatch,
    ) -> NDArray[np.float64]:
        _ = candidate
        return np.zeros(references.reference_count, dtype=np.float64)


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

    def squared_distances_to_batch(
        self,
        candidate: EncodedStructuredCandidate,
        references: _EncodedCandidateBatch,
    ) -> NDArray[np.float64]:
        differences = np.subtract(
            candidate.real_values[self.start : self.stop],
            references.real_values[:, self.start : self.stop],
        )
        return _normalized_squared_subtotals(differences, self.coordinate_span)


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

    def squared_distances_to_batch(
        self,
        candidate: EncodedStructuredCandidate,
        references: _EncodedCandidateBatch,
    ) -> NDArray[np.float64]:
        # Compilation proves that both coordinates and their differences fit int64.
        differences = np.subtract(
            np.asarray(
                candidate.integer_values[self.start : self.stop], dtype=np.int64
            ),
            references.integer_values[:, self.start : self.stop],
        )
        return _normalized_squared_subtotals(
            differences.astype(np.float64), self.coordinate_span
        )


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

    def squared_distances_to_batch(
        self,
        candidate: EncodedStructuredCandidate,
        references: _EncodedCandidateBatch,
    ) -> NDArray[np.float64]:
        mismatches = np.not_equal(
            candidate.discrete_values[self.start : self.stop],
            references.discrete_values[:, self.start : self.stop],
        )
        return _ordered_row_subtotals(mismatches.astype(np.float64))


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

    def squared_distances_to_batch(
        self,
        candidate: EncodedStructuredCandidate,
        references: _EncodedCandidateBatch,
    ) -> NDArray[np.float64]:
        squared_distances = np.zeros(references.reference_count, dtype=np.float64)
        for child_kernel in self.child_kernels:
            squared_distances += child_kernel.squared_distances_to_batch(
                candidate, references
            )
        return squared_distances


def _normalized_squared_subtotals(
    differences: NDArray[np.float64],
    coordinate_span: float,
) -> NDArray[np.float64]:
    np.abs(differences, out=differences)
    np.divide(differences, coordinate_span, out=differences)
    np.multiply(differences, differences, out=differences)
    return _ordered_row_subtotals(differences)


def _ordered_row_subtotals(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=np.float64)
    if values.shape[1] > 1:
        # sum/dot may regroup additions and change cutoff or nearest-neighbor ties.
        np.cumsum(values, axis=1, out=values)
    return values[:, -1]


@dataclass(slots=True)
class _GeometryPlanBuilder:
    real_value_count: int = 0
    integer_value_count: int = 0
    discrete_value_count: int = 0
    leaf_count: int = 0
    supports_array_batches: bool = True

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
    reuses_candidate_structure: bool
    _encoder: _CandidateEncoder = field(repr=False, compare=False)
    _kernel: _DistanceKernel = field(repr=False, compare=False)
    _real_value_count: int = field(repr=False)
    _integer_value_count: int = field(repr=False)
    _discrete_value_count: int = field(repr=False)
    _array_batch_minimum_size: int | None = field(repr=False)

    def prefers_batched_distances(self, reference_count: int) -> bool:
        """Return whether array batches amortize this plan's packing cost.

        Parameters
        ----------
        reference_count : int
            Number of reference encodings in one distance query.

        Returns
        -------
        bool
            Whether this plan supports array batches and the query reaches its
            packing crossover. False keeps scalar evaluation preferable.
        """
        minimum_size = self._array_batch_minimum_size
        return minimum_size is not None and reference_count >= minimum_size

    def encode(self, candidate: CandidateT) -> EncodedStructuredCandidate:
        """Validate and encode one canonical candidate."""
        self.space.validate(candidate)
        return self.encode_validated(candidate)

    def encode_validated(
        self,
        candidate: CandidateT,
    ) -> EncodedStructuredCandidate:
        """Encode a canonical candidate already validated by the caller."""
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

    def encode_many_validated(
        self,
        candidates: Sequence[CandidateT],
    ) -> tuple[EncodedStructuredCandidate, ...]:
        """Encode already validated candidates in input order."""
        return tuple(self.encode_validated(candidate) for candidate in candidates)

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
        return self._squared_distances_to_many(candidate, references)

    def pairwise_squared_distances(
        self,
        candidates: Sequence[EncodedStructuredCandidate],
    ) -> tuple[tuple[float, ...], ...]:
        """Return one symmetric squared-distance matrix."""
        candidate_tuple = tuple(candidates)
        for candidate in candidate_tuple:
            self._validate_encoding_alignment(candidate)

        candidate_count = len(candidate_tuple)
        distances = [
            [0.0 for _right_index in range(candidate_count)]
            for _left_index in range(candidate_count)
        ]
        minimum_batch_size = self._array_batch_minimum_size
        for left_index in range(candidate_count):
            if minimum_batch_size is None or left_index < minimum_batch_size:
                for right_index in range(left_index):
                    distance = self._kernel.squared_distance(
                        candidate_tuple[left_index], candidate_tuple[right_index]
                    )
                    distances[left_index][right_index] = distance
                    distances[right_index][left_index] = distance
                continue

            row_distances = self._squared_distances_to_many(
                candidate_tuple[left_index], candidate_tuple[:left_index]
            )
            for right_index, distance in enumerate(row_distances):
                distances[left_index][right_index] = distance
                distances[right_index][left_index] = distance
        return tuple(tuple(row) for row in distances)

    def _squared_distances_to_many(
        self,
        candidate: EncodedStructuredCandidate,
        references: Sequence[EncodedStructuredCandidate],
    ) -> tuple[float, ...]:
        minimum_batch_size = self._array_batch_minimum_size
        if minimum_batch_size is None or len(references) < minimum_batch_size:
            return tuple(
                self._kernel.squared_distance(candidate, reference)
                for reference in references
            )

        value_count = (
            self._real_value_count
            + self._integer_value_count
            + self._discrete_value_count
        )
        batch_size = max(1, min(256, 8192 // max(1, value_count)))
        reference_tuple = tuple(references)
        distances: list[float] = []
        # Match Python float arithmetic even when a caller enables NumPy warnings.
        with np.errstate(all="ignore"):
            for start in range(0, len(reference_tuple), batch_size):
                batch = _EncodedCandidateBatch.from_encodings(
                    reference_tuple[start : start + batch_size]
                )
                distances.extend(
                    float(distance)
                    for distance in self._kernel.squared_distances_to_batch(
                        candidate, batch
                    )
                )
        return tuple(distances)

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
    value_count = (
        builder.real_value_count
        + builder.integer_value_count
        + builder.discrete_value_count
    )
    # Amortize packing with at least 32 references and 256 coordinate comparisons.
    array_batch_minimum_size = (
        max(32, (256 + value_count - 1) // value_count)
        if builder.supports_array_batches and value_count > 0
        else None
    )
    return BuiltinStructuredGeometryPlan(
        space=candidate_space,
        plan_identity=StructuredGeometryPlanIdentity(),
        leaf_count=builder.leaf_count,
        reuses_candidate_structure=isinstance(
            space,
            (ArraySpace, RecordSpace, TupleSpace),
        ),
        _encoder=compiled_geometry.encoder,
        _kernel=compiled_geometry.kernel,
        _real_value_count=builder.real_value_count,
        _integer_value_count=builder.integer_value_count,
        _discrete_value_count=builder.discrete_value_count,
        _array_batch_minimum_size=array_batch_minimum_size,
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
    logarithmic = space.scale == "log"
    coordinate_span = (
        log(space.high) - log(space.low) if logarithmic else space.high - space.low
    )
    if not isfinite(coordinate_span) or coordinate_span <= 0.0:
        builder.supports_array_batches = False
    return _CompiledCandidateGeometry(
        encoder=_repeated_encoder(
            _RealValueEncoder(logarithmic=logarithmic), count, sequence=sequence
        ),
        kernel=_RealCoordinateDistanceKernel(
            start=start,
            stop=stop,
            coordinate_span=coordinate_span,
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
        coordinate_span = log(float(space.high)) - log(float(space.low))
        if not isfinite(coordinate_span) or coordinate_span <= 0.0:
            builder.supports_array_batches = False
        return _CompiledCandidateGeometry(
            encoder=_repeated_encoder(
                _IntegerValueEncoder(logarithmic=True),
                count,
                sequence=sequence,
            ),
            kernel=_RealCoordinateDistanceKernel(
                start=start,
                stop=stop,
                coordinate_span=coordinate_span,
            ),
        )
    start, stop = builder.reserve_integer_values(count)
    if space.low < -(2**63) or space.high >= 2**63 or space.high - space.low >= 2**63:
        builder.supports_array_batches = False
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
