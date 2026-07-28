"""Structured search-space diversity metrics derived from space semantics."""

import math
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeGuard, TypeVar

from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..distance import require_valid_distance
from ..spaces import StructuredSearchSpace
from ..spaces.geometry.compile import (
    compile_structured_geometry,
    generic_distance_parts,
)
from ..spaces.geometry.composites import (
    DistancePartValuesGeometry,
    ValidatedDistancePartValuesGeometry,
    geometry_has_distance_part_values,
    geometry_has_validated_distance_part_values,
)
from ..spaces.geometry.contracts import StructuredSpaceGeometry
from ..spaces.geometry.plan import (
    BuiltinStructuredGeometryPlan,
    EncodedStructuredCandidate,
    StructuredGeometryPlanIdentity,
    compile_builtin_geometry_plan,
)
from ..spaces.types import SpaceBoundaryValue, SpaceCandidateValue
from .base import DiversityMetric

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)
MetricCandidateT = TypeVar("MetricCandidateT")
PlanCandidateT_contra = TypeVar("PlanCandidateT_contra", contravariant=True)


class CandidateGeometryPlan(Protocol[PlanCandidateT_contra]):
    """Candidate-typed projection of an immutable structured geometry plan.

    The protocol erases the boundary-candidate type, which is irrelevant after
    CSA has admitted canonical candidates, while retaining the candidate type
    used by the encoder.
    """

    @property
    def plan_identity(self) -> StructuredGeometryPlanIdentity:
        """Return the identity shared by every encoding from this plan.

        Returns
        -------
        StructuredGeometryPlanIdentity
            Identity token used to reject encodings from another plan.
        """
        ...

    @property
    def leaf_count(self) -> int:
        """Return the number of normalized leaves in each encoding.

        Returns
        -------
        int
            Positive leaf count used by RMS distance normalization.
        """
        ...

    def encode_validated(
        self,
        candidate: PlanCandidateT_contra,
    ) -> EncodedStructuredCandidate:
        """Encode one canonical candidate already validated by the caller.

        Parameters
        ----------
        candidate : PlanCandidateT_contra
            Canonical candidate admitted through the owning space.

        Returns
        -------
        EncodedStructuredCandidate
            Candidate projection aligned to this plan.
        """
        ...

    def squared_distance(
        self,
        left: EncodedStructuredCandidate,
        right: EncodedStructuredCandidate,
    ) -> float:
        """Return the squared distance between two aligned encodings.

        Parameters
        ----------
        left : EncodedStructuredCandidate
            Left encoding produced by this plan.
        right : EncodedStructuredCandidate
            Right encoding produced by this plan.

        Returns
        -------
        float
            Sum of normalized squared leaf distances.
        """
        ...

    def squared_distances_to_many(
        self,
        candidate: EncodedStructuredCandidate,
        references: Sequence[EncodedStructuredCandidate],
    ) -> tuple[float, ...]:
        """Return squared distances from one encoding to ordered references.

        Parameters
        ----------
        candidate : EncodedStructuredCandidate
            Encoding whose distances are requested.
        references : Sequence[EncodedStructuredCandidate]
            Ordered reference encodings produced by this plan.

        Returns
        -------
        tuple[float, ...]
            Squared distances aligned to ``references``.
        """
        ...


@dataclass(frozen=True, slots=True)
class CompiledStructuredDistanceView(Generic[MetricCandidateT]):
    """Immutable encoded candidates aligned to one operation-local snapshot.

    Parameters
    ----------
    plan : CandidateGeometryPlan[MetricCandidateT]
        Immutable geometry plan that owns all encodings.
    candidates : tuple[MetricCandidateT, ...]
        Canonical candidates in snapshot order.
    encodings : tuple[EncodedStructuredCandidate, ...]
        Plan-owned encodings aligned one-to-one with ``candidates``.

    Notes
    -----
    The view is derived acceleration state. It must not be persisted as
    optimizer truth or shared across metrics with distinct geometry plans.
    """

    plan: CandidateGeometryPlan[MetricCandidateT] = field(repr=False)
    candidates: tuple[MetricCandidateT, ...] = field(repr=False)
    encodings: tuple[EncodedStructuredCandidate, ...] = field(repr=False)

    def __post_init__(self) -> None:
        """Validate candidate, encoding, and plan alignment.

        Raises
        ------
        ValueError
            If candidate and encoding counts differ or an encoding belongs to
            another plan.
        """
        if len(self.candidates) != len(self.encodings):
            msg = "compiled distance candidates and encodings must align"
            raise ValueError(msg)
        if any(
            encoding.plan_identity is not self.plan.plan_identity
            for encoding in self.encodings
        ):
            msg = "compiled distance encodings must belong to the view plan"
            raise ValueError(msg)

    @classmethod
    def from_candidates(
        cls,
        *,
        plan: CandidateGeometryPlan[MetricCandidateT],
        candidates: Sequence[MetricCandidateT],
    ) -> "CompiledStructuredDistanceView[MetricCandidateT]":
        """Encode candidates in snapshot order.

        Parameters
        ----------
        plan : CandidateGeometryPlan[MetricCandidateT]
            Geometry plan used to encode each candidate.
        candidates : Sequence[MetricCandidateT]
            Canonical candidates already validated by the operation boundary.

        Returns
        -------
        CompiledStructuredDistanceView[MetricCandidateT]
            Immutable view aligned to the supplied snapshot.
        """
        candidate_tuple = tuple(candidates)
        return cls(
            plan=plan,
            candidates=candidate_tuple,
            encodings=tuple(
                plan.encode_validated(candidate) for candidate in candidate_tuple
            ),
        )

    def rebase(
        self,
        *,
        candidates: Sequence[MetricCandidateT],
        invalidated_indices: AbstractSet[int],
    ) -> "CompiledStructuredDistanceView[MetricCandidateT]":
        """Realign the view while retaining only provably unchanged encodings.

        Parameters
        ----------
        candidates : Sequence[MetricCandidateT]
            Candidates in the next snapshot order.
        invalidated_indices : collections.abc.Set[int]
            Indices whose candidate semantics may have changed even when object
            identity is unchanged.

        Returns
        -------
        CompiledStructuredDistanceView[MetricCandidateT]
            View aligned to ``candidates``. Encodings are retained only when the
            index is not invalidated and candidate identity is unchanged.
        """
        candidate_tuple = tuple(candidates)
        invalidated_index_set = frozenset(
            index for index in invalidated_indices if index >= 0
        )
        encodings = tuple(
            (
                self.encodings[index]
                if index < len(self.candidates)
                and index not in invalidated_index_set
                and candidate is self.candidates[index]
                else self.plan.encode_validated(candidate)
            )
            for index, candidate in enumerate(candidate_tuple)
        )
        return type(self)(
            plan=self.plan,
            candidates=candidate_tuple,
            encodings=encodings,
        )

    def is_aligned_with(self, candidates: Sequence[MetricCandidateT]) -> bool:
        """Return whether candidates are the exact snapshot represented here.

        Parameters
        ----------
        candidates : Sequence[MetricCandidateT]
            Candidate sequence to compare by length and object identity.

        Returns
        -------
        bool
            Whether every candidate is the object encoded at the same index.
        """
        return len(candidates) == len(self.candidates) and all(
            candidate is self.candidates[index]
            for index, candidate in enumerate(candidates)
        )

    def distance(self, left_index: int, right_index: int) -> float:
        """Return the RMS distance between two encoded snapshot entries.

        Parameters
        ----------
        left_index : int
            Index of the left snapshot candidate.
        right_index : int
            Index of the right snapshot candidate.

        Returns
        -------
        float
            RMS normalized structured distance.

        Raises
        ------
        IndexError
            If either index lies outside the represented snapshot.
        """
        if (
            left_index < 0
            or left_index >= len(self.encodings)
            or right_index < 0
            or right_index >= len(self.encodings)
        ):
            msg = "distance indices must reference compiled snapshot candidates"
            raise IndexError(msg)
        if left_index == right_index:
            return 0.0
        return _distance_from_compiled_squared_distance(
            squared_distance=self.plan.squared_distance(
                self.encodings[left_index],
                self.encodings[right_index],
            ),
            leaf_count=self.plan.leaf_count,
        )

    def distances_to(
        self,
        candidate: MetricCandidateT,
    ) -> tuple[float, ...]:
        """Encode one candidate and return its RMS distance to every entry.

        Parameters
        ----------
        candidate : MetricCandidateT
            Canonical candidate already validated by the operation boundary.

        Returns
        -------
        tuple[float, ...]
            Distances aligned to :attr:`candidates`.
        """
        encoded_candidate = self.plan.encode_validated(candidate)
        return tuple(
            _distance_from_compiled_squared_distance(
                squared_distance=squared_distance,
                leaf_count=self.plan.leaf_count,
            )
            for squared_distance in self.plan.squared_distances_to_many(
                encoded_candidate,
                self.encodings,
            )
        )


class ValidatedStructuredDistanceMetric(Protocol[MetricCandidateT]):
    """Internal structured metric operations available after validation."""

    @property
    def _has_compiled_distance_plan(self) -> bool:
        """Return whether exact built-in compiled geometry is available."""
        ...

    def _compile_distance_view(
        self,
        candidates: Sequence[MetricCandidateT],
    ) -> CompiledStructuredDistanceView[MetricCandidateT] | None:
        """Compile an operation-local encoded candidate view."""
        ...

    def _owns_distance_view(
        self,
        view: CompiledStructuredDistanceView[MetricCandidateT],
    ) -> bool:
        """Return whether a view was compiled by this metric's plan."""
        ...

    def _distance_between_validated_candidates(
        self,
        left: MetricCandidateT,
        right: MetricCandidateT,
    ) -> float:
        """Return distance without repeating candidate-shape validation."""
        ...


@dataclass(frozen=True, slots=True)
class StructuredSpaceDiversityMetric(
    FrozenGenericSlotsCompat,
    DiversityMetric[CandidateT],
    Generic[BoundaryT, CandidateT],
):
    """Leaf-wise normalized diversity metric over one structured search space.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space whose geometry defines the diversity metric.
    """

    space: StructuredSearchSpace[BoundaryT, CandidateT]
    geometry: StructuredSpaceGeometry | None = field(init=False, repr=False)
    part_values_geometry: DistancePartValuesGeometry | None = field(
        init=False,
        repr=False,
    )
    validated_part_values_geometry: ValidatedDistancePartValuesGeometry | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    compiled_geometry_plan: (
        BuiltinStructuredGeometryPlan[BoundaryT, CandidateT] | None
    ) = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Compile and cache any built-in structured geometry once."""
        geometry = compile_structured_geometry(self.space)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(
            self,
            "part_values_geometry",
            (
                geometry
                if geometry is not None and geometry_has_distance_part_values(geometry)
                else None
            ),
        )
        object.__setattr__(
            self,
            "validated_part_values_geometry",
            (
                geometry
                if geometry is not None
                and geometry_has_validated_distance_part_values(geometry)
                else None
            ),
        )
        object.__setattr__(
            self,
            "compiled_geometry_plan",
            compile_builtin_geometry_plan(self.space),
        )

    @property
    def _has_compiled_distance_plan(self) -> bool:
        """Return whether exact built-in compiled geometry is available."""
        return self.compiled_geometry_plan is not None

    def _compile_distance_view(
        self,
        candidates: Sequence[CandidateT],
    ) -> CompiledStructuredDistanceView[CandidateT] | None:
        """Compile an operation-local encoded view of canonical candidates."""
        plan = self.compiled_geometry_plan
        if plan is None:
            return None
        return CompiledStructuredDistanceView.from_candidates(
            plan=plan,
            candidates=candidates,
        )

    def _owns_distance_view(
        self,
        view: CompiledStructuredDistanceView[CandidateT],
    ) -> bool:
        """Return whether a view was compiled by this metric's plan."""
        plan = self.compiled_geometry_plan
        return plan is not None and view.plan.plan_identity is plan.plan_identity

    def _distance_between_validated_candidates(
        self,
        left: CandidateT,
        right: CandidateT,
    ) -> float:
        """Return structured distance for candidates validated by this space."""
        validated_part_values_geometry = self.validated_part_values_geometry
        if validated_part_values_geometry is not None:
            (
                overlap_squared_distance,
                shared_leaf_count,
                topology_mismatch_leaf_count,
            ) = validated_part_values_geometry.distance_part_values_for_validated_candidates(
                left,
                right,
            )
            return _distance_from_part_values(
                overlap_squared_distance=overlap_squared_distance,
                shared_leaf_count=shared_leaf_count,
                topology_mismatch_leaf_count=topology_mismatch_leaf_count,
            )

        return self.distance(left, right)

    @override
    def distance(self, left: CandidateT, right: CandidateT) -> float:
        """Return the RMS normalized leaf distance between two candidates.

        Parameters
        ----------
        left : CandidateT
            Left canonical candidate.
        right : CandidateT
            Right canonical candidate.

        Returns
        -------
        float
            RMS normalized structured distance.
        """
        part_values_geometry = self.part_values_geometry
        if part_values_geometry is not None:
            (
                overlap_squared_distance,
                shared_leaf_count,
                topology_mismatch_leaf_count,
            ) = part_values_geometry.distance_part_values(left, right)
            return _distance_from_part_values(
                overlap_squared_distance=overlap_squared_distance,
                shared_leaf_count=shared_leaf_count,
                topology_mismatch_leaf_count=topology_mismatch_leaf_count,
            )
        geometry = self.geometry
        if geometry is None:
            parts = generic_distance_parts(
                self.space,
                left,
                right,
            )
            return _distance_from_part_values(
                overlap_squared_distance=parts.overlap_squared_distance,
                shared_leaf_count=parts.shared_leaf_count,
                topology_mismatch_leaf_count=parts.topology_mismatch_leaf_count,
            )
        parts = geometry.distance_parts(left, right)
        return _distance_from_part_values(
            overlap_squared_distance=parts.overlap_squared_distance,
            shared_leaf_count=parts.shared_leaf_count,
            topology_mismatch_leaf_count=parts.topology_mismatch_leaf_count,
        )


def supports_validated_structured_distance(
    metric: DiversityMetric[MetricCandidateT],
) -> TypeGuard[StructuredSpaceDiversityMetric[SpaceBoundaryValue, SpaceCandidateValue]]:
    """Return whether ``metric`` exposes the exact built-in structured contract."""
    return type(metric) is StructuredSpaceDiversityMetric


def supports_candidate_typed_structured_distance(
    metric: DiversityMetric[MetricCandidateT],
) -> TypeGuard[ValidatedStructuredDistanceMetric[MetricCandidateT]]:
    """Return whether ``metric`` exposes candidate-typed internal operations.

    Unlike :func:`supports_validated_structured_distance`, this internal guard
    preserves the caller's otherwise unconstrained candidate type instead of
    exposing the structured space declaration.
    """
    return type(metric) is StructuredSpaceDiversityMetric


def supports_compiled_structured_distance(
    metric: DiversityMetric[MetricCandidateT],
) -> bool:
    """Return whether ``metric`` exposes exact compiled candidate geometry."""
    return (
        supports_candidate_typed_structured_distance(metric)
        and metric._has_compiled_distance_plan
    )


def structured_distance_between_validated_candidates(
    metric: ValidatedStructuredDistanceMetric[MetricCandidateT],
    left: MetricCandidateT,
    right: MetricCandidateT,
) -> float:
    """Return structured distance for candidates validated by ``metric.space``.

    This internal algebra is intentionally not part of the facade-level
    diversity contract. Callers must own evidence that both candidates have
    already crossed the matching space validation boundary.
    """
    return metric._distance_between_validated_candidates(left, right)


def _distance_from_compiled_squared_distance(
    *,
    squared_distance: float,
    leaf_count: int,
) -> float:
    return _distance_from_part_values(
        overlap_squared_distance=squared_distance,
        shared_leaf_count=leaf_count,
        topology_mismatch_leaf_count=0,
    )


def _distance_from_part_values(
    *,
    overlap_squared_distance: float,
    shared_leaf_count: int,
    topology_mismatch_leaf_count: int,
) -> float:
    """Return the RMS structured distance from raw distance-part values."""
    total_leaf_count = shared_leaf_count + topology_mismatch_leaf_count
    if total_leaf_count == 0:
        msg = "structured diversity metric requires at least one leaf path"
        raise ValueError(msg)
    return require_valid_distance(
        math.sqrt(
            require_valid_distance(
                overlap_squared_distance + float(topology_mismatch_leaf_count)
            )
            / total_leaf_count,
        ),
    )
