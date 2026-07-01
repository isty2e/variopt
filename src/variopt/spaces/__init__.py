"""Search-space abstractions and built-in space implementations."""

from .base import SearchSpace
from .composites import ArraySpace, RecordCandidate, RecordSpace, TupleSpace
from .equality import CandidateEquality
from .geometry import (
    CompiledStructuredGeometryProvider,
    StructuredDistanceParts,
    StructuredSpaceGeometry,
)
from .permutation import PermutationSpace
from .projections import (
    ContinuousStructuredSpaceCodec,
    HomogeneousNumericSubspaceDescriptor,
    compile_homogeneous_numeric_subspace,
)
from .scalar import CategoricalSpace, IntegerSpace, RealSpace
from .structured import LeafPath, StructuredLeafSpace, StructuredSearchSpace
from .types import SpaceBoundaryValue, SpaceCandidateValue, SpaceScalarValue

__all__ = [
    "ArraySpace",
    "CategoricalSpace",
    "CandidateEquality",
    "CompiledStructuredGeometryProvider",
    "ContinuousStructuredSpaceCodec",
    "HomogeneousNumericSubspaceDescriptor",
    "IntegerSpace",
    "LeafPath",
    "PermutationSpace",
    "RealSpace",
    "RecordCandidate",
    "RecordSpace",
    "SearchSpace",
    "SpaceBoundaryValue",
    "SpaceCandidateValue",
    "SpaceScalarValue",
    "StructuredDistanceParts",
    "StructuredLeafSpace",
    "StructuredSpaceGeometry",
    "StructuredSearchSpace",
    "TupleSpace",
    "compile_homogeneous_numeric_subspace",
]
