"""Coordinate projection helpers for structured search spaces."""

from .continuous import ContinuousStructuredSpaceCodec
from .numeric import (
    HomogeneousNumericSubspaceDescriptor,
    compile_homogeneous_numeric_subspace,
)

__all__ = [
    "ContinuousStructuredSpaceCodec",
    "HomogeneousNumericSubspaceDescriptor",
    "compile_homogeneous_numeric_subspace",
]
