"""Structured-space geometry contracts and top-level compilation helpers."""

from .compile import (
    compile_structured_geometry,
    distance_parts,
    generic_distance_parts,
)
from .contracts import CompiledStructuredGeometryProvider, StructuredSpaceGeometry
from .parts import StructuredDistanceParts

__all__ = [
    "CompiledStructuredGeometryProvider",
    "StructuredDistanceParts",
    "StructuredSpaceGeometry",
    "compile_structured_geometry",
    "distance_parts",
    "generic_distance_parts",
]
