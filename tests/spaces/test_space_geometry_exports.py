"""Tests for structured geometry package facade exports."""

import variopt.spaces.geometry as geometry
from variopt.spaces import (
    CompiledStructuredGeometryProvider as SpacesCompiledStructuredGeometryProvider,
)
from variopt.spaces import StructuredDistanceParts as SpacesStructuredDistanceParts
from variopt.spaces import StructuredSpaceGeometry as SpacesStructuredSpaceGeometry
from variopt.spaces.geometry import (
    CompiledStructuredGeometryProvider,
    StructuredDistanceParts,
    StructuredSpaceGeometry,
    compile_structured_geometry,
    distance_parts,
    generic_distance_parts,
)
from variopt.spaces.geometry.compile import (
    compile_structured_geometry as compile_structured_geometry_submodule,
)
from variopt.spaces.geometry.compile import distance_parts as distance_parts_submodule
from variopt.spaces.geometry.compile import (
    generic_distance_parts as generic_distance_parts_submodule,
)
from variopt.spaces.geometry.contracts import (
    CompiledStructuredGeometryProvider as CompiledStructuredGeometryProviderSubmodule,
)
from variopt.spaces.geometry.contracts import (
    StructuredSpaceGeometry as StructuredSpaceGeometrySubmodule,
)
from variopt.spaces.geometry.parts import (
    StructuredDistanceParts as StructuredDistancePartsSubmodule,
)


class SpaceGeometryExportTests:
    """Regression tests for structured geometry facade identity."""

    def test_geometry_facade_re_exports_contract_symbols(self) -> None:
        assert StructuredSpaceGeometry is StructuredSpaceGeometrySubmodule
        assert (
            CompiledStructuredGeometryProvider
            is CompiledStructuredGeometryProviderSubmodule
        )
        assert StructuredDistanceParts is StructuredDistancePartsSubmodule

    def test_spaces_facade_re_exports_public_geometry_contracts(self) -> None:
        assert SpacesStructuredSpaceGeometry is StructuredSpaceGeometrySubmodule
        assert (
            SpacesCompiledStructuredGeometryProvider
            is CompiledStructuredGeometryProviderSubmodule
        )
        assert SpacesStructuredDistanceParts is StructuredDistancePartsSubmodule

    def test_geometry_facade_re_exports_compile_helpers(self) -> None:
        assert compile_structured_geometry is compile_structured_geometry_submodule
        assert distance_parts is distance_parts_submodule
        assert generic_distance_parts is generic_distance_parts_submodule

    def test_geometry_facade_omits_internal_detail_symbols(self) -> None:
        assert not (hasattr(geometry, "ArraySpaceGeometry"))
        assert not (hasattr(geometry, "collect_child_geometries"))
        assert not (hasattr(geometry, "compile_builtin_structured_geometry"))
        assert not (hasattr(geometry, "is_builtin_child_space"))
        assert not (hasattr(geometry, "BuiltinGeometrySpace"))
        assert not (hasattr(geometry, "normalized_squared_leaf_distance"))
        assert not (hasattr(geometry, "require_candidate_tuple"))
        assert not (hasattr(geometry, "require_geometry_candidate_tuple"))
        assert not (hasattr(geometry, "validate_categorical_choice"))
