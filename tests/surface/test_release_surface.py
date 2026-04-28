"""Regression coverage for release-facing public-surface boundaries."""

import importlib

import pytest

from variopt.generic_runtime import FrozenGenericSlotsCompat


class ReleaseSurfaceBoundaryTests:
    """Lock the 0.1.0 release-facing public-surface boundaries."""

    def test_benchmark_package_is_not_release_surface(self) -> None:
        removed_modules = (
            "variopt.benchmarks",
            "variopt.benchmarks.parity",
            "variopt.benchmarks.external",
        )

        for module_name in removed_modules:
            with pytest.raises(ModuleNotFoundError):
                _ = importlib.import_module(module_name)

    def test_generic_runtime_uses_public_import_path(self) -> None:
        assert FrozenGenericSlotsCompat.__module__ == "variopt.generic_runtime"
        with pytest.raises(ModuleNotFoundError):
            _ = importlib.import_module("variopt._generic_runtime")

    def test_taxonomy_refactor_removes_old_module_paths(self) -> None:
        removed_modules = (
            "variopt.algorithms.continuous",
            "variopt.algorithms.structured",
            "variopt.evaluation",
            "variopt.proposal_evaluation",
        )

        for module_name in removed_modules:
            with pytest.raises(ModuleNotFoundError):
                _ = importlib.import_module(module_name)

    def test_removed_flat_external_modules_are_not_importable(self) -> None:
        removed_modules = (
            "variopt.benchmarks.external.coco",
            "variopt.benchmarks.external.ioh",
            "variopt.benchmarks.external.cmaes",
            "variopt.benchmarks.external.basinhopping",
            "variopt.benchmarks.external.dual_annealing",
            "variopt.benchmarks.external.pymoo",
        )

        for module_name in removed_modules:
            with pytest.raises(ModuleNotFoundError):
                _ = importlib.import_module(module_name)
