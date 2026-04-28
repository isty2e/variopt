"""Regression tests for DE family support-module boundaries."""

import importlib

import pytest

from variopt.algorithms.population.de.variation import (
    differential_evolution_variation,
    require_numeric_structured_space,
)


class DEFamilyImportBoundaryTests:
    """Lock the DE family support-module boundary."""

    def test_de_variation_module_is_importable_under_de_family(self) -> None:
        module = importlib.import_module("variopt.algorithms.population.de.variation")

        assert module.__name__ == "variopt.algorithms.population.de.variation"
        assert "differential_evolution_variation" in module.__dict__
        assert "require_numeric_structured_space" in module.__dict__
        assert differential_evolution_variation is not None
        assert require_numeric_structured_space is not None

    def test_old_population_root_de_support_module_is_removed(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            _ = importlib.import_module(
                "variopt.algorithms.population.structured_differential_evolution",
            )
