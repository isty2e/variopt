"""Regression tests for algorithm-family import isolation."""

import subprocess
import sys

import pytest

BACKEND_FREE_ALGORITHM_IMPORTS = (
    "import variopt.algorithms",
    "import variopt.algorithms.population",
    "from variopt.algorithms.population.ga import GeneticAlgorithmOptimizer",
    "from variopt.algorithms.population import CSAOptimizer",
    "from variopt.algorithms import ScipyMinimizeKernel",
)


@pytest.mark.parametrize("import_statement", BACKEND_FREE_ALGORITHM_IMPORTS)
def test_algorithm_import_does_not_load_scipy(
    import_statement: str,
) -> None:
    script = f"""
import sys

{import_statement}

sys.stdout.write("\\n".join(sorted(
    name
    for name in sys.modules
    if name == "scipy" or name.startswith("scipy.")
)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""
