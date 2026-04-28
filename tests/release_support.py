"""Repo-local support helpers for release-surface regression tests."""

import importlib
from dataclasses import dataclass
from typing import cast

BASE_INSTALL_MODULES: tuple[str, ...] = (
    "variopt",
    "variopt.artifacts",
    "variopt.spaces",
    "variopt.evaluators",
    "variopt.study",
    "variopt.algorithms",
    "variopt.algorithms.population",
    "variopt.algorithms.local_search",
    "variopt.spaces.projections",
)

BASE_INSTALL_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("variopt", "Problem"),
    ("variopt", "Study"),
    ("variopt", "RealSpace"),
    ("variopt.evaluators", "SequentialEvaluator"),
    ("variopt.evaluators", "JoblibEvaluator"),
    ("variopt.evaluators", "AsyncJoblibEvaluator"),
    ("variopt.evaluators", "MpiEvaluator"),
    ("variopt.evaluators", "MpiExecutorFactory"),
    ("variopt.algorithms.population", "CSAOptimizer"),
    ("variopt.algorithms.population", "DifferentialEvolutionOptimizer"),
    ("variopt.algorithms.local_search", "StructuredHillClimbKernel"),
    ("variopt.algorithms.local_search", "ScipyMinimizeKernel"),
)


@dataclass(frozen=True, slots=True)
class ImportFailure:
    """One failed module or symbol import assertion."""

    target: str
    detail: str


def collect_base_import_failures() -> tuple[ImportFailure, ...]:
    """Return any module or symbol failures for the base-install surface."""
    failures: list[ImportFailure] = []

    for module_name in BASE_INSTALL_MODULES:
        try:
            _ = importlib.import_module(module_name)
        except Exception as exception:  # pragma: no cover - exercised in smoke use
            failures.append(
                ImportFailure(
                    target=module_name,
                    detail=f"{type(exception).__name__}: {exception}",
                )
            )

    for module_name, symbol_name in BASE_INSTALL_SYMBOLS:
        try:
            module = importlib.import_module(module_name)
            _ = cast(object, getattr(module, symbol_name))
        except Exception as exception:  # pragma: no cover - exercised in smoke use
            failures.append(
                ImportFailure(
                    target=f"{module_name}.{symbol_name}",
                    detail=f"{type(exception).__name__}: {exception}",
                )
            )

    return tuple(failures)
