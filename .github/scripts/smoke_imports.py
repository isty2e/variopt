"""Installed-package smoke checks for CI-built variopt wheels."""

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from sys import argv
from types import ModuleType
from typing import Literal

SmokeSurface = Literal["base", "joblib-private", "mpi"]

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

JOBLIB_LOKY_EXCEPTION_SYMBOLS: tuple[str, ...] = (
    "BrokenProcessPool",
    "TerminatedWorkerError",
)

MPI_INSTALL_MODULES: tuple[str, ...] = (
    "mpi4py",
    "mpi4py.futures",
)


@dataclass(frozen=True, slots=True)
class SmokeFailure:
    """One installed-world smoke failure."""

    target: str
    detail: str


def import_module_or_failure(module_name: str) -> ModuleType | SmokeFailure:
    """Import one module or return a structured smoke failure."""
    try:
        return import_module(module_name)
    except Exception as exception:
        return SmokeFailure(
            target=module_name,
            detail=f"{type(exception).__name__}: {exception}",
        )


def require_symbol(module_name: str, symbol_name: str) -> SmokeFailure | None:
    """Return a smoke failure when one installed module symbol is unavailable."""
    module = import_module_or_failure(module_name)
    if isinstance(module, SmokeFailure):
        return module

    candidate: object = getattr(module, symbol_name, None)
    if candidate is None:
        return SmokeFailure(
            target=f"{module_name}.{symbol_name}",
            detail="symbol is unavailable",
        )
    return None


def collect_base_failures() -> tuple[SmokeFailure, ...]:
    """Return base installed-package import failures."""
    failures: list[SmokeFailure] = []
    for module_name in BASE_INSTALL_MODULES:
        module = import_module_or_failure(module_name)
        if isinstance(module, SmokeFailure):
            failures.append(module)

    for module_name, symbol_name in BASE_INSTALL_SYMBOLS:
        failure = require_symbol(module_name, symbol_name)
        if failure is not None:
            failures.append(failure)

    return tuple(failures)


def collect_joblib_private_failures() -> tuple[SmokeFailure, ...]:
    """Return drift failures for joblib private surfaces used by async retries."""
    failures: list[SmokeFailure] = []
    loky_process_executor = import_module_or_failure(
        "joblib.externals.loky.process_executor"
    )
    if isinstance(loky_process_executor, SmokeFailure):
        failures.append(loky_process_executor)
    else:
        for symbol_name in JOBLIB_LOKY_EXCEPTION_SYMBOLS:
            candidate: object = getattr(loky_process_executor, symbol_name, None)
            if not isinstance(candidate, type) or not issubclass(
                candidate,
                BaseException,
            ):
                failures.append(
                    SmokeFailure(
                        target=f"joblib.externals.loky.process_executor.{symbol_name}",
                        detail="exception type is unavailable",
                    )
                )

    joblib = import_module_or_failure("joblib")
    if isinstance(joblib, SmokeFailure):
        failures.append(joblib)
    else:
        parallel_candidate: object = getattr(joblib, "Parallel", None)
        if not callable(parallel_candidate):
            failures.append(
                SmokeFailure(target="joblib.Parallel", detail="callable is unavailable")
            )
        else:
            runner = parallel_candidate(
                n_jobs=1,
                backend="threading",
                return_as="generator_unordered",
            )
            if getattr(runner, "_abort", None) is None:
                print(
                    "joblib.Parallel._abort is unavailable;",
                    "AsyncJoblibEvaluator will exercise generator.close() fallback.",
                )

    return tuple(failures)


def collect_mpi_failures() -> tuple[SmokeFailure, ...]:
    """Return optional MPI installed-world import failures."""
    failures: list[SmokeFailure] = []
    mpi4py_futures: ModuleType | None = None
    for module_name in MPI_INSTALL_MODULES:
        module = import_module_or_failure(module_name)
        if isinstance(module, SmokeFailure):
            failures.append(module)
        elif module_name == "mpi4py.futures":
            mpi4py_futures = module

    if mpi4py_futures is not None:
        candidate: object = getattr(mpi4py_futures, "MPIPoolExecutor", None)
        if candidate is None:
            failures.append(
                SmokeFailure(
                    target="mpi4py.futures.MPIPoolExecutor",
                    detail="symbol is unavailable",
                )
            )

    failure = require_symbol("variopt.evaluators", "MpiEvaluator")
    if failure is not None:
        failures.append(failure)

    return tuple(failures)


def collect_failures(surfaces: Sequence[SmokeSurface]) -> tuple[SmokeFailure, ...]:
    """Return all smoke failures for the requested surfaces."""
    failures: list[SmokeFailure] = []
    for surface in surfaces:
        if surface == "base":
            failures.extend(collect_base_failures())
        elif surface == "joblib-private":
            failures.extend(collect_joblib_private_failures())
        elif surface == "mpi":
            failures.extend(collect_mpi_failures())
        else:
            raise AssertionError(surface)
    return tuple(failures)


def parse_surface(value: str) -> SmokeSurface:
    """Return one supported smoke surface from a command-line value."""
    if value == "base":
        return "base"
    if value == "joblib-private":
        return "joblib-private"
    if value == "mpi":
        return "mpi"

    msg = f"unsupported smoke surface: {value!r}"
    raise ValueError(msg)


def parse_args(arguments: Sequence[str] | None = None) -> tuple[SmokeSurface, ...]:
    """Parse repeated ``--surface VALUE`` arguments."""
    raw_arguments = tuple(argv[1:] if arguments is None else arguments)
    surfaces: list[SmokeSurface] = []
    index = 0
    while index < len(raw_arguments):
        option = raw_arguments[index]
        if option != "--surface":
            msg = f"unsupported argument: {option!r}"
            raise ValueError(msg)
        index += 1
        if index >= len(raw_arguments):
            msg = "--surface requires a value"
            raise ValueError(msg)
        surfaces.append(parse_surface(raw_arguments[index]))
        index += 1

    if not surfaces:
        msg = "at least one --surface value is required"
        raise ValueError(msg)
    return tuple(surfaces)


def main() -> None:
    """Run installed-package smoke checks."""
    failures = collect_failures(parse_args())
    if not failures:
        print("installed-package smoke checks passed")
        return

    for failure in failures:
        print(f"{failure.target}: {failure.detail}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
