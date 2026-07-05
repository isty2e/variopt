"""Installed-package smoke checks for CI-built variopt wheels."""

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from sys import argv
from types import ModuleType
from typing import Literal, cast

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


def normalize_public_symbol_names(
    candidate: list[object] | tuple[object, ...],
) -> tuple[str, ...] | None:
    """Return canonical public symbol names when all entries are valid."""
    symbol_names: list[str] = []
    for name in candidate:
        if not isinstance(name, str) or name == "":
            return None
        symbol_names.append(name)
    return tuple(symbol_names)


def public_symbol_names(
    module: ModuleType, module_name: str
) -> tuple[str, ...] | SmokeFailure:
    """Return a module's canonical public names or a smoke failure."""
    all_names: object = getattr(module, "__all__", None)
    symbol_names: tuple[str, ...] | None = None
    if isinstance(all_names, list):
        raw_names = cast(list[object], all_names)
        symbol_names = normalize_public_symbol_names(raw_names)
    elif isinstance(all_names, tuple):
        raw_names = cast(tuple[object, ...], all_names)
        symbol_names = normalize_public_symbol_names(raw_names)

    if symbol_names is None:
        return SmokeFailure(
            target=f"{module_name}.__all__",
            detail="__all__ must be a list or tuple of non-empty strings",
        )

    return symbol_names


def collect_base_failures() -> tuple[SmokeFailure, ...]:
    """Return base installed-package import failures."""
    failures: list[SmokeFailure] = []
    for module_name in BASE_INSTALL_MODULES:
        module = import_module_or_failure(module_name)
        if isinstance(module, SmokeFailure):
            failures.append(module)
            continue

        symbol_names = public_symbol_names(module, module_name)
        if isinstance(symbol_names, SmokeFailure):
            failures.append(symbol_names)
            continue

        for symbol_name in symbol_names:
            candidate: object = getattr(module, symbol_name, None)
            if candidate is None:
                failures.append(
                    SmokeFailure(
                        target=f"{module_name}.{symbol_name}",
                        detail="public __all__ symbol is unavailable",
                    )
                )

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


def configure_mpi4py_smoke_runtime() -> SmokeFailure | None:
    """Disable MPI auto-finalize for import-only smoke checks."""
    mpi4py = import_module_or_failure("mpi4py")
    if isinstance(mpi4py, SmokeFailure):
        return mpi4py

    rc: object = getattr(mpi4py, "rc", None)
    if rc is None:
        return SmokeFailure(
            target="mpi4py.rc",
            detail="runtime configuration is unavailable",
        )
    setattr(rc, "finalize", False)
    return None


def collect_mpi_failures() -> tuple[SmokeFailure, ...]:
    """Return optional MPI installed-world import failures."""
    failures: list[SmokeFailure] = []
    runtime_failure = configure_mpi4py_smoke_runtime()
    if runtime_failure is not None:
        failures.append(runtime_failure)
        return tuple(failures)

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
