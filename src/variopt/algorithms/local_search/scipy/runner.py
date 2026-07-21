"""Execution helpers for SciPy local-search integration."""

from collections.abc import Callable, Sequence

from .contracts import ScipyMinimizeMethod, ScipyOptimizeResult


def run_scipy_minimize(
    *,
    objective_in_coordinate_space: Callable[[Sequence[float]], float],
    initial_coordinates: tuple[float, ...],
    method: ScipyMinimizeMethod,
    coordinate_bounds: tuple[tuple[float, float], ...],
    tolerance: float | None,
    options: dict[str, int],
) -> ScipyOptimizeResult:
    """Run ``scipy.optimize.minimize`` behind a typed boundary helper.

    Parameters
    ----------
    objective_in_coordinate_space : Callable[[Sequence[float]], float]
        Objective callable in coordinate space.
    initial_coordinates : tuple[float, ...]
        Initial coordinate vector passed to SciPy.
    method : ScipyMinimizeMethod
        Supported SciPy minimize method.
    coordinate_bounds : tuple[tuple[float, float], ...]
        Coordinate-space bounds passed to SciPy.
    tolerance : float | None
        Optional solver tolerance.
    options : dict[str, int]
        SciPy option dictionary.

    Returns
    -------
    ScipyOptimizeResult
        Raw SciPy optimize result exposing the supported result surface.
    """
    # Keep the SciPy backend outside algorithm-facade initialization.
    from scipy import optimize  # pyright: ignore[reportMissingTypeStubs]

    return optimize.minimize(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        objective_in_coordinate_space,
        x0=initial_coordinates,
        method=method,
        bounds=coordinate_bounds,
        tol=tolerance,
        options=options,
    )
