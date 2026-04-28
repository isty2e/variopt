"""Validation and shared utility helpers for CSA operator wrappers."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from .....spaces import SearchSpace, StructuredSearchSpace
from .....spaces.types import SpaceCandidateValue

BoundaryT = TypeVar("BoundaryT")
GeneralCandidateT = TypeVar("GeneralCandidateT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)


def validate_fraction(value: float, *, name: str) -> None:
    """Require one fractional hyperparameter in ``(0.0, 1.0]``.

    Parameters
    ----------
    value : float
        Fractional hyperparameter value to validate.
    name : str
        Parameter name used in error messages.

    Raises
    ------
    ValueError
        If ``value`` is outside ``(0.0, 1.0]``.
    """
    if value <= 0.0 or value > 1.0:
        msg = f"{name} must be in the interval (0.0, 1.0]"
        raise ValueError(msg)


def validate_probability(value: float, *, name: str) -> None:
    """Require one probability hyperparameter in ``[0.0, 1.0]``.

    Parameters
    ----------
    value : float
        Probability value to validate.
    name : str
        Parameter name used in error messages.

    Raises
    ------
    ValueError
        If ``value`` is outside ``[0.0, 1.0]``.
    """
    if value < 0.0 or value > 1.0:
        msg = f"{name} must be in the interval [0.0, 1.0]"
        raise ValueError(msg)


def require_parent_count(
    parents: Sequence[GeneralCandidateT],
    *,
    arity: int,
) -> None:
    """Require one exact parent count for one variation-operator call.

    Parameters
    ----------
    parents : Sequence[GeneralCandidateT]
        Parent candidates supplied to the operator.
    arity : int
        Required parent count.

    Raises
    ------
    ValueError
        If ``parents`` does not contain exactly ``arity`` candidates.
    """
    if len(parents) != arity:
        msg = f"expected exactly {arity} parent candidate(s), got {len(parents)}"
        raise ValueError(msg)


def sample_weighted_index(
    weights: Sequence[float],
    random_state: np.random.RandomState,
) -> int:
    """Return one sampled operator index from non-negative weights.

    Parameters
    ----------
    weights : Sequence[float]
        Non-negative sampling weights.
    random_state : np.random.RandomState
        Random state used for sampling.

    Returns
    -------
    int
        Sampled weight index. When all weights are zero, the final index is
        returned by construction.
    """
    total_weight = sum(weights)
    threshold = float(random_state.random_sample()) * total_weight
    cumulative_weight = 0.0

    for index, weight in enumerate(weights):
        cumulative_weight += weight
        if threshold <= cumulative_weight:
            return index

    return len(weights) - 1


def require_structured_space(
    space: SearchSpace[BoundaryT, StructuredCandidateT],
) -> StructuredSearchSpace[BoundaryT, StructuredCandidateT]:
    """Normalize one search-space boundary value into the structured contract.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Search space supplied to a built-in structured operator.

    Returns
    -------
    StructuredSearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space accepted by the built-in operator family.

    Raises
    ------
    TypeError
        If ``space`` is not a structured search space.
    """
    if not isinstance(space, StructuredSearchSpace):
        msg = "CSA built-in structured operators require a structured search space"
        raise TypeError(msg)
    _ = space.leaf_paths()
    return space
