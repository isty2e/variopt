"""Leaf-level geometry helpers."""

from math import isfinite
from typing import TypeGuard

from ..composites import RecordCandidate
from ..scalar import (
    CategoricalSpace,
    require_categorical_scalar,
)
from ..structured import StructuredLeafSpace
from ..types import SpaceCandidateValue, SpaceScalarValue


def require_geometry_real_candidate(
    *,
    value: SpaceCandidateValue,
    message: str,
) -> float:
    """Return one canonical real candidate value for geometry calculations.

    Parameters
    ----------
    value : SpaceCandidateValue
        Candidate value to validate.
    message : str
        Error message used when the type check fails.

    Returns
    -------
    float
        Finite canonical real candidate.

    Raises
    ------
    TypeError
        If ``value`` is not a canonical float.
    ValueError
        If ``value`` is not finite.
    """
    if type(value) is not float:
        raise TypeError(message)

    if not isfinite(value):
        msg = "real candidate must be finite"
        raise ValueError(msg)
    return value


def require_geometry_integer_candidate(
    *,
    value: SpaceCandidateValue,
    message: str,
) -> int:
    """Return one canonical integer candidate value for geometry calculations.

    Parameters
    ----------
    value : SpaceCandidateValue
        Candidate value to validate.
    message : str
        Error message used when the type check fails.

    Returns
    -------
    int
        Canonical integer candidate.

    Raises
    ------
    TypeError
        If ``value`` is not a canonical integer.
    """
    if type(value) is not int:
        raise TypeError(message)
    return value


def require_geometry_candidate_tuple(
    *,
    value: SpaceCandidateValue,
    message: str,
) -> tuple[SpaceCandidateValue, ...]:
    """Return one canonical tuple candidate value for geometry calculations.

    Parameters
    ----------
    value : SpaceCandidateValue
        Candidate value to validate.
    message : str
        Error message used when the type check fails.

    Returns
    -------
    tuple[SpaceCandidateValue, ...]
        Canonical tuple candidate.

    Raises
    ------
    TypeError
        If ``value`` is not a tuple.
    """
    if not isinstance(value, tuple):
        raise TypeError(message)
    return value


def require_geometry_record_candidate(
    *,
    value: SpaceCandidateValue,
    message: str,
) -> RecordCandidate:
    """Return one canonical record candidate value for geometry calculations.

    Parameters
    ----------
    value : SpaceCandidateValue
        Candidate value to validate.
    message : str
        Error message used when the type check fails.

    Returns
    -------
    RecordCandidate
        Canonical record candidate.

    Raises
    ------
    TypeError
        If ``value`` is not a :class:`RecordCandidate`.
    """
    if not isinstance(value, RecordCandidate):
        raise TypeError(message)
    return value


def validate_categorical_choice(
    space: CategoricalSpace[SpaceScalarValue],
    value: SpaceCandidateValue,
) -> None:
    """Validate one categorical candidate against the declared choices.

    Parameters
    ----------
    space : CategoricalSpace[SpaceScalarValue]
        Categorical space defining the allowed choices.
    value : SpaceCandidateValue
        Candidate value to validate.

    Raises
    ------
    TypeError
        If ``value`` is not a scalar categorical candidate.
    ValueError
        If ``value`` is not one of the declared categorical choices.
    """
    scalar_value = require_categorical_scalar(value)
    space.validate(scalar_value)


def is_categorical_leaf_space(
    space: StructuredLeafSpace,
) -> TypeGuard[CategoricalSpace[SpaceScalarValue]]:
    """Return whether one structured leaf space is categorical.

    Parameters
    ----------
    space : StructuredLeafSpace
        Leaf space to classify.

    Returns
    -------
    TypeGuard[CategoricalSpace[SpaceScalarValue]]
        ``True`` when ``space`` is categorical.
    """
    return isinstance(space, CategoricalSpace)
