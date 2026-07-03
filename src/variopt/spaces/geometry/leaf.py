"""Leaf-level geometry helpers and built-in-family type guards."""

from math import isfinite
from typing import TypeAlias, TypeGuard, TypeVar

from ..composites import RecordCandidate
from ..composites.array_space import ArraySpace
from ..composites.record_space import RecordSpace
from ..composites.tuple_space import TupleSpace
from ..permutation import PermutationSpace
from ..scalar import (
    CategoricalSpace,
    IntegerSpace,
    RealSpace,
    require_categorical_scalar,
)
from ..structured import StructuredLeafSpace, StructuredSearchSpace
from ..types import SpaceBoundaryValue, SpaceCandidateValue, SpaceScalarValue

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)
BuiltinChildSpace: TypeAlias = (
    RealSpace
    | IntegerSpace
    | CategoricalSpace[SpaceScalarValue]
    | PermutationSpace
    | TupleSpace
    | RecordSpace
    | ArraySpace[SpaceBoundaryValue, SpaceCandidateValue]
)


def require_real_candidate(
    *,
    value: SpaceCandidateValue,
    message: str,
) -> float:
    """Return one canonical real candidate value.

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


def require_integer_candidate(
    *,
    value: SpaceCandidateValue,
    message: str,
) -> int:
    """Return one canonical integer candidate value.

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


def require_candidate_tuple(
    *,
    value: SpaceCandidateValue,
    message: str,
) -> tuple[SpaceCandidateValue, ...]:
    """Return one canonical tuple candidate value.

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


def require_record_candidate(
    *,
    value: SpaceCandidateValue,
    message: str,
) -> RecordCandidate:
    """Return one canonical record candidate value.

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


def is_builtin_structured_space(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
) -> TypeGuard[BuiltinChildSpace]:
    """Return whether one structured space uses the built-in geometry family.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured space to classify.

    Returns
    -------
    TypeGuard[BuiltinChildSpace]
        ``True`` when ``space`` belongs to the built-in geometry family.
    """
    return isinstance(
        space,
        (
            RealSpace,
            IntegerSpace,
            CategoricalSpace,
            PermutationSpace,
            TupleSpace,
            RecordSpace,
            ArraySpace,
        ),
    )


def is_builtin_child_space(
    space: object,
) -> TypeGuard[BuiltinChildSpace]:
    """Return whether one search space is in the built-in structured geometry family.

    Parameters
    ----------
    space : object
        Search space object to classify.

    Returns
    -------
    TypeGuard[BuiltinChildSpace]
        ``True`` when ``space`` belongs to the built-in geometry family.
    """
    return isinstance(
        space,
        (
            RealSpace,
            IntegerSpace,
            CategoricalSpace,
            PermutationSpace,
            TupleSpace,
            RecordSpace,
            ArraySpace,
        ),
    )


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
