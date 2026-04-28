"""Structured leaf editing primitives shared by CSA operator kernels."""

from collections.abc import Sequence

import numpy as np

from .....randomness import (
    random_state_choice_indices_without_replacement,
    random_state_randint,
)
from .....spaces import (
    IntegerSpace,
    LeafPath,
    RealSpace,
    StructuredLeafSpace,
)
from .....spaces.geometry.leaf import is_categorical_leaf_space
from .....spaces.types import SpaceCandidateValue, SpaceScalarValue


def sample_exchange_count(
    *,
    leaf_count: int,
    max_exchange_fraction: float,
    random_state: np.random.RandomState,
) -> int:
    """Return a legacy-style random exchange count.

    Parameters
    ----------
    leaf_count : int
        Number of mutable leaf positions available.
    max_exchange_fraction : float
        Fractional cap on the number of exchanged leaf positions.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    int
        Sampled exchange count.

    Raises
    ------
    ValueError
        If ``leaf_count`` is not positive.
    """
    if leaf_count <= 0:
        msg = "leaf_count must be positive"
        raise ValueError(msg)

    if leaf_count == 1:
        return 1

    upper_exclusive = min(
        leaf_count + 1,
        max(2, int(leaf_count * max_exchange_fraction)),
    )
    return random_state_randint(random_state, 1, upper_exclusive)


def choose_paths_without_replacement(
    paths: Sequence[LeafPath],
    count: int,
    random_state: np.random.RandomState,
) -> tuple[LeafPath, ...]:
    """Return random distinct paths without replacement.

    Parameters
    ----------
    paths : Sequence[LeafPath]
        Candidate leaf paths to choose from.
    count : int
        Number of distinct paths to select.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    tuple[LeafPath, ...]
        Randomly selected distinct leaf paths.

    Raises
    ------
    ValueError
        If ``count`` is non-positive or larger than ``len(paths)``.
    """
    if count <= 0:
        msg = "count must be positive"
        raise ValueError(msg)

    if count > len(paths):
        msg = "count must not exceed the number of paths"
        raise ValueError(msg)

    selected_indices = random_state_choice_indices_without_replacement(
        random_state,
        len(paths),
        count,
    )
    return tuple(paths[index] for index in selected_indices)


def random_index(length: int, random_state: np.random.RandomState) -> int:
    """Return a random index in ``[0, length)``.

    Parameters
    ----------
    length : int
        Exclusive upper bound.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    int
        Random index in ``[0, length)``.

    Raises
    ------
    ValueError
        If ``length`` is not positive.
    """
    if length <= 0:
        msg = "length must be positive"
        raise ValueError(msg)

    return random_state_randint(random_state, length)


def sample_leaf_value(
    space: StructuredLeafSpace,
    random_state: np.random.RandomState,
) -> SpaceCandidateValue:
    """Sample a leaf value from the given leaf space.

    Parameters
    ----------
    space : StructuredLeafSpace
        Leaf space to sample from.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    SpaceCandidateValue
        Sampled canonical leaf value.
    """
    return space.sample(random_state)


def mutate_leaf_value(
    *,
    space: StructuredLeafSpace,
    value: SpaceCandidateValue,
    max_perturbation_fraction: float,
    random_state: np.random.RandomState,
) -> SpaceCandidateValue:
    """Return a bounded mutation of a leaf value.

    Parameters
    ----------
    space : StructuredLeafSpace
        Leaf space that owns ``value``.
    value : SpaceCandidateValue
        Canonical leaf value to mutate.
    max_perturbation_fraction : float
        Fractional cap on the mutation magnitude.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    SpaceCandidateValue
        Mutated canonical leaf value.

    Raises
    ------
    TypeError
        If ``value`` is not compatible with ``space`` or if the leaf space type
        is unsupported.
    """
    if isinstance(space, RealSpace):
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            msg = "real leaf mutation requires a numeric candidate value"
            raise TypeError(msg)
        current_value = float(value)
        coordinate_low, coordinate_high = space.coordinate_bounds()
        coordinate_span = coordinate_high - coordinate_low
        if coordinate_span == 0.0:
            return current_value

        delta = coordinate_span * max_perturbation_fraction * (
            2.0 * float(random_state.random_sample()) - 1.0
        )
        return space.project_coordinate(space.to_coordinate(current_value) + delta)

    if isinstance(space, IntegerSpace):
        if type(value) is not int:
            msg = "integer leaf mutation requires a canonical integer value"
            raise TypeError(msg)
        current_value = value
        coordinate_low, coordinate_high = space.coordinate_bounds()
        coordinate_span = coordinate_high - coordinate_low
        if coordinate_span == 0.0:
            return current_value

        coordinate_delta = coordinate_span * max_perturbation_fraction * (
            2.0 * float(random_state.random_sample()) - 1.0
        )
        return space.project_coordinate(
            space.to_coordinate(current_value) + coordinate_delta,
        )

    if not is_categorical_leaf_space(space):
        msg = f"unsupported structured leaf space: {type(space)!r}"
        raise TypeError(msg)

    scalar_value = require_scalar_leaf_value(value)
    choices: tuple[SpaceScalarValue, ...] = space.alternatives(scalar_value)
    if not choices:
        return value

    return choices[random_index(len(choices), random_state)]


def differential_leaf_value(
    *,
    space: RealSpace | IntegerSpace,
    base_value: float | int,
    differential_value_a: float | int,
    differential_value_b: float | int,
    mutation_factor: float,
) -> float | int:
    """Return a DE donor leaf value.

    Parameters
    ----------
    space : RealSpace | IntegerSpace
        Numeric leaf space that bounds the donor value.
    base_value : float | int
        Base leaf value.
    differential_value_a : float | int
        First differential source value.
    differential_value_b : float | int
        Second differential source value.
    mutation_factor : float
        Differential scaling factor.

    Returns
    -------
    float | int
        Donor leaf value projected back into the declared leaf bounds.

    Raises
    ------
    TypeError
        If integer differential evolution is requested with non-canonical
        integer values.
    """
    if isinstance(space, RealSpace):
        donor_value = (
            float(base_value)
            + mutation_factor
            * (float(differential_value_a) - float(differential_value_b))
        )
        return min(space.high, max(space.low, donor_value))

    if (
        type(base_value) is not int
        or type(differential_value_a) is not int
        or type(differential_value_b) is not int
    ):
        msg = "integer differential evolution requires canonical integer leaf values"
        raise TypeError(msg)

    donor_value = (
        float(base_value)
        + mutation_factor
        * (float(differential_value_a) - float(differential_value_b))
    )
    rounded_value = int(round(donor_value))
    return min(space.high, max(space.low, rounded_value))


def require_scalar_leaf_value(value: SpaceCandidateValue) -> SpaceScalarValue:
    """Return a scalar leaf value or raise.

    Parameters
    ----------
    value : SpaceCandidateValue
        Candidate value to validate.

    Returns
    -------
    SpaceScalarValue
        Canonical scalar leaf value.

    Raises
    ------
    TypeError
        If ``value`` is not scalar.
    """
    if isinstance(value, (bool, int, float, str, bytes, bytearray)):
        return value

    msg = "categorical leaf mutation requires a scalar canonical value"
    raise TypeError(msg)
