"""Structured differential-evolution variation primitives."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from ....randomness import random_state_choice_indices_without_replacement
from ....spaces import IntegerSpace, RealSpace, SearchSpace, StructuredSearchSpace
from ....spaces.structured import LeafPath
from ....spaces.types import SpaceCandidateValue

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)


def require_numeric_structured_space(
    space: SearchSpace[BoundaryT, StructuredCandidateT],
) -> StructuredSearchSpace[BoundaryT, StructuredCandidateT]:
    """Return a numeric structured space or raise.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Search space supplied to the DE variation operator.

    Returns
    -------
    StructuredSearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space containing only numeric editable leaves.

    Raises
    ------
    TypeError
        If ``space`` is not structured or exposes non-numeric leaves.
    ValueError
        If ``space`` does not expose any editable leaves.
    """
    if not isinstance(space, StructuredSearchSpace):
        msg = "differential evolution requires a structured search space"
        raise TypeError(msg)

    leaf_paths = space.leaf_paths()
    if len(leaf_paths) == 0:
        msg = "differential evolution requires at least one editable structured leaf"
        raise ValueError(msg)

    for path in leaf_paths:
        if not isinstance(space.leaf_space_at_path(path), (RealSpace, IntegerSpace)):
            msg = "space must contain only numeric leaves for differential evolution"
            raise TypeError(msg)

    return space


def differential_evolution_variation(
    *,
    space: StructuredSearchSpace[BoundaryT, StructuredCandidateT],
    target_parent: StructuredCandidateT,
    base_parent: StructuredCandidateT,
    differential_parent_a: StructuredCandidateT,
    differential_parent_b: StructuredCandidateT,
    mutation_factor: float,
    recombination_probability: float,
    n_cross: int,
    random_state: np.random.RandomState,
) -> StructuredCandidateT:
    """Return a DE-style child under numeric structured leaves.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space defining the editable numeric leaves.
    target_parent : StructuredCandidateT
        Target parent whose topology and untouched leaves are preserved.
    base_parent : StructuredCandidateT
        Base parent contributing the donor anchor values.
    differential_parent_a : StructuredCandidateT
        First differential parent used for the difference vector.
    differential_parent_b : StructuredCandidateT
        Second differential parent used for the difference vector.
    mutation_factor : float
        Differential-evolution mutation factor.
    recombination_probability : float
        Probability of copying each donor leaf into the child.
    n_cross : int
        Minimum number of leaves forced to cross over from the donor.
    random_state : np.random.RandomState
        Random state used for crossover-path sampling.

    Returns
    -------
    StructuredCandidateT
        Child candidate produced by structured DE variation.

    Raises
    ------
    ValueError
        If the parents do not share the same active topology or ``n_cross``
        exceeds the number of editable leaves.
    TypeError
        If any editable leaf space or leaf value is non-numeric.
    """
    space.validate(target_parent)
    space.validate(base_parent)
    space.validate(differential_parent_a)
    space.validate(differential_parent_b)

    editable_paths = space.active_leaf_paths_for_validated_candidate(target_parent)
    if editable_paths != space.active_leaf_paths_for_validated_candidate(base_parent):
        msg = "differential evolution variation requires parents with matching active topology"
        raise ValueError(msg)
    if editable_paths != space.active_leaf_paths_for_validated_candidate(
        differential_parent_a,
    ):
        msg = "differential evolution variation requires parents with matching active topology"
        raise ValueError(msg)
    if editable_paths != space.active_leaf_paths_for_validated_candidate(
        differential_parent_b,
    ):
        msg = "differential evolution variation requires parents with matching active topology"
        raise ValueError(msg)

    if n_cross > len(editable_paths):
        msg = "n_cross must not exceed the number of editable leaves"
        raise ValueError(msg)

    donor_replacements = {
        path: differential_leaf_value(
            space=require_numeric_leaf_space(space.leaf_space_at_path(path)),
            base_value=require_numeric_leaf_value(
                space.leaf_value_at_validated_path(base_parent, path),
            ),
            differential_value_a=require_numeric_leaf_value(
                space.leaf_value_at_validated_path(differential_parent_a, path),
            ),
            differential_value_b=require_numeric_leaf_value(
                space.leaf_value_at_validated_path(differential_parent_b, path),
            ),
            mutation_factor=mutation_factor,
        )
        for path in editable_paths
    }

    selected_paths = {
        path
        for path in editable_paths
        if float(random_state.random_sample()) < recombination_probability
    }
    selected_paths.update(
        choose_paths_without_replacement(
            editable_paths,
            n_cross,
            random_state,
        ),
    )

    replacements = {path: donor_replacements[path] for path in selected_paths}
    return space.replace_leaf_values_in_validated_candidate(
        target_parent,
        replacements,
    )


def choose_paths_without_replacement(
    paths: Sequence[LeafPath],
    count: int,
    random_state: np.random.RandomState,
) -> tuple[LeafPath, ...]:
    """Return random distinct paths without replacement.

    Parameters
    ----------
    paths : Sequence[LeafPath]
        Candidate leaf paths to sample from.
    count : int
        Number of distinct paths to sample.
    random_state : np.random.RandomState
        Random state used for the sampling step.

    Returns
    -------
    tuple[LeafPath, ...]
        Distinct sampled paths in sampled order.

    Raises
    ------
    ValueError
        If ``count`` is not positive or exceeds ``len(paths)``.
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


def require_numeric_leaf_space(
    space: object,
) -> RealSpace | IntegerSpace:
    """Return a numeric leaf space or raise.

    Parameters
    ----------
    space : object
        Runtime leaf space to validate.

    Returns
    -------
    RealSpace | IntegerSpace
        Numeric leaf space accepted by the DE operator.

    Raises
    ------
    TypeError
        If ``space`` is not a numeric leaf space.
    """
    if isinstance(space, (RealSpace, IntegerSpace)):
        return space

    msg = "space must contain only numeric leaves for differential evolution"
    raise TypeError(msg)


def require_numeric_leaf_value(value: SpaceCandidateValue) -> float | int:
    """Return a numeric leaf value or raise.

    Parameters
    ----------
    value : SpaceCandidateValue
        Runtime leaf value to validate.

    Returns
    -------
    float | int
        Canonical numeric leaf value accepted by the DE operator.

    Raises
    ------
    TypeError
        If ``value`` is boolean or otherwise non-numeric.
    """
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        msg = "differential evolution requires numeric leaf values"
        raise TypeError(msg)
    return value


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
        Numeric leaf space constraining the donor value.
    base_value : float | int
        Base parent value for the leaf.
    differential_value_a : float | int
        First differential parent value.
    differential_value_b : float | int
        Second differential parent value.
    mutation_factor : float
        Differential-evolution mutation factor.

    Returns
    -------
    float | int
        Donor value clamped to the leaf-space bounds.

    Raises
    ------
    TypeError
        If an integer leaf receives non-canonical integer values.
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
