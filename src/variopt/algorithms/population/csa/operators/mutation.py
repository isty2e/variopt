"""Structured mutation kernels for built-in CSA operators."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from .....spaces import StructuredSearchSpace
from .....spaces.structured import LeafPath
from .....spaces.types import SpaceCandidateValue
from .editing import (
    choose_paths_without_replacement,
    mutate_leaf_value,
    sample_exchange_count,
    sample_leaf_value,
)

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)


def random_reset_mutation(
    *,
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    candidate: CandidateT,
    max_exchange_fraction: float,
    random_state: np.random.RandomState,
    validate_candidate: bool = True,
) -> CandidateT:
    """Resample a bounded subset of editable leaves.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space that defines editable leaves.
    candidate : CandidateT
        Parent candidate to mutate.
    max_exchange_fraction : float
        Maximum fraction of active leaves that may be resampled.
    random_state : np.random.RandomState
        Random state used for path selection and leaf resampling.
    validate_candidate : bool, default=True
        Whether to validate ``candidate`` before using validated-space accessors.
        CSA generation can disable this for candidates already sourced from its
        validated bank state.

    Returns
    -------
    CandidateT
        Mutated child with selected leaves redrawn from their declared spaces.
    """
    if validate_candidate:
        space.validate(candidate)
    editable_paths = space.active_leaf_paths_for_validated_candidate(candidate)
    exchange_count = sample_exchange_count(
        leaf_count=len(editable_paths),
        max_exchange_fraction=max_exchange_fraction,
        random_state=random_state,
    )
    selected_paths = choose_paths_without_replacement(
        editable_paths,
        exchange_count,
        random_state,
    )
    replacements = {
        path: sample_leaf_value(space.leaf_space_at_path(path), random_state)
        for path in selected_paths
    }
    return space.replace_leaf_values_in_validated_candidate(candidate, replacements)


def random_reset_mutation_on_paths(
    *,
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    candidate: CandidateT,
    selected_paths: Sequence[LeafPath],
    random_state: np.random.RandomState,
    validate_candidate: bool = True,
) -> CandidateT:
    """Resample one explicit set of leaves.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space that defines editable leaves.
    candidate : CandidateT
        Parent candidate to mutate.
    selected_paths : Sequence[LeafPath]
        Explicit leaf paths to resample.
    random_state : np.random.RandomState
        Random state used for leaf resampling.
    validate_candidate : bool, default=True
        Whether to validate ``candidate`` before using validated-space accessors.
        CSA generation can disable this for candidates already sourced from its
        validated bank state.

    Returns
    -------
    CandidateT
        Mutated child with the requested leaves redrawn from their declared
        spaces.

    Raises
    ------
    ValueError
        If ``selected_paths`` is empty.
    """
    if len(selected_paths) == 0:
        msg = "selected_paths must not be empty"
        raise ValueError(msg)

    if validate_candidate:
        space.validate(candidate)
    replacements = {
        path: sample_leaf_value(space.leaf_space_at_path(path), random_state)
        for path in selected_paths
    }
    return space.replace_leaf_values_in_validated_candidate(candidate, replacements)


def bounded_mutation(
    *,
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    candidate: CandidateT,
    max_perturbation_fraction: float,
    random_state: np.random.RandomState,
    validate_candidate: bool = True,
) -> CandidateT:
    """Perturb a bounded subset of editable leaves in place.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space that defines editable leaves.
    candidate : CandidateT
        Parent candidate to mutate.
    max_perturbation_fraction : float
        Maximum fraction of active leaves that may be perturbed.
    random_state : np.random.RandomState
        Random state used for path selection and bounded perturbation.
    validate_candidate : bool, default=True
        Whether to validate ``candidate`` before using validated-space accessors.
        CSA generation can disable this for candidates already sourced from its
        validated bank state.

    Returns
    -------
    CandidateT
        Mutated child with selected leaves perturbed relative to the parent.
    """
    if validate_candidate:
        space.validate(candidate)
    editable_paths = space.active_leaf_paths_for_validated_candidate(candidate)
    exchange_count = sample_exchange_count(
        leaf_count=len(editable_paths),
        max_exchange_fraction=max_perturbation_fraction,
        random_state=random_state,
    )
    selected_paths = choose_paths_without_replacement(
        editable_paths,
        exchange_count,
        random_state,
    )
    replacements = {
        path: mutate_leaf_value(
            space=space.leaf_space_at_path(path),
            value=space.leaf_value_at_validated_path(candidate, path),
            max_perturbation_fraction=max_perturbation_fraction,
            random_state=random_state,
        )
        for path in selected_paths
    }
    return space.replace_leaf_values_in_validated_candidate(candidate, replacements)


def bounded_mutation_on_paths(
    *,
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    candidate: CandidateT,
    selected_paths: Sequence[LeafPath],
    max_perturbation_fraction: float,
    random_state: np.random.RandomState,
    validate_candidate: bool = True,
) -> CandidateT:
    """Perturb one explicit set of leaves.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space that defines editable leaves.
    candidate : CandidateT
        Parent candidate to mutate.
    selected_paths : Sequence[LeafPath]
        Explicit leaf paths to perturb.
    max_perturbation_fraction : float
        Maximum perturbation fraction passed to leaf-level mutation.
    random_state : np.random.RandomState
        Random state used for bounded perturbation.
    validate_candidate : bool, default=True
        Whether to validate ``candidate`` before using validated-space accessors.
        CSA generation can disable this for candidates already sourced from its
        validated bank state.

    Returns
    -------
    CandidateT
        Mutated child with the requested leaves perturbed.

    Raises
    ------
    ValueError
        If ``selected_paths`` is empty.
    """
    if len(selected_paths) == 0:
        msg = "selected_paths must not be empty"
        raise ValueError(msg)

    if validate_candidate:
        space.validate(candidate)
    replacements = {
        path: mutate_leaf_value(
            space=space.leaf_space_at_path(path),
            value=space.leaf_value_at_validated_path(candidate, path),
            max_perturbation_fraction=max_perturbation_fraction,
            random_state=random_state,
        )
        for path in selected_paths
    }
    return space.replace_leaf_values_in_validated_candidate(candidate, replacements)
