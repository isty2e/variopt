"""Private covariance-aware helpers for CSA proposal adaptation."""

from collections.abc import Sequence
from typing import Protocol, TypeVar, cast

import numpy as np
import numpy.typing as npt

from ......spaces import LeafPath
from ......spaces.projections import HomogeneousNumericSubspaceDescriptor
from ......spaces.types import SpaceCandidateValue
from .state.aggregate import CSAProposalState
from .state.attribution import (
    NumericSubspaceAttribution,
    NumericSubspaceDisplacement,
)
from .state.stats import ProposalNumericSubspaceCovarianceStat

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)
FloatVector = npt.NDArray[np.float64]
FloatMatrix = npt.NDArray[np.float64]


class _FloatMatrixEigh(Protocol):
    """Typed callable view of ``numpy.linalg.eigh`` for float matrices."""

    def __call__(self, matrix: FloatMatrix) -> tuple[FloatVector, FloatMatrix]:
        """Return eigenvalues and eigenvectors for ``matrix``."""
        ...


def _eigh_float_matrix(matrix: FloatMatrix) -> tuple[FloatVector, FloatMatrix]:
    """Return eigenvalues and eigenvectors for one float covariance matrix."""
    eigh_float_matrix = cast(_FloatMatrixEigh, getattr(np.linalg, "eigh"))
    raw_eigenvalues, raw_eigenvectors = eigh_float_matrix(matrix)
    return (
        np.asarray(raw_eigenvalues, dtype=np.float64),
        np.asarray(raw_eigenvectors, dtype=np.float64),
    )


def _transpose_float_matrix(matrix: FloatMatrix) -> FloatMatrix:
    """Return the transpose of one float matrix."""
    return cast(FloatMatrix, getattr(matrix, "T"))


def build_numeric_subspace_attribution(
    *,
    descriptor: HomogeneousNumericSubspaceDescriptor[BoundaryT, StructuredCandidateT],
    source_candidate: StructuredCandidateT,
) -> NumericSubspaceAttribution:
    """Return immutable numeric-subspace attribution for one source candidate.

    Parameters
    ----------
    descriptor : HomogeneousNumericSubspaceDescriptor[BoundaryT, StructuredCandidateT]
        Numeric subspace descriptor used to project the candidate.
    source_candidate : StructuredCandidateT
        Candidate that seeds the attribution payload.

    Returns
    -------
    NumericSubspaceAttribution
        Immutable attribution payload storing the source coordinates.
    """
    return NumericSubspaceAttribution(
        leaf_paths=descriptor.leaf_paths,
        source_coordinates=descriptor.coordinates_from_candidate(source_candidate),
    )


def infer_numeric_subspace_displacement(
    *,
    descriptor: HomogeneousNumericSubspaceDescriptor[BoundaryT, StructuredCandidateT],
    attribution: NumericSubspaceAttribution,
    observed_candidate: StructuredCandidateT,
) -> NumericSubspaceDisplacement | None:
    """Return observed displacement from one attributed numeric source.

    Parameters
    ----------
    descriptor : HomogeneousNumericSubspaceDescriptor[BoundaryT, StructuredCandidateT]
        Numeric subspace descriptor used to project the observed candidate.
    attribution : NumericSubspaceAttribution
        Attribution payload created from the source candidate.
    observed_candidate : StructuredCandidateT
        Candidate observed after the attributed proposal step.

    Returns
    -------
    NumericSubspaceDisplacement | None
        Observed displacement when the descriptor matches the attribution leaf
        paths, otherwise ``None``.
    """
    if descriptor.leaf_paths != attribution.leaf_paths:
        return None

    observed_coordinates = descriptor.coordinates_from_candidate(observed_candidate)
    return NumericSubspaceDisplacement(
        leaf_paths=descriptor.leaf_paths,
        displacement_coordinates=tuple(
            observed_coordinate - source_coordinate
            for observed_coordinate, source_coordinate in zip(
                observed_coordinates,
                attribution.source_coordinates,
                strict=True,
            )
        ),
    )


def sample_covariance_guided_candidate(
    *,
    descriptor: HomogeneousNumericSubspaceDescriptor[BoundaryT, StructuredCandidateT],
    source_candidate: StructuredCandidateT,
    selected_paths: Sequence[LeafPath],
    proposal_state: CSAProposalState,
    max_coordinate_fraction: float,
    random_state: np.random.RandomState,
) -> tuple[StructuredCandidateT, tuple[LeafPath, ...]] | None:
    """Return one covariance-guided child when proposal state has enough signal.

    Parameters
    ----------
    descriptor : HomogeneousNumericSubspaceDescriptor[BoundaryT, StructuredCandidateT]
        Numeric subspace descriptor for the selected leaf family.
    source_candidate : StructuredCandidateT
        Candidate to perturb.
    selected_paths : Sequence[LeafPath]
        Editable leaf paths selected by the outer proposal logic.
    proposal_state : CSAProposalState
        Proposal-adaptation state carrying covariance estimates.
    max_coordinate_fraction : float
        Maximum fraction of the numeric range allowed per coordinate update.
    random_state : np.random.RandomState
        Random state used for covariance sampling.

    Returns
    -------
    tuple[StructuredCandidateT, tuple[LeafPath, ...]] | None
        Covariance-guided child and changed paths, or ``None`` when the state
        does not provide enough covariance signal.
    """
    policy = proposal_state.policy
    if policy.numeric_covariance_strength <= 0.0:
        return None

    covariance_stat = proposal_state.covariance_stat_for_leaf_paths(
        descriptor.leaf_paths,
    )
    if covariance_stat is None:
        return None

    if covariance_stat.observation_count < policy.numeric_covariance_min_observations:
        return None

    selected_path_set = {tuple(path) for path in selected_paths}
    if len(selected_path_set) == 0:
        return None

    sampled_delta = sample_covariance_delta(
        covariance_stat,
        proposal_state=proposal_state,
        random_state=random_state,
    )
    masked_delta = tuple(
        0.0 if path not in selected_path_set else sampled_delta[index]
        for index, path in enumerate(descriptor.leaf_paths)
    )
    clipped_delta = descriptor.clip_coordinate_deltas(
        tuple(policy.numeric_covariance_strength * delta for delta in masked_delta),
        max_coordinate_fraction=max_coordinate_fraction,
    )
    source_coordinates = descriptor.coordinates_from_candidate(source_candidate)
    candidate = descriptor.candidate_from_coordinates(
        source_candidate,
        tuple(
            source_coordinate + delta
            for source_coordinate, delta in zip(
                source_coordinates,
                clipped_delta,
                strict=True,
            )
        ),
    )
    changed_paths = descriptor.changed_leaf_paths(source_candidate, candidate)
    if len(changed_paths) == 0:
        return None
    return candidate, changed_paths


def sample_covariance_delta(
    covariance_stat: ProposalNumericSubspaceCovarianceStat,
    *,
    proposal_state: CSAProposalState,
    random_state: np.random.RandomState,
) -> tuple[float, ...]:
    """Return one correlated coordinate delta sampled from proposal covariance.

    Parameters
    ----------
    covariance_stat : ProposalNumericSubspaceCovarianceStat
        Covariance summary for one numeric leaf family.
    proposal_state : CSAProposalState
        Proposal-adaptation state carrying decay and ridge settings.
    random_state : np.random.RandomState
        Random state used for multivariate sampling.

    Returns
    -------
    tuple[float, ...]
        Sampled coordinate delta in descriptor order.
    """
    covariance_mean = np.asarray(
        covariance_stat.effective_mean(
            current_update_index=proposal_state.update_index,
            credit_decay=proposal_state.policy.credit_decay,
        ),
        dtype=np.float64,
    )
    covariance_matrix = np.asarray(
        covariance_stat.effective_covariance(
            current_update_index=proposal_state.update_index,
            credit_decay=proposal_state.policy.credit_decay,
        ),
        dtype=np.float64,
    )
    stabilized_covariance = stabilize_covariance_matrix(
        covariance_matrix,
        ridge=proposal_state.policy.numeric_covariance_ridge,
    )
    sampled_delta: FloatVector = np.asarray(
        random_state.multivariate_normal(
            covariance_mean,
            stabilized_covariance,
        ),
        dtype=np.float64,
    )
    dimension = int(sampled_delta.size)
    return tuple(
        float(cast(np.float64, sampled_delta[index])) for index in range(dimension)
    )


def stabilize_covariance_matrix(
    covariance_matrix: FloatMatrix,
    *,
    ridge: float,
) -> FloatMatrix:
    """Return one symmetric positive-semidefinite covariance matrix.

    Parameters
    ----------
    covariance_matrix : FloatMatrix
        Raw covariance estimate.
    ridge : float
        Non-negative ridge added to each stabilized eigenvalue.

    Returns
    -------
    FloatMatrix
        Symmetric stabilized covariance matrix suitable for sampling.
    """
    covariance_transpose = _transpose_float_matrix(covariance_matrix)
    symmetric_covariance: FloatMatrix = np.asarray(
        0.5 * (covariance_matrix + covariance_transpose),
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = _eigh_float_matrix(symmetric_covariance)
    stabilized_eigenvalues: FloatVector = np.asarray(
        np.maximum(eigenvalues, 0.0) + ridge,
        dtype=np.float64,
    )
    dimension = int(stabilized_eigenvalues.size)
    stabilized_diagonal: FloatMatrix = np.zeros(
        (dimension, dimension), dtype=np.float64
    )
    np.fill_diagonal(stabilized_diagonal, stabilized_eigenvalues)
    eigenvector_transpose = _transpose_float_matrix(eigenvectors)
    stabilized_covariance: FloatMatrix = np.asarray(
        eigenvectors @ stabilized_diagonal @ eigenvector_transpose,
        dtype=np.float64,
    )
    stabilized_covariance_transpose = _transpose_float_matrix(
        stabilized_covariance,
    )
    return np.asarray(
        0.5 * (stabilized_covariance + stabilized_covariance_transpose),
        dtype=np.float64,
    )
