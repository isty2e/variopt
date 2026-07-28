"""Cluster-linkage helpers for CSA clustering state."""

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from ......distance import require_valid_distance
from ......diversity import DiversityMetric
from ......typevars import CandidateT
from ..bank import BankEntry


class IndexedDistanceWorkspace(Protocol[CandidateT]):
    """Index-aligned pairwise distance source for one immutable entry snapshot."""

    def is_aligned_with_entries(
        self,
        entries: Sequence[BankEntry[CandidateT]],
    ) -> bool:
        """Return whether entries carry the represented candidates in order."""
        ...

    def distance(self, left_index: int, right_index: int) -> float:
        """Return the distance between two snapshot indices."""
        ...


def cluster_labels_for_entries(
    *,
    entries: Sequence[BankEntry[CandidateT]],
    cluster_distance: float,
    diversity_metric: DiversityMetric[CandidateT],
    distance_workspace: IndexedDistanceWorkspace[CandidateT] | None = None,
) -> tuple[int, ...]:
    """Cluster a bank snapshot with hierarchical linkage.

    Parameters
    ----------
    entries : Sequence[BankEntry[CandidateT]]
        Bank entries to cluster.
    cluster_distance : float
        Flat-cluster distance threshold passed to SciPy.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute pairwise distances.
    distance_workspace : IndexedDistanceWorkspace[CandidateT] | None, default=None
        Optional operation-local distance source aligned to ``entries``.

    Returns
    -------
    tuple[int, ...]
        Linkage-derived cluster labels aligned with ``entries``.
    """
    if len(entries) == 0:
        return ()

    if len(entries) == 1:
        return (1,)

    # Keep the SciPy backend outside population-facade initialization.
    from scipy.cluster import hierarchy  # pyright: ignore[reportMissingTypeStubs]

    condensed_distance_values = np.asarray(
        condensed_distances(
            entries=entries,
            diversity_metric=diversity_metric,
            distance_workspace=distance_workspace,
        ),
        dtype=np.float64,
    )
    linkage_output = cast(
        npt.NDArray[np.float64],
        hierarchy.linkage(condensed_distance_values),  # pyright: ignore[reportUnknownMemberType]
    )
    linkage_matrix = np.asarray(linkage_output, dtype=np.float64)
    cluster_label_array = cast(
        npt.NDArray[np.int_],
        hierarchy.fcluster(  # pyright: ignore[reportUnknownMemberType]
            linkage_matrix,
            cluster_distance,
            criterion="distance",
        ),
    )
    cluster_label_list = cast(
        list[int],
        np.asarray(cluster_label_array, dtype=np.int_).tolist(),
    )
    return tuple(cluster_label_list)


def condensed_distances(
    *,
    entries: Sequence[BankEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
    distance_workspace: IndexedDistanceWorkspace[CandidateT] | None = None,
) -> tuple[float, ...]:
    """Return condensed pairwise distances for hierarchical clustering.

    Parameters
    ----------
    entries : Sequence[BankEntry[CandidateT]]
        Bank entries to compare.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute pairwise distances.
    distance_workspace : IndexedDistanceWorkspace[CandidateT] | None, default=None
        Optional operation-local distance source aligned to ``entries``.

    Returns
    -------
    tuple[float, ...]
        Condensed pairwise distance vector suitable for SciPy linkage.

    Raises
    ------
    ValueError
        If ``distance_workspace`` is not aligned to ``entries``.
    """
    if (
        distance_workspace is not None
        and not distance_workspace.is_aligned_with_entries(entries)
    ):
        msg = "distance_workspace must align with clustering entries"
        raise ValueError(msg)

    distances: list[float] = []
    for left_index, left_entry in enumerate(entries[:-1]):
        for right_index, right_entry in enumerate(
            entries[left_index + 1 :],
            start=left_index + 1,
        ):
            distance = (
                diversity_metric.distance(
                    left_entry.candidate,
                    right_entry.candidate,
                )
                if distance_workspace is None
                else distance_workspace.distance(
                    left_index,
                    right_index,
                )
            )
            distances.append(require_valid_distance(distance))
    return tuple(distances)
