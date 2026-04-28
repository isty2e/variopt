"""Cluster-linkage helpers for CSA clustering state."""

from collections.abc import Sequence

import numpy as np
from scipy.cluster import hierarchy  # pyright: ignore[reportMissingTypeStubs]

from ......distance import require_valid_distance
from ......diversity import DiversityMetric
from ......typevars import CandidateT
from ..bank import BankEntry


def cluster_labels_for_entries(
    *,
    entries: Sequence[BankEntry[CandidateT]],
    cluster_distance: float,
    diversity_metric: DiversityMetric[CandidateT],
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

    Returns
    -------
    tuple[int, ...]
        Linkage-derived cluster labels aligned with ``entries``.
    """
    if len(entries) == 0:
        return ()

    if len(entries) == 1:
        return (1,)

    condensed_distance_values = np.asarray(
        condensed_distances(entries=entries, diversity_metric=diversity_metric),
        dtype=np.float64,
    )
    linkage_matrix = np.asarray(
        hierarchy.linkage(condensed_distance_values),  # pyright: ignore[reportUnknownMemberType]
        dtype=np.float64,
    )
    cluster_labels = np.asarray(
        hierarchy.fcluster(  # pyright: ignore[reportUnknownMemberType]
            linkage_matrix,
            cluster_distance,
            criterion="distance",
        ),
        dtype=np.int64,
    )
    return tuple(int(label) for label in cluster_labels)  # pyright: ignore[reportAny]


def condensed_distances(
    *,
    entries: Sequence[BankEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
) -> tuple[float, ...]:
    """Return condensed pairwise distances for hierarchical clustering.

    Parameters
    ----------
    entries : Sequence[BankEntry[CandidateT]]
        Bank entries to compare.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to compute pairwise distances.

    Returns
    -------
    tuple[float, ...]
        Condensed pairwise distance vector suitable for SciPy linkage.
    """
    distances: list[float] = []
    for left_index, left_entry in enumerate(entries[:-1]):
        for right_entry in entries[left_index + 1 :]:
            distances.append(
                require_valid_distance(
                    diversity_metric.distance(
                        left_entry.candidate,
                        right_entry.candidate,
                    )
                )
            )
    return tuple(distances)
