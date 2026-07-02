"""CSA clustering state definitions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic

import numpy as np

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ......diversity import DiversityMetric
from ......json_types import JSONDict, JSONValue
from ......typevars import CandidateT
from ..bank import BankEntry
from .linkage import cluster_labels_for_entries
from .policy import CSAClusteringPolicy


@dataclass(frozen=True, slots=True)
class CSAClusterUpdateDecision:
    """Resolved cluster-update decision for one scored trial.

    Parameters
    ----------
    remove_index : int
        Bank index to remove if the trial is admitted.
    comparison_index : int
        Bank index used for shaped-score comparison.
    comparison_score : float
        Shaped score at ``comparison_index``.
    """

    remove_index: int
    comparison_index: int
    comparison_score: float


@dataclass(frozen=True, slots=True)
class CSAClusteringState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical state for CSA cluster-aware semantics.

    Parameters
    ----------
    policy : CSAClusteringPolicy
        Clustering policy controlling whether cluster-aware behavior is active.
    cluster_distance : float | None, default=None
        Distance threshold used for hierarchical flat clustering.
    cluster_labels : tuple[int, ...], default=()
        Cluster labels aligned with the current bank snapshot.
    """

    policy: CSAClusteringPolicy
    cluster_distance: float | None = None
    cluster_labels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid clustering states."""
        if not self.policy.enabled:
            if self.cluster_distance is not None or self.cluster_labels:
                msg = "disabled clustering runtime must not carry active state"
                raise ValueError(msg)
            return

        if self.cluster_distance is None:
            if self.cluster_labels:
                msg = "cluster_labels require an initialized cluster_distance"
                raise ValueError(msg)
            return

        if self.cluster_distance < 0.0:
            msg = "cluster_distance must be non-negative"
            raise ValueError(msg)

        if any(label <= 0 for label in self.cluster_labels):
            msg = "cluster_labels must be positive integers"
            raise ValueError(msg)

    @property
    def enabled(self) -> bool:
        """Return whether cluster-aware CSA semantics are active."""
        return self.policy.enabled

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the clustering state.

        Returns
        -------
        JSONDict
            JSON-safe clustering-state snapshot.
        """
        return {
            "cluster_distance": self.cluster_distance,
            "cluster_labels": list(self.cluster_labels),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        policy: CSAClusteringPolicy,
    ) -> "CSAClusteringState[CandidateT]":
        """Build a clustering state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe clustering-state snapshot.
        policy : CSAClusteringPolicy
            Clustering policy that owns the reconstructed state.

        Returns
        -------
        CSAClusteringState[CandidateT]
            Reconstructed clustering state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        cluster_distance = data.get("cluster_distance")
        raw_cluster_labels = data.get("cluster_labels")
        if cluster_distance is not None and not isinstance(cluster_distance, (int, float)):
            msg = "clustering-state snapshot requires numeric cluster_distance or null"
            raise TypeError(msg)
        if not isinstance(raw_cluster_labels, list):
            msg = "clustering-state snapshot requires cluster_labels list"
            raise TypeError(msg)

        cluster_labels: list[int] = []
        for raw_label in raw_cluster_labels:
            if not isinstance(raw_label, int):
                msg = "clustering-state snapshot cluster labels must be integers"
                raise TypeError(msg)
            cluster_labels.append(raw_label)

        return cls(
            policy=policy,
            cluster_distance=(
                None
                if cluster_distance is None
                else float(cluster_distance)
            ),
            cluster_labels=tuple(cluster_labels),
        )

    @property
    def is_initialized(self) -> bool:
        """Return whether clustering state has been initialized."""
        return self.cluster_distance is not None

    def requires_initialization(
        self,
        *,
        entries: Sequence[BankEntry[CandidateT]],
    ) -> bool:
        """Return whether clustering metadata must be built for entries.

        Parameters
        ----------
        entries : Sequence[BankEntry[CandidateT]]
            Current bank entries that clustering metadata should align with.

        Returns
        -------
        bool
            ``True`` when clustering is enabled and the current metadata is not
            aligned with ``entries``.
        """
        return self.enabled and not (
            self.is_initialized and len(self.cluster_labels) == len(entries)
        )

    def reset(self) -> "CSAClusteringState[CandidateT]":
        """Return the initial state implied by this clustering policy."""
        return type(self)(policy=self.policy)

    def ensure_initialized(
        self,
        *,
        entries: Sequence[BankEntry[CandidateT]],
        reference_average_distance: float,
        diversity_metric: DiversityMetric[CandidateT],
    ) -> "CSAClusteringState[CandidateT]":
        """Return a state initialized for one reference-average scale.

        Parameters
        ----------
        entries : Sequence[BankEntry[CandidateT]]
            Current bank entries to cluster.
        reference_average_distance : float
            Reference-average distance used to derive the clustering threshold.
        diversity_metric : DiversityMetric[CandidateT]
            Diversity metric used for pairwise bank-entry distances.

        Returns
        -------
        CSAClusteringState[CandidateT]
            Initialized clustering state, or the original state when
            clustering is disabled or already aligned with ``entries``.
        """
        if not self.requires_initialization(entries=entries):
            return self

        cluster_distance = reference_average_distance / self.policy.cluster_distance_ratio
        return type(self)(
            policy=self.policy,
            cluster_distance=cluster_distance,
            cluster_labels=cluster_labels_for_entries(
                entries=entries,
                cluster_distance=cluster_distance,
                diversity_metric=diversity_metric,
            ),
        )

    def recluster(
        self,
        *,
        entries: Sequence[BankEntry[CandidateT]],
        diversity_metric: DiversityMetric[CandidateT],
    ) -> "CSAClusteringState[CandidateT]":
        """Return a state with cluster labels rebuilt for one bank snapshot.

        Parameters
        ----------
        entries : Sequence[BankEntry[CandidateT]]
            Current bank entries to recluster.
        diversity_metric : DiversityMetric[CandidateT]
            Diversity metric used for pairwise bank-entry distances.

        Returns
        -------
        CSAClusteringState[CandidateT]
            Reclustering result, or the original state when clustering is
            disabled or uninitialized.
        """
        if not self.enabled or self.cluster_distance is None:
            return self

        return type(self)(
            policy=self.policy,
            cluster_distance=self.cluster_distance,
            cluster_labels=cluster_labels_for_entries(
                entries=entries,
                cluster_distance=self.cluster_distance,
                diversity_metric=diversity_metric,
            ),
        )

    def remove_top_cutoff(
        self,
        *,
        distance_cutoff: float,
    ) -> float:
        """Return the cutoff used by legacy remove-top gating.

        Parameters
        ----------
        distance_cutoff : float
            Active CSA distance cutoff.

        Returns
        -------
        float
            Remove-top cutoff after applying the clustering ratio when enabled.
        """
        if not self.enabled:
            return distance_cutoff

        return distance_cutoff * self.policy.cluster_cutoff_ratio

    def should_attempt_cluster_update(
        self,
        *,
        nearest_distance: float,
        distance_cutoff: float,
    ) -> bool:
        """Return whether legacy cluster-update gating is satisfied.

        Parameters
        ----------
        nearest_distance : float
            Distance from the trial candidate to the nearest bank entry.
        distance_cutoff : float
            Active CSA distance cutoff.

        Returns
        -------
        bool
            ``True`` when cluster-update gating is satisfied.
        """
        if not self.enabled:
            return False

        return (
            nearest_distance >= distance_cutoff
            and nearest_distance < self.remove_top_cutoff(distance_cutoff=distance_cutoff)
        )

    def select_cluster_update(
        self,
        *,
        shaped_scores: Sequence[float],
        nearest_index: int,
    ) -> CSAClusterUpdateDecision | None:
        """Return the legacy cluster-update comparison and removal targets.

        Parameters
        ----------
        shaped_scores : Sequence[float]
            Shaped scores aligned with the current bank snapshot.
        nearest_index : int
            Index of the nearest bank entry to the trial candidate.

        Returns
        -------
        CSAClusterUpdateDecision | None
            Resolved comparison/removal decision, or ``None`` when clustering is
            disabled or uninitialized.

        Raises
        ------
        ValueError
            If ``shaped_scores`` is not aligned with ``cluster_labels``.
        """
        if not self.enabled or self.cluster_distance is None:
            return None

        if len(shaped_scores) != len(self.cluster_labels):
            msg = "shaped_scores must align with cluster_labels"
            raise ValueError(msg)

        if len(self.cluster_labels) == 0:
            return None

        current_cluster = self.cluster_labels[nearest_index]
        current_cluster_indices = tuple(
            index
            for index, label in enumerate(self.cluster_labels)
            if label == current_cluster
        )
        comparison_index = max(
            current_cluster_indices,
            key=shaped_scores.__getitem__,
        )
        comparison_score = shaped_scores[comparison_index]

        if self.policy.update_mode == "largest_cluster":
            cluster_counts = np.bincount(
                np.asarray(self.cluster_labels, dtype=np.int64),
            )
            largest_cluster = int(np.argmax(cluster_counts[1:])) + 1
            removal_cluster_indices = tuple(
                index
                for index, label in enumerate(self.cluster_labels)
                if label == largest_cluster
            )
            remove_index = max(
                removal_cluster_indices,
                key=shaped_scores.__getitem__,
            )
        else:
            remove_index = comparison_index

        return CSAClusterUpdateDecision(
            remove_index=remove_index,
            comparison_index=comparison_index,
            comparison_score=comparison_score,
        )

    def register_admission(
        self,
        *,
        admitted_index: int,
        nearest_index: int,
        nearest_distance: float,
        appended: bool,
    ) -> "CSAClusteringState[CandidateT]":
        """Return a state updated for one in-batch bank admission.

        Parameters
        ----------
        admitted_index : int
            Index at which the new candidate was admitted.
        nearest_index : int
            Index of the nearest existing bank entry.
        nearest_distance : float
            Distance from the admitted candidate to the nearest bank entry.
        appended : bool
            Whether the admission appended a new entry instead of replacing an
            existing one.

        Returns
        -------
        CSAClusteringState[CandidateT]
            Updated clustering state reflecting the admission.

        Raises
        ------
        ValueError
            If the admission indices are inconsistent with ``cluster_labels``.
        """
        if not self.enabled or self.cluster_distance is None:
            return self

        labels = list(self.cluster_labels)
        if appended:
            if admitted_index != len(labels):
                msg = "appended admissions must target the next cluster-label slot"
                raise ValueError(msg)
        elif admitted_index >= len(labels):
            msg = "admitted_index must reference an existing cluster label"
            raise ValueError(msg)

        if nearest_distance > self.cluster_distance:
            admitted_label = max(labels, default=0) + 1
        elif appended:
            admitted_label = 1 if not labels else labels[nearest_index]
        else:
            admitted_label = labels[nearest_index]

        if appended:
            labels.append(admitted_label)
        else:
            labels[admitted_index] = admitted_label

        return type(self)(
            policy=self.policy,
            cluster_distance=self.cluster_distance,
            cluster_labels=tuple(labels),
        )
