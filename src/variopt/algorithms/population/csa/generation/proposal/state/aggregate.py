"""Aggregate reducer state for CSA proposal adaptation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from typing_extensions import Self

from .......json_types import JSONDict, JSONValue, require_json_int
from .......spaces import LeafPath
from ..policy import CSAProposalPolicy
from .attribution import NumericSubspaceDisplacement, ProposalAttribution
from .stats import (
    ProposalFamilyStat,
    ProposalLeafStat,
    ProposalNumericSubspaceCovarianceStat,
)


@dataclass(frozen=True, slots=True)
class CSAProposalState:
    """Canonical adaptive-memory state for proposal-side CSA improvements.

    Parameters
    ----------
    policy : CSAProposalPolicy
        Proposal adaptation policy that controls whether and how statistics are
        updated.
    pending_attributions : tuple[ProposalAttribution, ...], default=()
        Attributions recorded at proposal time and waiting to be matched to
        observations.
    family_stats : tuple[ProposalFamilyStat, ...], default=()
        Accumulated family-level reward statistics.
    leaf_stats : tuple[ProposalLeafStat, ...], default=()
        Accumulated mutated-leaf reward statistics.
    local_displacement_leaf_stats : tuple[ProposalLeafStat, ...], default=()
        Accumulated local-displacement reward statistics inferred after local
        search.
    numeric_covariance_stats : tuple[ProposalNumericSubspaceCovarianceStat, ...], default=()
        Accumulated successful numeric displacement moments keyed by leaf
        families.
    update_index : int, default=0
        Monotone reducer update counter used for lazy decay.
    """

    policy: CSAProposalPolicy
    pending_attributions: tuple[ProposalAttribution, ...] = ()
    family_stats: tuple[ProposalFamilyStat, ...] = ()
    leaf_stats: tuple[ProposalLeafStat, ...] = ()
    local_displacement_leaf_stats: tuple[ProposalLeafStat, ...] = ()
    numeric_covariance_stats: tuple[ProposalNumericSubspaceCovarianceStat, ...] = ()
    update_index: int = 0

    @classmethod
    def from_policy(cls, policy: CSAProposalPolicy) -> Self:
        """Return one canonical proposal state from boundary policy input.

        Parameters
        ----------
        policy : CSAProposalPolicy
            Proposal adaptation policy to bind to the initial state.

        Returns
        -------
        Self
            Fresh proposal state with empty adaptive-memory statistics.
        """
        return cls(policy=policy)

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the proposal state.

        Returns
        -------
        JSONDict
            JSON-safe proposal-state snapshot.

        Raises
        ------
        ValueError
            If pending proposal attributions are still present.
        """
        if self.pending_attributions:
            msg = "proposal-state checkpoints require an empty pending attribution queue"
            raise ValueError(msg)

        return {
            "pending_attributions": [],
            "family_stats": [
                family_stat.to_dict()
                for family_stat in self.family_stats
            ],
            "leaf_stats": [
                leaf_stat.to_dict()
                for leaf_stat in self.leaf_stats
            ],
            "local_displacement_leaf_stats": [
                leaf_stat.to_dict()
                for leaf_stat in self.local_displacement_leaf_stats
            ],
            "numeric_covariance_stats": [
                covariance_stat.to_dict()
                for covariance_stat in self.numeric_covariance_stats
            ],
            "update_index": self.update_index,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        policy: CSAProposalPolicy,
    ) -> Self:
        """Build a proposal state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe proposal-state snapshot.
        policy : CSAProposalPolicy
            Proposal policy that owns the reconstructed state.

        Returns
        -------
        Self
            Reconstructed proposal state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        ValueError
            If the snapshot attempts to restore pending attributions.
        """
        raw_pending_attributions = data.get("pending_attributions")
        raw_family_stats = data.get("family_stats")
        raw_leaf_stats = data.get("leaf_stats")
        raw_local_displacement_leaf_stats = data.get("local_displacement_leaf_stats")
        raw_numeric_covariance_stats = data.get("numeric_covariance_stats")
        update_index = require_json_int(
            data.get("update_index"),
            field_name="update_index",
        )
        if not isinstance(raw_pending_attributions, list):
            msg = "proposal-state snapshot requires pending_attributions list"
            raise TypeError(msg)
        if len(raw_pending_attributions) != 0:
            msg = "proposal-state checkpoints require an empty pending attribution queue"
            raise ValueError(msg)
        if not isinstance(raw_family_stats, list):
            msg = "proposal-state snapshot requires family_stats list"
            raise TypeError(msg)
        if not isinstance(raw_leaf_stats, list):
            msg = "proposal-state snapshot requires leaf_stats list"
            raise TypeError(msg)
        if not isinstance(raw_local_displacement_leaf_stats, list):
            msg = "proposal-state snapshot requires local_displacement_leaf_stats list"
            raise TypeError(msg)
        if not isinstance(raw_numeric_covariance_stats, list):
            msg = "proposal-state snapshot requires numeric_covariance_stats list"
            raise TypeError(msg)
        family_stats: list[ProposalFamilyStat] = []
        for raw_family_stat in raw_family_stats:
            if not isinstance(raw_family_stat, dict):
                msg = "proposal-state family_stats entries must be mappings"
                raise TypeError(msg)
            family_stats.append(ProposalFamilyStat.from_dict(raw_family_stat))

        leaf_stats: list[ProposalLeafStat] = []
        for raw_leaf_stat in raw_leaf_stats:
            if not isinstance(raw_leaf_stat, dict):
                msg = "proposal-state leaf_stats entries must be mappings"
                raise TypeError(msg)
            leaf_stats.append(ProposalLeafStat.from_dict(raw_leaf_stat))

        local_displacement_leaf_stats: list[ProposalLeafStat] = []
        for raw_leaf_stat in raw_local_displacement_leaf_stats:
            if not isinstance(raw_leaf_stat, dict):
                msg = "proposal-state local displacement leaf stats entries must be mappings"
                raise TypeError(msg)
            local_displacement_leaf_stats.append(ProposalLeafStat.from_dict(raw_leaf_stat))

        numeric_covariance_stats: list[ProposalNumericSubspaceCovarianceStat] = []
        for raw_covariance_stat in raw_numeric_covariance_stats:
            if not isinstance(raw_covariance_stat, dict):
                msg = "proposal-state covariance-stat entries must be mappings"
                raise TypeError(msg)
            numeric_covariance_stats.append(
                ProposalNumericSubspaceCovarianceStat.from_dict(raw_covariance_stat),
            )

        return cls(
            policy=policy,
            pending_attributions=(),
            family_stats=tuple(family_stats),
            leaf_stats=tuple(leaf_stats),
            local_displacement_leaf_stats=tuple(local_displacement_leaf_stats),
            numeric_covariance_stats=tuple(numeric_covariance_stats),
            update_index=update_index,
        )

    def get_pending_attribution(self, proposal_id: str) -> ProposalAttribution | None:
        """Return the pending attribution matching one proposal id, if present.

        Parameters
        ----------
        proposal_id : str
            Proposal identifier to look up.

        Returns
        -------
        ProposalAttribution | None
            Matching pending attribution, or ``None`` when the proposal has no
            recorded pending attribution.
        """
        for attribution in self.pending_attributions:
            if attribution.proposal_id == proposal_id:
                return attribution

        return None

    def register_pending_attribution(self, attribution: ProposalAttribution) -> Self:
        """Return a state with one additional pending proposal attribution.

        Parameters
        ----------
        attribution : ProposalAttribution
            Proposal-side attribution to register.

        Returns
        -------
        Self
            State with ``attribution`` appended to the pending queue.

        Raises
        ------
        ValueError
            If another pending attribution already uses the same proposal id.
        """
        if self.get_pending_attribution(attribution.proposal_id) is not None:
            msg = "pending proposal attributions must have distinct proposal ids"
            raise ValueError(msg)

        return replace(
            self,
            pending_attributions=self.pending_attributions + (attribution,),
        )

    def consume_pending_attribution(
        self,
        proposal_id: str,
    ) -> tuple[ProposalAttribution | None, Self]:
        """Return one pending attribution together with the reduced state.

        Parameters
        ----------
        proposal_id : str
            Proposal identifier to consume from the pending queue.

        Returns
        -------
        tuple[ProposalAttribution | None, Self]
            Matched attribution, if present, together with the state after that
            attribution has been removed from the pending queue.
        """
        matched_attribution: ProposalAttribution | None = None
        remaining_attributions: list[ProposalAttribution] = []
        for attribution in self.pending_attributions:
            if attribution.proposal_id == proposal_id and matched_attribution is None:
                matched_attribution = attribution
                continue

            remaining_attributions.append(attribution)

        return matched_attribution, replace(
            self,
            pending_attributions=tuple(remaining_attributions),
        )

    def family_stat_for_key(self, family_key: str) -> ProposalFamilyStat | None:
        """Return the accumulated family stat for one proposal family key.

        Parameters
        ----------
        family_key : str
            Canonical proposal family identifier.

        Returns
        -------
        ProposalFamilyStat | None
            Matching family statistic, or ``None`` when the family has not yet
            been observed.
        """
        for family_stat in self.family_stats:
            if family_stat.family_key == family_key:
                return family_stat

        return None

    def record_score_improvement(
        self,
        *,
        family_key: str | None,
        leaf_paths: Sequence[LeafPath],
        local_displacement_leaf_paths: Sequence[LeafPath] = (),
        numeric_displacement: NumericSubspaceDisplacement | None = None,
        score_improvement: float,
    ) -> "CSAProposalState":
        """Return a state with one additional proposal-side reward update.

        Parameters
        ----------
        family_key : str | None
            Proposal family key associated with the observation, if any.
        leaf_paths : Sequence[LeafPath]
            Mutated structured leaf paths credited to the proposal.
        local_displacement_leaf_paths : Sequence[LeafPath], default=()
            Additional leaf paths credited through local post-processing.
        numeric_displacement : NumericSubspaceDisplacement | None, default=None
            Successful numeric displacement inferred from local post-processing.
        score_improvement : float
            Improvement credit, conventionally ``source_score - observed_score``.

        Returns
        -------
        CSAProposalState
            Updated proposal state with all applicable adaptive statistics
            incremented.
        """
        if (
            family_key is None
            and len(leaf_paths) == 0
            and len(local_displacement_leaf_paths) == 0
            and numeric_displacement is None
        ):
            return self

        next_update_index = self.update_index + 1
        next_state: CSAProposalState = self
        if family_key is not None:
            next_state = next_state.record_family_score_improvement(
                family_key,
                score_improvement=score_improvement,
                next_update_index=next_update_index,
            )

        if len(leaf_paths) > 0:
            next_state = next_state.record_leaf_score_improvement(
                leaf_paths,
                score_improvement=score_improvement,
                next_update_index=next_update_index,
            )

        if len(local_displacement_leaf_paths) > 0:
            next_state = next_state.record_local_displacement_score_improvement(
                local_displacement_leaf_paths,
                score_improvement=score_improvement,
                next_update_index=next_update_index,
            )

        if numeric_displacement is not None:
            next_state = next_state.record_numeric_covariance_displacement(
                numeric_displacement,
                score_improvement=score_improvement,
                next_update_index=next_update_index,
            )

        return replace(next_state, update_index=next_update_index)

    def covariance_stat_for_leaf_paths(
        self,
        leaf_paths: Sequence[LeafPath],
    ) -> ProposalNumericSubspaceCovarianceStat | None:
        """Return the covariance stat matching one structured leaf family.

        Parameters
        ----------
        leaf_paths : Sequence[LeafPath]
            Structured leaf family key to look up.

        Returns
        -------
        ProposalNumericSubspaceCovarianceStat | None
            Matching covariance statistic, or ``None`` when no successful
            displacement has been recorded for that family.
        """
        normalized_leaf_paths = tuple(tuple(path) for path in leaf_paths)
        for covariance_stat in self.numeric_covariance_stats:
            if covariance_stat.leaf_paths == normalized_leaf_paths:
                return covariance_stat
        return None

    def record_family_score_improvement(
        self,
        family_key: str,
        *,
        score_improvement: float,
        next_update_index: int,
    ) -> Self:
        """Return a state with accumulated score-improvement family statistics.

        Parameters
        ----------
        family_key : str
            Proposal family key receiving the reward update.
        score_improvement : float
            Improvement credit to accumulate.
        next_update_index : int
            Reducer update index associated with this observation.

        Returns
        -------
        Self
            State with updated family-level reward statistics.
        """
        next_family_stats_by_key = {
            family_stat.family_key: family_stat for family_stat in self.family_stats
        }
        ordered_keys: list[str] = list(next_family_stats_by_key)

        current_stat = next_family_stats_by_key.get(family_key)
        if current_stat is None:
            current_stat = ProposalFamilyStat(
                family_key=family_key,
                last_update_index=next_update_index,
            )
            ordered_keys.append(family_key)

        next_family_stats_by_key[family_key] = current_stat.record_score_improvement(
            score_improvement,
            current_update_index=next_update_index,
            score_decay=self.policy.score_decay,
        )

        return replace(
            self,
            family_stats=tuple(
                next_family_stats_by_key[key]
                for key in ordered_keys
            ),
        )

    def record_leaf_score_improvement(
        self,
        leaf_paths: Sequence[LeafPath],
        *,
        score_improvement: float,
        next_update_index: int,
    ) -> Self:
        """Return a state with accumulated per-leaf outcome statistics.

        Parameters
        ----------
        leaf_paths : Sequence[LeafPath]
            Mutated leaf paths receiving the reward update.
        score_improvement : float
            Improvement credit to accumulate.
        next_update_index : int
            Reducer update index associated with this observation.

        Returns
        -------
        Self
            State with updated per-leaf reward statistics.
        """
        if len(leaf_paths) == 0:
            return self

        next_leaf_stats_by_path = {
            leaf_stat.path: leaf_stat for leaf_stat in self.leaf_stats
        }
        ordered_paths: list[LeafPath] = list(next_leaf_stats_by_path)

        for path in leaf_paths:
            normalized_path = tuple(path)
            current_stat = next_leaf_stats_by_path.get(normalized_path)
            if current_stat is None:
                current_stat = ProposalLeafStat(
                    path=normalized_path,
                    last_update_index=next_update_index,
                )
                ordered_paths.append(normalized_path)

            next_leaf_stats_by_path[normalized_path] = current_stat.record_outcome(
                score_improvement,
                current_update_index=next_update_index,
                score_decay=self.policy.score_decay,
            )

        return replace(
            self,
            leaf_stats=tuple(
                next_leaf_stats_by_path[path]
                for path in ordered_paths
            ),
        )

    def record_local_displacement_score_improvement(
        self,
        leaf_paths: Sequence[LeafPath],
        *,
        score_improvement: float,
        next_update_index: int,
    ) -> Self:
        """Return a state with accumulated local-displacement leaf outcomes.

        Parameters
        ----------
        leaf_paths : Sequence[LeafPath]
            Leaf paths changed by local post-processing.
        score_improvement : float
            Improvement credit to accumulate.
        next_update_index : int
            Reducer update index associated with this observation.

        Returns
        -------
        Self
            State with updated local-displacement reward statistics.
        """
        if len(leaf_paths) == 0:
            return self

        next_leaf_stats_by_path = {
            leaf_stat.path: leaf_stat for leaf_stat in self.local_displacement_leaf_stats
        }
        ordered_paths: list[LeafPath] = list(next_leaf_stats_by_path)

        for path in leaf_paths:
            normalized_path = tuple(path)
            current_stat = next_leaf_stats_by_path.get(normalized_path)
            if current_stat is None:
                current_stat = ProposalLeafStat(
                    path=normalized_path,
                    last_update_index=next_update_index,
                )
                ordered_paths.append(normalized_path)

            next_leaf_stats_by_path[normalized_path] = current_stat.record_outcome(
                score_improvement,
                current_update_index=next_update_index,
                score_decay=self.policy.score_decay,
            )

        return replace(
            self,
            local_displacement_leaf_stats=tuple(
                next_leaf_stats_by_path[path]
                for path in ordered_paths
            ),
        )

    def record_numeric_covariance_displacement(
        self,
        numeric_displacement: NumericSubspaceDisplacement,
        *,
        score_improvement: float,
        next_update_index: int,
    ) -> Self:
        """Return a state with one additional numeric covariance update.

        Parameters
        ----------
        numeric_displacement : NumericSubspaceDisplacement
            Successful numeric displacement inferred from a structured local
            search step.
        score_improvement : float
            Improvement credit associated with the displacement.
        next_update_index : int
            Reducer update index associated with this observation.

        Returns
        -------
        Self
            State with updated numeric covariance statistics when the policy and
            score improvement permit it, otherwise the original state.
        """
        if score_improvement <= 0.0 or self.policy.numeric_covariance_strength <= 0.0:
            return self

        next_covariance_stats_by_paths = {
            covariance_stat.leaf_paths: covariance_stat
            for covariance_stat in self.numeric_covariance_stats
        }
        ordered_leaf_paths: list[tuple[LeafPath, ...]] = list(next_covariance_stats_by_paths)
        current_stat = next_covariance_stats_by_paths.get(
            numeric_displacement.leaf_paths,
        )
        if current_stat is None:
            current_stat = ProposalNumericSubspaceCovarianceStat(
                leaf_paths=numeric_displacement.leaf_paths,
                discounted_displacement_sum=tuple(
                    0.0 for _ in numeric_displacement.displacement_coordinates
                ),
                discounted_outer_product_sum=tuple(
                    tuple(
                        0.0
                        for _ in numeric_displacement.displacement_coordinates
                    )
                    for _ in numeric_displacement.displacement_coordinates
                ),
                last_update_index=next_update_index,
            )
            ordered_leaf_paths.append(numeric_displacement.leaf_paths)

        next_covariance_stats_by_paths[numeric_displacement.leaf_paths] = (
            current_stat.record_successful_displacement(
                numeric_displacement,
                current_update_index=next_update_index,
                score_decay=self.policy.score_decay,
            )
        )
        return replace(
            self,
            numeric_covariance_stats=tuple(
                next_covariance_stats_by_paths[leaf_paths_key]
                for leaf_paths_key in ordered_leaf_paths
            ),
        )
