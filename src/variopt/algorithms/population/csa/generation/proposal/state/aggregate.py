"""Aggregate reducer state for CSA proposal adaptation."""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace

from typing_extensions import Self

from .......json_types import (
    JSONDict,
    JSONValue,
    require_json_field,
    require_json_int,
    require_json_list,
    require_json_mapping,
)
from .......spaces import LeafPath
from ..policy import CSAProposalPolicy
from .attribution import ProposalProvenance
from .generation_evidence import ProposalGenerationAdaptationEvidence
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
    pending_attributions : tuple[ProposalProvenance, ...], default=()
        Explicit adaptive or non-adaptive provenance recorded at proposal time
        and waiting to be matched to terminal proposal feedback.
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
    pending_attributions: tuple[ProposalProvenance, ...] = ()
    family_stats: tuple[ProposalFamilyStat, ...] = ()
    leaf_stats: tuple[ProposalLeafStat, ...] = ()
    local_displacement_leaf_stats: tuple[ProposalLeafStat, ...] = ()
    numeric_covariance_stats: tuple[ProposalNumericSubspaceCovarianceStat, ...] = ()
    update_index: int = 0

    def __post_init__(self) -> None:
        """Normalize tuple state and reject misaligned adaptive memory."""
        object.__setattr__(
            self, "pending_attributions", tuple(self.pending_attributions)
        )
        object.__setattr__(self, "family_stats", tuple(self.family_stats))
        object.__setattr__(self, "leaf_stats", tuple(self.leaf_stats))
        object.__setattr__(
            self,
            "local_displacement_leaf_stats",
            tuple(self.local_displacement_leaf_stats),
        )
        object.__setattr__(
            self,
            "numeric_covariance_stats",
            tuple(self.numeric_covariance_stats),
        )
        if type(self.update_index) is not int:
            msg = "update_index must be an int"
            raise TypeError(msg)
        if self.update_index < 0:
            msg = "update_index must be non-negative"
            raise ValueError(msg)

        proposal_ids = tuple(
            attribution.proposal_id for attribution in self.pending_attributions
        )
        if len(set(proposal_ids)) != len(proposal_ids):
            msg = "pending proposal attributions must use distinct proposal ids"
            raise ValueError(msg)
        family_keys = tuple(stat.family_key for stat in self.family_stats)
        if len(set(family_keys)) != len(family_keys):
            msg = "family_stats must use distinct family keys"
            raise ValueError(msg)
        leaf_paths = tuple(stat.path for stat in self.leaf_stats)
        if len(set(leaf_paths)) != len(leaf_paths):
            msg = "leaf_stats must use distinct paths"
            raise ValueError(msg)
        local_leaf_paths = tuple(
            stat.path for stat in self.local_displacement_leaf_stats
        )
        if len(set(local_leaf_paths)) != len(local_leaf_paths):
            msg = "local_displacement_leaf_stats must use distinct paths"
            raise ValueError(msg)
        covariance_paths = tuple(
            stat.leaf_paths for stat in self.numeric_covariance_stats
        )
        if len(set(covariance_paths)) != len(covariance_paths):
            msg = "numeric_covariance_stats must use distinct leaf path families"
            raise ValueError(msg)

        all_stats = (
            *self.family_stats,
            *self.leaf_stats,
            *self.local_displacement_leaf_stats,
            *self.numeric_covariance_stats,
        )
        if any(stat.last_update_index > self.update_index for stat in all_stats):
            msg = "proposal stat update indices must not exceed state update_index"
            raise ValueError(msg)

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
            msg = (
                "proposal-state checkpoints require an empty pending attribution queue"
            )
            raise ValueError(msg)

        return {
            "pending_attributions": [],
            "family_stats": [
                family_stat.to_dict() for family_stat in self.family_stats
            ],
            "leaf_stats": [leaf_stat.to_dict() for leaf_stat in self.leaf_stats],
            "local_displacement_leaf_stats": [
                leaf_stat.to_dict() for leaf_stat in self.local_displacement_leaf_stats
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
        raw_pending_attributions = require_json_list(
            require_json_field(data, "pending_attributions"),
            field_name="pending_attributions",
        )
        if len(raw_pending_attributions) != 0:
            msg = (
                "proposal-state checkpoints require an empty pending attribution queue"
            )
            raise ValueError(msg)
        raw_family_stats = require_json_list(
            require_json_field(data, "family_stats"),
            field_name="family_stats",
        )
        raw_leaf_stats = require_json_list(
            require_json_field(data, "leaf_stats"),
            field_name="leaf_stats",
        )
        raw_local_displacement_leaf_stats = require_json_list(
            require_json_field(data, "local_displacement_leaf_stats"),
            field_name="local_displacement_leaf_stats",
        )
        raw_numeric_covariance_stats = require_json_list(
            require_json_field(data, "numeric_covariance_stats"),
            field_name="numeric_covariance_stats",
        )
        update_index = require_json_int(
            require_json_field(data, "update_index"),
            field_name="update_index",
        )
        family_stats: list[ProposalFamilyStat] = []
        for raw_position, raw_family_stat in enumerate(raw_family_stats):
            family_stats.append(
                ProposalFamilyStat.from_dict(
                    require_json_mapping(
                        raw_family_stat,
                        field_name=f"family_stats[{raw_position}]",
                    ),
                ),
            )

        leaf_stats: list[ProposalLeafStat] = []
        for raw_position, raw_leaf_stat in enumerate(raw_leaf_stats):
            leaf_stats.append(
                ProposalLeafStat.from_dict(
                    require_json_mapping(
                        raw_leaf_stat,
                        field_name=f"leaf_stats[{raw_position}]",
                    ),
                ),
            )

        local_displacement_leaf_stats: list[ProposalLeafStat] = []
        for raw_position, raw_leaf_stat in enumerate(raw_local_displacement_leaf_stats):
            local_displacement_leaf_stats.append(
                ProposalLeafStat.from_dict(
                    require_json_mapping(
                        raw_leaf_stat,
                        field_name=f"local_displacement_leaf_stats[{raw_position}]",
                    ),
                ),
            )

        numeric_covariance_stats: list[ProposalNumericSubspaceCovarianceStat] = []
        for raw_position, raw_covariance_stat in enumerate(
            raw_numeric_covariance_stats
        ):
            numeric_covariance_stats.append(
                ProposalNumericSubspaceCovarianceStat.from_dict(
                    require_json_mapping(
                        raw_covariance_stat,
                        field_name=f"numeric_covariance_stats[{raw_position}]",
                    ),
                ),
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

    def get_pending_attribution(self, proposal_id: str) -> ProposalProvenance | None:
        """Return the pending provenance matching one proposal id, if present.

        Parameters
        ----------
        proposal_id : str
            Proposal identifier to look up.

        Returns
        -------
        ProposalProvenance | None
            Matching pending provenance, or ``None`` when the proposal has no
            recorded pending provenance.
        """
        for attribution in self.pending_attributions:
            if attribution.proposal_id == proposal_id:
                return attribution

        return None

    def register_pending_attribution(self, attribution: ProposalProvenance) -> Self:
        """Return a state with one additional pending proposal provenance.

        Parameters
        ----------
        attribution : ProposalProvenance
            Proposal-side provenance to register.

        Returns
        -------
        Self
            State with ``attribution`` appended to the pending queue.

        Raises
        ------
        ValueError
            If another pending attribution already uses the same proposal id.
        """
        return self.register_pending_attributions((attribution,))

    def register_pending_attributions(
        self,
        attributions: Sequence[ProposalProvenance],
    ) -> Self:
        """Return a state with one batch of distinct pending provenance.

        Parameters
        ----------
        attributions : Sequence[ProposalProvenance]
            Provenance records to append in issue order.

        Returns
        -------
        Self
            State with all supplied provenance appended once.

        Raises
        ------
        ValueError
            If a proposal id repeats in existing or supplied provenance.
        """
        attribution_tuple = tuple(attributions)
        if len(attribution_tuple) == 0:
            return self

        existing_proposal_ids = {
            attribution.proposal_id for attribution in self.pending_attributions
        }
        new_proposal_ids: set[str] = set()
        for attribution in attribution_tuple:
            proposal_id = attribution.proposal_id
            if proposal_id in existing_proposal_ids or proposal_id in new_proposal_ids:
                msg = "pending proposal attributions must have distinct proposal ids"
                raise ValueError(msg)
            new_proposal_ids.add(proposal_id)

        return replace(
            self,
            pending_attributions=self.pending_attributions + attribution_tuple,
        )

    def consume_pending_attribution(
        self,
        proposal_id: str,
    ) -> tuple[ProposalProvenance | None, Self]:
        """Return one pending provenance together with the reduced state.

        Parameters
        ----------
        proposal_id : str
            Proposal identifier to consume from the pending queue.

        Returns
        -------
        tuple[ProposalProvenance | None, Self]
            Matched provenance, if present, together with the state after that
            provenance has been removed from the pending queue.
        """
        matched_attributions, next_state = self.consume_pending_attributions(
            (proposal_id,),
            require_all=False,
        )
        return matched_attributions[0], next_state

    def consume_pending_attributions(
        self,
        proposal_ids: Sequence[str],
        *,
        require_all: bool = True,
    ) -> tuple[tuple[ProposalProvenance | None, ...], Self]:
        """Consume a proposal-id batch in one provenance scan.

        Parameters
        ----------
        proposal_ids : Sequence[str]
            Proposal identifiers to consume in result order.
        require_all : bool, default=True
            Whether every identifier must have pending provenance.

        Returns
        -------
        tuple[tuple[ProposalProvenance | None, ...], Self]
            Provenance aligned with ``proposal_ids`` and the reduced state.

        Raises
        ------
        ValueError
            If identifiers repeat or required provenance is missing.
        """
        proposal_id_tuple = tuple(proposal_ids)
        proposal_id_set = set(proposal_id_tuple)
        if len(proposal_id_set) != len(proposal_id_tuple):
            msg = "consumed proposal ids must be distinct"
            raise ValueError(msg)
        if len(proposal_id_tuple) == 0:
            return (), self

        attribution_by_id = {
            attribution.proposal_id: attribution
            for attribution in self.pending_attributions
        }
        matched_attributions = tuple(
            attribution_by_id.get(proposal_id) for proposal_id in proposal_id_tuple
        )
        if require_all and any(
            attribution is None for attribution in matched_attributions
        ):
            msg = "proposal evaluation has no pending adaptation provenance"
            raise ValueError(msg)

        return matched_attributions, replace(
            self,
            pending_attributions=tuple(
                attribution
                for attribution in self.pending_attributions
                if attribution.proposal_id not in proposal_id_set
            ),
        )

    def remove_pending_attributions(self, proposal_ids: AbstractSet[str]) -> Self:
        """Return a state with selected pending attributions discarded.

        Parameters
        ----------
        proposal_ids : collections.abc.Set[str]
            Proposal identifiers whose in-flight attributions should be
            removed without recording adaptation evidence.

        Returns
        -------
        Self
            Proposal state with matching pending attributions removed.
        """
        if not proposal_ids or len(self.pending_attributions) == 0:
            return self

        return replace(
            self,
            pending_attributions=tuple(
                attribution
                for attribution in self.pending_attributions
                if attribution.proposal_id not in proposal_ids
            ),
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

    def record_generation_evidence(
        self,
        batch: ProposalGenerationAdaptationEvidence,
    ) -> "CSAProposalState":
        """Return state updated from one canonical completed-generation batch.

        Parameters
        ----------
        batch : ProposalGenerationAdaptationEvidence
            Scale-invariant family, leaf, and displacement evidence.

        Returns
        -------
        CSAProposalState
            State after one generation-level decay and evidence update.
        """
        if not self.policy.enabled:
            return self

        next_update_index = self.update_index + 1
        next_family_stats_by_key = {
            family_stat.family_key: family_stat for family_stat in self.family_stats
        }
        ordered_family_keys: list[str] = list(next_family_stats_by_key)
        for summary in batch.family_summaries:
            current_stat = next_family_stats_by_key.get(summary.family_key)
            if current_stat is None:
                current_stat = ProposalFamilyStat(
                    family_key=summary.family_key,
                    last_update_index=next_update_index,
                )
                ordered_family_keys.append(summary.family_key)
            next_family_stats_by_key[summary.family_key] = (
                current_stat.record_generation(
                    summary,
                    current_update_index=next_update_index,
                    adaptation_decay=self.policy.adaptation_decay,
                )
            )

        next_leaf_stats_by_path = {
            leaf_stat.path: leaf_stat for leaf_stat in self.leaf_stats
        }
        ordered_leaf_paths: list[LeafPath] = list(next_leaf_stats_by_path)
        for summary in batch.mutation_leaf_summaries:
            current_stat = next_leaf_stats_by_path.get(summary.path)
            if current_stat is None:
                current_stat = ProposalLeafStat(
                    path=summary.path,
                    last_update_index=next_update_index,
                )
                ordered_leaf_paths.append(summary.path)
            next_leaf_stats_by_path[summary.path] = current_stat.record_generation(
                summary,
                current_update_index=next_update_index,
                adaptation_decay=self.policy.adaptation_decay,
            )

        next_local_leaf_stats_by_path = {
            leaf_stat.path: leaf_stat
            for leaf_stat in self.local_displacement_leaf_stats
        }
        ordered_local_leaf_paths: list[LeafPath] = list(next_local_leaf_stats_by_path)
        for summary in batch.local_displacement_leaf_summaries:
            current_stat = next_local_leaf_stats_by_path.get(summary.path)
            if current_stat is None:
                current_stat = ProposalLeafStat(
                    path=summary.path,
                    last_update_index=next_update_index,
                )
                ordered_local_leaf_paths.append(summary.path)
            next_local_leaf_stats_by_path[summary.path] = (
                current_stat.record_generation(
                    summary,
                    current_update_index=next_update_index,
                    adaptation_decay=self.policy.adaptation_decay,
                )
            )

        next_covariance_stats_by_paths = {
            covariance_stat.leaf_paths: covariance_stat
            for covariance_stat in self.numeric_covariance_stats
        }
        ordered_covariance_paths: list[tuple[LeafPath, ...]] = list(
            next_covariance_stats_by_paths
        )
        if self.policy.numeric_covariance_strength > 0.0:
            for numeric_evidence in batch.numeric_displacement_evidence:
                displacement = numeric_evidence.displacement
                current_stat = next_covariance_stats_by_paths.get(
                    displacement.leaf_paths
                )
                if current_stat is None:
                    current_stat = ProposalNumericSubspaceCovarianceStat(
                        leaf_paths=displacement.leaf_paths,
                        discounted_displacement_sum=tuple(
                            0.0 for _ in displacement.displacement_coordinates
                        ),
                        discounted_outer_product_sum=tuple(
                            tuple(0.0 for _ in displacement.displacement_coordinates)
                            for _ in displacement.displacement_coordinates
                        ),
                        last_update_index=next_update_index,
                    )
                    ordered_covariance_paths.append(displacement.leaf_paths)
                next_covariance_stats_by_paths[displacement.leaf_paths] = (
                    current_stat.record_successful_displacement(
                        displacement,
                        survival_efficiency=numeric_evidence.survival_efficiency,
                        current_update_index=next_update_index,
                        adaptation_decay=self.policy.adaptation_decay,
                    )
                )

        return replace(
            self,
            family_stats=tuple(
                next_family_stats_by_key[key] for key in ordered_family_keys
            ),
            leaf_stats=tuple(
                next_leaf_stats_by_path[path] for path in ordered_leaf_paths
            ),
            local_displacement_leaf_stats=tuple(
                next_local_leaf_stats_by_path[path] for path in ordered_local_leaf_paths
            ),
            numeric_covariance_stats=tuple(
                next_covariance_stats_by_paths[paths]
                for paths in ordered_covariance_paths
            ),
            update_index=next_update_index,
        )

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
