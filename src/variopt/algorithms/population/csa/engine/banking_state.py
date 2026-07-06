"""Canonical CSA banking aggregate state."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....json_types import (
    JSONDict,
    JSONValue,
    require_json_field,
    require_json_mapping,
)
from .....typevars import CandidateT
from ..banking.bank import Bank
from ..banking.clustering import CSAClusteringState
from ..banking.clustering.policy import CSAClusteringPolicy
from ..banking.growth import CSABankGrowthState
from ..banking.growth.policy import CSABankGrowthPolicy
from ..banking.reference import ReferenceBank, ReferenceRefreshState


@dataclass(frozen=True, slots=True)
class CSABankingState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical banking aggregate for CSA execution.

    Parameters
    ----------
    bank : Bank[CandidateT]
        Active bank snapshot.
    reference_bank : ReferenceBank[CandidateT]
        Reference-bank snapshot used by selection and refresh logic.
    refresh_state : ReferenceRefreshState[CandidateT] | None
        Optional reference-refresh runtime state.
    growth_state : CSABankGrowthState[CandidateT]
        Adaptive bank-growth runtime state.
    clustering_state : CSAClusteringState[CandidateT]
        Cluster-aware banking runtime state.
    """

    bank: Bank[CandidateT]
    reference_bank: ReferenceBank[CandidateT]
    refresh_state: ReferenceRefreshState[CandidateT] | None
    growth_state: CSABankGrowthState[CandidateT]
    clustering_state: CSAClusteringState[CandidateT]

    def to_dict(
        self,
        *,
        candidate_to_dict: Callable[[CandidateT], JSONValue],
    ) -> JSONDict:
        """Return a JSON-safe mapping for the banking aggregate.

        Parameters
        ----------
        candidate_to_dict : Callable[[CandidateT], JSONValue]
            Callback that converts canonical candidates into JSON-safe values.

        Returns
        -------
        JSONDict
            JSON-safe banking-state snapshot.

        Raises
        ------
        ValueError
            If reference refresh is still in progress.
        """
        if self.refresh_state is not None:
            msg = "banking-state checkpoints require reference refresh to be idle"
            raise ValueError(msg)

        return {
            "bank": self.bank.to_dict(candidate_to_dict=candidate_to_dict),
            "reference_bank": self.reference_bank.to_dict(
                candidate_to_dict=candidate_to_dict,
            ),
            "refresh_state": None,
            "growth_state": self.growth_state.to_dict(),
            "clustering_state": self.clustering_state.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        candidate_from_dict: Callable[[JSONValue], CandidateT],
        growth_policy: CSABankGrowthPolicy,
        clustering_policy: CSAClusteringPolicy,
    ) -> "CSABankingState[CandidateT]":
        """Build a banking aggregate from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe banking-state snapshot.
        candidate_from_dict : Callable[[JSONValue], CandidateT]
            Callback that reconstructs canonical candidates from JSON-safe
            values.
        growth_policy : CSABankGrowthPolicy
            Growth policy that owns the reconstructed growth state.
        clustering_policy : CSAClusteringPolicy
            Clustering policy that owns the reconstructed clustering state.

        Returns
        -------
        CSABankingState[CandidateT]
            Reconstructed banking aggregate.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        ValueError
            If the snapshot attempts to restore an active refresh pool.
        """
        bank_data = require_json_mapping(
            require_json_field(data, "bank"),
            field_name="bank",
        )
        reference_bank_data = require_json_mapping(
            require_json_field(data, "reference_bank"),
            field_name="reference_bank",
        )
        raw_refresh_state = require_json_field(data, "refresh_state")
        if raw_refresh_state is not None:
            msg = "banking-state checkpoints require reference refresh to be idle"
            raise ValueError(msg)
        growth_state_data = require_json_mapping(
            require_json_field(data, "growth_state"),
            field_name="growth_state",
        )
        clustering_state_data = require_json_mapping(
            require_json_field(data, "clustering_state"),
            field_name="clustering_state",
        )
        return cls(
            bank=Bank[CandidateT].from_dict(
                bank_data,
                candidate_from_dict=candidate_from_dict,
            ),
            reference_bank=ReferenceBank[CandidateT].from_dict(
                reference_bank_data,
                candidate_from_dict=candidate_from_dict,
            ),
            refresh_state=None,
            growth_state=CSABankGrowthState[CandidateT].from_dict(
                growth_state_data,
                policy=growth_policy,
            ),
            clustering_state=CSAClusteringState[CandidateT].from_dict(
                clustering_state_data,
                policy=clustering_policy,
            ),
        )
