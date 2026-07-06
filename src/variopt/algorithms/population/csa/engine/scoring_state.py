"""Canonical CSA scoring aggregate state."""

from collections.abc import Mapping
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
from ..scoring.acceptance import CSAAcceptancePolicy
from ..scoring.acceptance_state import CSAAcceptanceState
from ..scoring.model import CSAScoreModel
from ..scoring.model_state import CSAScoreModelState


@dataclass(frozen=True, slots=True)
class CSAScoringState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical scoring aggregate for CSA execution.

    Parameters
    ----------
    acceptance_state : CSAAcceptanceState
        Acceptance runtime state.
    model_state : CSAScoreModelState[CandidateT]
        Score-model runtime state.
    """

    acceptance_state: CSAAcceptanceState
    model_state: CSAScoreModelState[CandidateT]

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the scoring aggregate.

        Returns
        -------
        JSONDict
            JSON-safe scoring-state snapshot.
        """
        return {
            "acceptance_state": self.acceptance_state.to_dict(),
            "model_state": self.model_state.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        acceptance_policy: CSAAcceptancePolicy,
        score_model: CSAScoreModel[CandidateT],
    ) -> "CSAScoringState[CandidateT]":
        """Build a scoring aggregate from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe scoring-state snapshot.
        acceptance_policy : CSAAcceptancePolicy
            Acceptance policy that owns the reconstructed state.
        score_model : CSAScoreModel[CandidateT]
            Score model that owns the reconstructed state.

        Returns
        -------
        CSAScoringState[CandidateT]
            Reconstructed scoring aggregate.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        acceptance_state_data = require_json_mapping(
            require_json_field(data, "acceptance_state"),
            field_name="acceptance_state",
        )
        model_state_data = require_json_mapping(
            require_json_field(data, "model_state"),
            field_name="model_state",
        )
        return cls(
            acceptance_state=CSAAcceptanceState.from_dict(
                acceptance_state_data,
                policy=acceptance_policy,
            ),
            model_state=CSAScoreModelState[CandidateT].from_dict(
                model_state_data,
                score_model=score_model,
            ),
        )
