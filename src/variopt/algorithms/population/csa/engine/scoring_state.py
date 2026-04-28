"""Canonical CSA scoring aggregate state."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....json_types import JSONDict, JSONValue
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
        raw_acceptance_state = data.get("acceptance_state")
        raw_model_state = data.get("model_state")
        if not isinstance(raw_acceptance_state, dict):
            msg = "scoring-state snapshot requires acceptance_state mapping"
            raise TypeError(msg)
        if not isinstance(raw_model_state, dict):
            msg = "scoring-state snapshot requires model_state mapping"
            raise TypeError(msg)
        return cls(
            acceptance_state=CSAAcceptanceState.from_dict(
                raw_acceptance_state,
                policy=acceptance_policy,
            ),
            model_state=CSAScoreModelState[CandidateT].from_dict(
                raw_model_state,
                score_model=score_model,
            ),
        )
