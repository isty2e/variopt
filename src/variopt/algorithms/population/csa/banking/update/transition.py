"""Canonical CSA bank-transition artifacts."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import Literal

from typing_extensions import Self

CSABankTransitionRoute = Literal["initial", "local", "growth", "cluster", "far"]
CSABankTransitionDisposition = Literal["rejected", "appended", "replaced"]


@dataclass(frozen=True, slots=True)
class CSABankTransition:
    """Canonical bank transition aligned with one evaluated proposal.

    Parameters
    ----------
    proposal_id : str
        Stable identifier of the evaluated proposal.
    route : CSABankTransitionRoute
        Policy route that admitted the proposal or made the conclusive rejection.
    disposition : CSABankTransitionDisposition
        Immediate structural effect on the shadow bank.
    target_index : int | None
        Immediate shadow-bank index appended or replaced by the proposal. Rejected
        proposals have no target index.
    survived_batch : bool
        Whether the proposal remains in the final bank after all later observations
        and post-batch energy-cut reduction.
    """

    proposal_id: str
    route: CSABankTransitionRoute
    disposition: CSABankTransitionDisposition
    target_index: int | None
    survived_batch: bool

    def __post_init__(self) -> None:
        """Reject transition states that cannot describe one bank decision."""
        if not isinstance(self.proposal_id, str):
            msg = "proposal_id must be a string"
            raise TypeError(msg)
        if self.proposal_id == "":
            msg = "proposal_id must not be empty"
            raise ValueError(msg)
        if self.route not in ("initial", "local", "growth", "cluster", "far"):
            msg = "route must identify a canonical CSA bank-update route"
            raise ValueError(msg)
        if self.disposition not in ("rejected", "appended", "replaced"):
            msg = "disposition must identify a canonical bank structural effect"
            raise ValueError(msg)
        if not isinstance(self.survived_batch, bool):
            msg = "survived_batch must be a bool"
            raise TypeError(msg)
        if self.route in ("initial", "growth") and self.disposition != "appended":
            msg = "initial and growth routes must append"
            raise ValueError(msg)
        if self.route in ("local", "cluster", "far") and self.disposition == "appended":
            msg = "local, cluster, and far routes cannot append"
            raise ValueError(msg)

        if self.disposition == "rejected":
            if self.target_index is not None:
                msg = "rejected transitions must not declare a target_index"
                raise ValueError(msg)
            if self.survived_batch:
                msg = "rejected transitions cannot survive the batch"
                raise ValueError(msg)
            return

        if type(self.target_index) is not int:
            msg = "admitted transitions must declare an integer target_index"
            raise TypeError(msg)
        if self.target_index < 0:
            msg = "target_index must be non-negative"
            raise ValueError(msg)

    def reconcile_final_survival(
        self,
        surviving_proposal_ids: AbstractSet[str],
    ) -> Self:
        """Return this transition reconciled against final bank membership."""
        survived_batch = (
            self.disposition != "rejected"
            and self.proposal_id in surviving_proposal_ids
        )
        if survived_batch == self.survived_batch:
            return self
        return replace(self, survived_batch=survived_batch)
