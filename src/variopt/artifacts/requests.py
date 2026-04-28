"""Request-plane artifact definitions."""

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..typevars import CandidateT

ObservationCandidateT = TypeVar("ObservationCandidateT")


@dataclass(frozen=True, slots=True)
class Proposal(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Immutable proposal over a canonical candidate.

    Parameters
    ----------
    candidate : CandidateT
        Candidate to evaluate or refine.
    proposal_id : str | None, optional
        Optional stable identifier for diagnostics, traces, or external
        bookkeeping.
    """

    candidate: CandidateT
    proposal_id: str | None = None

    def __post_init__(self) -> None:
        """Validate proposal metadata.

        Raises
        ------
        ValueError
            If ``proposal_id`` is the empty string.
        """
        if self.proposal_id == "":
            msg = "proposal_id must not be empty"
            raise ValueError(msg)


class ProposalEvaluationSpec(ABC):
    """Marker base class for immutable per-proposal request metadata.

    Notes
    -----
    Concrete subclasses may encode fidelity, provenance, resume linkage, or
    other request-plane semantics while :class:`Proposal` remains a pure ask
    artifact over candidate identity.
    """


@dataclass(frozen=True, slots=True)
class EvaluationRequest(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical execution request.

    Parameters
    ----------
    proposal : Proposal[CandidateT]
        Proposal whose candidate should be evaluated.
    proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
        Optional request-local metadata attached to the proposal.
    """

    proposal: Proposal[CandidateT]
    proposal_evaluation_spec: ProposalEvaluationSpec | None = None

    @property
    def candidate(self) -> CandidateT:
        """Return the canonical candidate owned by this request.

        Returns
        -------
        CandidateT
            Candidate carried by ``proposal``.
        """
        return self.proposal.candidate

    @property
    def proposal_id(self) -> str | None:
        """Return the optional proposal identifier.

        Returns
        -------
        str | None
            Proposal identifier carried by ``proposal``.
        """
        return self.proposal.proposal_id


class InteractionEvaluationSpec(ABC):
    """Marker base class for interaction-aware evaluation metadata.

    Notes
    -----
    Concrete realizations can encode opponent roles, bracket context, shared
    cohort metadata, or similar grouping semantics without forcing those ideas
    into proposal-local request artifacts.
    """


@dataclass(frozen=True, slots=True, init=False)
class InteractionEvaluationUnit(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical interaction-aware evaluation unit.

    Parameters
    ----------
    requests : Sequence[EvaluationRequest[CandidateT]]
        Requests that must be evaluated together.
    interaction_evaluation_spec : InteractionEvaluationSpec | None, optional
        Optional metadata shared by the full interaction unit.
    """

    requests: tuple[EvaluationRequest[CandidateT], ...]
    interaction_evaluation_spec: InteractionEvaluationSpec | None = None

    def __init__(
        self,
        *,
        requests: Sequence[EvaluationRequest[CandidateT]],
        interaction_evaluation_spec: InteractionEvaluationSpec | None = None,
    ) -> None:
        """Create one immutable interaction-aware evaluation unit.

        Parameters
        ----------
        requests : Sequence[EvaluationRequest[CandidateT]]
            Requests that participate in one interaction-aware evaluation.
        interaction_evaluation_spec : InteractionEvaluationSpec | None, optional
            Optional shared metadata for the request group.

        Raises
        ------
        ValueError
            If ``requests`` is empty.
        """
        request_tuple = tuple(requests)
        if len(request_tuple) == 0:
            msg = "interaction evaluation unit requires at least one request"
            raise ValueError(msg)

        object.__setattr__(self, "requests", request_tuple)
        object.__setattr__(
            self,
            "interaction_evaluation_spec",
            interaction_evaluation_spec,
        )

    @property
    def request_count(self) -> int:
        """Return the number of participating requests.

        Returns
        -------
        int
            Number of requests contained in the interaction unit.
        """
        return len(self.requests)

    @property
    def proposals(self) -> tuple[Proposal[CandidateT], ...]:
        """Return the proposal compatibility view.

        Returns
        -------
        tuple[Proposal[CandidateT], ...]
            Proposals carried by the requests in this unit.
        """
        return tuple(request.proposal for request in self.requests)

    @property
    def candidates(self) -> tuple[CandidateT, ...]:
        """Return the participating candidates.

        Returns
        -------
        tuple[CandidateT, ...]
            Candidates carried by the requests in this unit.
        """
        return tuple(request.candidate for request in self.requests)


def normalize_evaluation_request(
    *,
    request: EvaluationRequest[ObservationCandidateT] | None = None,
    proposal: Proposal[ObservationCandidateT] | None = None,
    proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
) -> EvaluationRequest[ObservationCandidateT]:
    """Normalize boundary input into one canonical request.

    Parameters
    ----------
    request : EvaluationRequest[ObservationCandidateT] | None, optional
        Existing canonical request.
    proposal : Proposal[ObservationCandidateT] | None, optional
        Proposal to lower into a canonical request.
    proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
        Optional request metadata used when lowering ``proposal``.

    Returns
    -------
    EvaluationRequest[ObservationCandidateT]
        Canonical request constructed from the provided inputs.

    Raises
    ------
    ValueError
        If neither or both of ``request`` and ``proposal`` are provided, or if
        ``proposal_evaluation_spec`` is supplied together with ``request``.
    RuntimeError
        If request normalization fails unexpectedly.
    """
    if (request is None) == (proposal is None):
        msg = "exactly one of request or proposal must be provided"
        raise ValueError(msg)

    if request is not None:
        if proposal_evaluation_spec is not None:
            msg = "proposal_evaluation_spec must not be provided with request"
            raise ValueError(msg)
        return request

    if proposal is None:
        msg = "evaluation request normalization failed"
        raise RuntimeError(msg)

    return EvaluationRequest(
        proposal=proposal,
        proposal_evaluation_spec=proposal_evaluation_spec,
    )
