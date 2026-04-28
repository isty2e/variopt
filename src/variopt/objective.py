"""Evaluation protocol and scalar objective interface definitions."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

from typing_extensions import TypeVar, override

from .artifacts import (
    EvaluationRequest,
    InteractionEvaluationSpec,
    InteractionEvaluationUnit,
    Observation,
    Proposal,
    ProposalEvaluationSpec,
)
from .direction import OptimizationDirection
from .typevars import CandidateT

ProtocolRecordT = TypeVar("ProtocolRecordT")
InteractionProtocolRecordT = TypeVar("InteractionProtocolRecordT")


class EvaluationProtocol(ABC, Generic[CandidateT, ProtocolRecordT]):
    """Evaluate one canonical request into one semantic record.

    Notes
    -----
    This is the direction-free evaluation contract used by the core execution
    pipeline. Implementations should interpret exactly one
    :class:`~variopt.artifacts.EvaluationRequest` and return exactly one
    domain-level record. Scalar ordering semantics, execution accounting, and
    batching policy belong to adjacent layers rather than to this protocol.
    """

    @abstractmethod
    def evaluate_request(
        self,
        request: EvaluationRequest[CandidateT],
    ) -> ProtocolRecordT:
        """Evaluate one canonical request.

        Parameters
        ----------
        request : EvaluationRequest[CandidateT]
            Canonical request describing the proposal to evaluate and any
            request-local evaluation metadata.

        Returns
        -------
        ProtocolRecordT
            Semantic evaluation record produced for ``request``.
        """

    def evaluate_proposal(
        self,
        proposal: Proposal[CandidateT],
        *,
        proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
    ) -> ProtocolRecordT:
        """Evaluate one proposal through the request-first contract.

        Parameters
        ----------
        proposal : Proposal[CandidateT]
            Proposal to lower into a canonical evaluation request.
        proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
            Optional request-local metadata to attach while lowering the
            proposal into an :class:`~variopt.artifacts.EvaluationRequest`.

        Returns
        -------
        ProtocolRecordT
            Semantic evaluation record produced for the lowered request.
        """
        return self.evaluate_request(
            EvaluationRequest(
                proposal=proposal,
                proposal_evaluation_spec=proposal_evaluation_spec,
            )
        )


class InteractionEvaluationProtocol(
    ABC,
    Generic[CandidateT, InteractionProtocolRecordT],
):
    """Evaluate one interaction-aware request group into one semantic record.

    Notes
    -----
    Use this sibling protocol when evaluation semantics depend on multiple
    requests at once, such as tournaments, pairwise comparisons, or shared
    cohort context. It keeps proposal-local
    :class:`EvaluationProtocol` honest instead of stretching that contract to
    cover multi-request behavior.
    """

    @abstractmethod
    def evaluate_interaction_unit(
        self,
        interaction_unit: InteractionEvaluationUnit[CandidateT],
    ) -> InteractionProtocolRecordT:
        """Evaluate one canonical interaction unit.

        Parameters
        ----------
        interaction_unit : InteractionEvaluationUnit[CandidateT]
            Canonical grouping of requests that must be evaluated together.

        Returns
        -------
        InteractionProtocolRecordT
            Semantic record produced for the full interaction unit.
        """

    def evaluate_requests(
        self,
        requests: Sequence[EvaluationRequest[CandidateT]],
        *,
        interaction_evaluation_spec: InteractionEvaluationSpec | None = None,
    ) -> InteractionProtocolRecordT:
        """Evaluate a request group through the interaction contract.

        Parameters
        ----------
        requests : Sequence[EvaluationRequest[CandidateT]]
            Requests that participate in a single interaction-aware evaluation.
        interaction_evaluation_spec : InteractionEvaluationSpec | None, optional
            Optional metadata shared by the interaction group.

        Returns
        -------
        InteractionProtocolRecordT
            Semantic record produced for the grouped requests.
        """
        return self.evaluate_interaction_unit(
            InteractionEvaluationUnit(
                requests=requests,
                interaction_evaluation_spec=interaction_evaluation_spec,
            )
        )


class ObservationEvaluationProtocol(ABC, Generic[CandidateT]):
    """Evaluate requests into scalar observations with explicit direction.

    Notes
    -----
    This is the scalar compatibility layer over the direction-free
    :class:`EvaluationProtocol` contract. It keeps raw objective direction
    handling explicit at the boundary where scalar observations are created.
    """

    @abstractmethod
    def evaluate_request(
        self,
        request: EvaluationRequest[CandidateT],
        *,
        direction: OptimizationDirection,
    ) -> Observation[CandidateT]:
        """Evaluate one request into a scalar observation.

        Parameters
        ----------
        request : EvaluationRequest[CandidateT]
            Canonical request to evaluate.
        direction : OptimizationDirection
            Interpretation of the raw objective value produced by the
            implementation.

        Returns
        -------
        Observation[CandidateT]
            Scalar observation carrying both the raw value and the canonical
            minimization score.
        """

    def evaluate_proposal(
        self,
        proposal: Proposal[CandidateT],
        *,
        direction: OptimizationDirection,
        proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
    ) -> Observation[CandidateT]:
        """Evaluate one proposal into a scalar observation.

        Parameters
        ----------
        proposal : Proposal[CandidateT]
            Proposal to evaluate.
        direction : OptimizationDirection
            Interpretation of the raw objective value produced by the
            implementation.
        proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
            Optional request-local metadata to attach while lowering the
            proposal into a canonical request.

        Returns
        -------
        Observation[CandidateT]
            Scalar observation produced for the lowered request.
        """
        return self.evaluate_request(
            EvaluationRequest(
                proposal=proposal,
                proposal_evaluation_spec=proposal_evaluation_spec,
            ),
            direction=direction,
        )


class ScalarEvaluationProtocol(
    ObservationEvaluationProtocol[CandidateT],
    ABC,
    Generic[CandidateT],
):
    """Scalar evaluation contract over canonical candidates.

    Notes
    -----
    Implementations expose the smallest useful scalar interface,
    :meth:`evaluate`, while inheriting the request-based compatibility behavior
    from :class:`ObservationEvaluationProtocol`.
    """

    @abstractmethod
    def evaluate(self, candidate: CandidateT) -> float:
        """Return a raw scalar objective value for one candidate.

        Parameters
        ----------
        candidate : CandidateT
            Canonical candidate to score.

        Returns
        -------
        float
            Raw objective value before direction normalization.
        """

    @override
    def evaluate_request(
        self,
        request: EvaluationRequest[CandidateT],
        *,
        direction: OptimizationDirection,
    ) -> Observation[CandidateT]:
        """Evaluate one request by delegating to :meth:`evaluate`.

        Parameters
        ----------
        request : EvaluationRequest[CandidateT]
            Canonical request whose candidate should be scored.
        direction : OptimizationDirection
            Interpretation of the raw scalar value returned by
            :meth:`evaluate`.

        Returns
        -------
        Observation[CandidateT]
            Scalar observation constructed from the raw objective value.
        """
        candidate = request.candidate
        return Observation.from_objective_value(
            request=request,
            candidate=candidate,
            value=self.evaluate(candidate),
            direction=direction,
        )


class Objective(ScalarEvaluationProtocol[CandidateT], ABC):
    """Ergonomic scalar objective interface for deterministic scoring rules.

    Notes
    -----
    ``Objective`` is the user-facing scalar hook used by the simplest
    ``Problem`` configurations. Internally, the library lowers it into the
    canonical request-based evaluation protocol, but keeping this class makes
    scalar optimization problems easy to define and read.
    """
