"""Problem definitions."""

from dataclasses import dataclass, field
from typing import Generic

from typing_extensions import TypeVar, override

from variopt.generic_runtime import (
    FrozenGenericSlotsCompat,
    install_frozen_generic_slots_pickle,
)

from .artifacts import (
    EvaluationRequest,
    ObservationPayload,
    Proposal,
)
from .direction import OptimizationDirection
from .objective import (
    EvaluationProtocol,
    InteractionEvaluationProtocol,
    Objective,
    ObservationEvaluationProtocol,
)
from .spaces import SearchSpace
from .typevars import CandidateT

BoundaryT = TypeVar("BoundaryT")
ProblemPayloadT = TypeVar("ProblemPayloadT", default=ObservationPayload)
InteractionProblemRecordT = TypeVar("InteractionProblemRecordT")


@dataclass(frozen=True, slots=True)
class _ProtocolObjectiveCompatibilityView(FrozenGenericSlotsCompat, Generic[CandidateT], Objective[CandidateT]):
    """Scalar objective view derived from an observation protocol.

    Notes
    -----
    This adapter is internal to :class:`Problem`. It exists so that a
    direction-bound observation protocol can still expose the ergonomic
    :class:`~variopt.objective.Objective` compatibility view when callers ask
    for ``problem.objective``.
    """

    evaluation_protocol: ObservationEvaluationProtocol[CandidateT]
    direction: OptimizationDirection

    @override
    def evaluate(self, candidate: CandidateT) -> float:
        """Return the wrapped protocol's raw scalar value.

        Parameters
        ----------
        candidate : CandidateT
            Canonical candidate to evaluate.

        Returns
        -------
        float
            Raw objective value recovered from the wrapped observation
            protocol.
        """
        payload = self.evaluation_protocol.evaluate_request(
            EvaluationRequest(proposal=Proposal(candidate=candidate)),
            direction=self.direction,
        )
        return payload.value


@dataclass(frozen=True, slots=True)
class _ObservationProtocolEvaluationProtocolAdapter(FrozenGenericSlotsCompat,
    Generic[CandidateT],
    EvaluationProtocol[CandidateT, ObservationPayload],
):
    """Direction-free evaluation view over a scalar observation protocol.

    Notes
    -----
    ``Problem`` uses this adapter to bind scalar direction semantics exactly
    once at construction time, then expose a canonical
    :class:`~variopt.objective.EvaluationProtocol` to the rest of the
    execution stack.
    """

    observation_evaluation_protocol: ObservationEvaluationProtocol[CandidateT]
    direction: OptimizationDirection

    @override
    def evaluate_request(
        self,
        request: EvaluationRequest[CandidateT],
    ) -> ObservationPayload:
        """Evaluate a canonical request through the wrapped protocol.

        Parameters
        ----------
        request : EvaluationRequest[CandidateT]
            Canonical request to forward.

        Returns
        -------
        ObservationPayload
            Scalar payload emitted by the wrapped observation protocol.
        """
        return self.observation_evaluation_protocol.evaluate_request(
            request,
            direction=self.direction,
        )


@dataclass(frozen=True, slots=True, init=False)
class Problem(FrozenGenericSlotsCompat, Generic[BoundaryT, CandidateT, ProblemPayloadT]):
    """Immutable proposal-local optimization problem.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, CandidateT]
        Canonical search-space definition for valid candidates.
    objective : Objective[CandidateT] | None, optional
        Optional scalar objective compatibility view. Provide this for the
        simplest scalar problem definitions.
    evaluation_protocol : EvaluationProtocol[CandidateT, ProblemPayloadT] | ObservationEvaluationProtocol[CandidateT] | None, optional
        Optional canonical request-aligned evaluation protocol. This may be
        direction-free or a scalar observation protocol that ``Problem`` will
        lower into the canonical request-based contract.
    direction : OptimizationDirection, default=OptimizationDirection.MINIMIZE
        Scalar optimization direction to bind when ``objective`` or an
        observation protocol is supplied.
    name : str | None, optional
        Optional human-readable label for reports, examples, and diagnostics.

    Notes
    -----
    Exactly one of ``objective`` or ``evaluation_protocol`` must be provided.
    Internally, ``Problem`` stores the direction-free
    :class:`~variopt.objective.EvaluationProtocol` contract and exposes
    ``objective`` only as a compatibility view when a scalar interpretation is
    available. The canonical evaluation protocol must emit request-free
    payloads; execution layers own request identity and attempt recording.
    """

    space: SearchSpace[BoundaryT, CandidateT]
    evaluation_protocol: EvaluationProtocol[CandidateT, ProblemPayloadT]
    direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    name: str | None = None
    _objective_compat: Objective[CandidateT] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        *,
        space: SearchSpace[BoundaryT, CandidateT],
        objective: Objective[CandidateT] | None = None,
        evaluation_protocol: (
            EvaluationProtocol[CandidateT, ProblemPayloadT]
            | ObservationEvaluationProtocol[CandidateT]
        )
        | None = None,
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE,
        name: str | None = None,
    ) -> None:
        """Create an immutable problem specification.

        Parameters
        ----------
        space : SearchSpace[BoundaryT, CandidateT]
            Search-space definition that owns candidate validity and sampling
            semantics.
        objective : Objective[CandidateT] | None, optional
            Optional scalar objective compatibility view.
        evaluation_protocol : EvaluationProtocol[CandidateT, ProblemPayloadT] | ObservationEvaluationProtocol[CandidateT] | None, optional
            Optional canonical payload-returning evaluation protocol or scalar
            observation protocol.
        direction : OptimizationDirection, default=OptimizationDirection.MINIMIZE
            Scalar direction bound at construction time when needed.
        name : str | None, optional
            Optional human-readable problem label.

        Raises
        ------
        ValueError
            If neither or both of ``objective`` and ``evaluation_protocol`` are
            provided.
        RuntimeError
            If protocol normalization fails unexpectedly.
        """
        if (objective is None) == (evaluation_protocol is None):
            msg = "exactly one of objective or evaluation_protocol must be provided"
            raise ValueError(msg)

        protocol = objective if objective is not None else evaluation_protocol
        if protocol is None:
            msg = "evaluation_protocol normalization failed"
            raise RuntimeError(msg)

        objective_compat: Objective[CandidateT] | None
        if objective is not None:
            canonical_protocol = _ObservationProtocolEvaluationProtocolAdapter(
                observation_evaluation_protocol=objective,
                direction=direction,
            )
            objective_compat = objective
        elif isinstance(protocol, Objective):
            canonical_protocol = _ObservationProtocolEvaluationProtocolAdapter(
                observation_evaluation_protocol=protocol,
                direction=direction,
            )
            objective_compat = protocol
        elif isinstance(protocol, ObservationEvaluationProtocol):
            canonical_protocol = _ObservationProtocolEvaluationProtocolAdapter(
                observation_evaluation_protocol=protocol,
                direction=direction,
            )
            objective_compat = _ProtocolObjectiveCompatibilityView(
                evaluation_protocol=protocol,
                direction=direction,
            )
        else:
            canonical_protocol = protocol
            objective_compat = None

        object.__setattr__(self, "space", space)
        object.__setattr__(self, "evaluation_protocol", canonical_protocol)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_objective_compat", objective_compat)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate user-facing ``Problem`` invariants.

        Raises
        ------
        ValueError
            If ``name`` is the empty string.
        TypeError
            If ``direction`` is not a concrete
            :class:`~variopt.direction.OptimizationDirection`.
        """
        if self.name == "":
            msg = "name must not be empty"
            raise ValueError(msg)

        if type(self.direction) is not OptimizationDirection:
            msg = "direction must be an OptimizationDirection"
            raise TypeError(msg)

    @property
    def objective(self) -> Objective[CandidateT]:
        """Return the scalar objective compatibility view.

        Returns
        -------
        Objective[CandidateT]
            Scalar objective view for this problem.

        Raises
        ------
        TypeError
            If the problem was defined with a non-scalar evaluation protocol
            and therefore has no scalar compatibility view.

        Notes
        -----
        Prefer :attr:`evaluation_protocol` in canonical internal code. This
        property is for boundary convenience when a scalar objective view is
        meaningful.
        """
        if self._objective_compat is None:
            msg = "problem does not expose a scalar Objective compatibility view"
            raise TypeError(msg)
        return self._objective_compat

    @property
    def direct_objective(self) -> Objective[CandidateT] | None:
        """Return the direct scalar objective configured on this problem, if any.

        Returns
        -------
        Objective[CandidateT] | None
            The scalar objective supplied directly at construction time. Returns
            ``None`` for non-scalar protocols and for scalar observation
            protocols adapted through request-aware compatibility views.

        Notes
        -----
        This property is intentionally narrower than :attr:`objective`.
        Request-aware observation protocols may depend on request metadata, so
        execution code must not bypass their canonical request contract.
        """
        objective = self._objective_compat
        if objective is None:
            return None

        if type(objective) is _ProtocolObjectiveCompatibilityView:
            return None

        return objective


install_frozen_generic_slots_pickle(Problem)


@dataclass(frozen=True, slots=True)
class InteractionProblem(FrozenGenericSlotsCompat, Generic[BoundaryT, CandidateT, InteractionProblemRecordT]):
    """Immutable interaction-aware optimization problem.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, CandidateT]
        Canonical search-space definition for the participating candidates.
    interaction_evaluation_protocol : InteractionEvaluationProtocol[CandidateT, InteractionProblemRecordT]
        Interaction-aware evaluation contract for grouped requests.
    name : str | None, optional
        Optional human-readable problem label.
    """

    space: SearchSpace[BoundaryT, CandidateT]
    interaction_evaluation_protocol: InteractionEvaluationProtocol[
        CandidateT,
        InteractionProblemRecordT,
    ]
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate interaction-problem metadata.

        Raises
        ------
        ValueError
            If ``name`` is the empty string.
        """
        if self.name == "":
            msg = "name must not be empty"
            raise ValueError(msg)


install_frozen_generic_slots_pickle(InteractionProblem)
