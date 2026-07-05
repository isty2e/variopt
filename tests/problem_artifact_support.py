"""Shared support for problem and artifact tests."""

from dataclasses import dataclass

from typing_extensions import override

from variopt import (
    EvaluationProtocol,
    EvaluationRequest,
    InteractionEvaluationProtocol,
    Objective,
    ObservationEvaluationProtocol,
    OptimizationDirection,
    Proposal,
)
from variopt.artifacts import (
    InteractionEvaluationSpec,
    InteractionEvaluationUnit,
    ObservationPayload,
    ProposalEvaluationSpec,
)


class SquareObjective(Objective[int]):
    """Minimal objective used to exercise problem and artifact contracts."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


class NaNObjective(Objective[int]):
    """Objective that returns NaN to exercise non-finite evaluation paths."""

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        return float("nan")


class ShiftedObservationProtocol(ObservationEvaluationProtocol[int]):
    """Non-Objective protocol used to exercise protocol-first execution."""

    @override
    def evaluate_request(
        self,
        request: EvaluationRequest[int],
        *,
        direction: OptimizationDirection,
    ) -> ObservationPayload:
        candidate = request.candidate - 1
        return ObservationPayload.from_objective_value(
            value=float(candidate * candidate),
            direction=direction,
        )


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """Simple request-aligned compatibility payload for protocol-path tests."""

    request: EvaluationRequest[int]
    candidate: int

    label: str

    @property
    def proposal(self) -> Proposal[int]:
        """Return the proposal compatibility view."""
        return self.request.proposal

    @property
    def proposal_evaluation_spec(self) -> ProposalEvaluationSpec | None:
        """Return request-local metadata attached to the source request."""
        return self.request.proposal_evaluation_spec


class LabelProtocol(EvaluationProtocol[int, LabelRecord]):
    """Non-scalar protocol used to exercise record-first execution."""

    @override
    def evaluate_request(
        self,
        request: EvaluationRequest[int],
    ) -> LabelRecord:
        candidate = request.candidate
        return LabelRecord(
            request=request,
            candidate=candidate,
            label=f"parity:{candidate % 2}",
        )


@dataclass(frozen=True, slots=True)
class MatchupSpec(InteractionEvaluationSpec):
    """Simple interaction metadata used for sibling interaction-basis tests."""

    arena: str


@dataclass(frozen=True, slots=True)
class MatchupRecord:
    """Simple interaction-aware payload used for sibling interaction tests."""

    interaction_unit: InteractionEvaluationUnit[int]

    winner: int
    arena: str

    @property
    def requests(self) -> tuple[EvaluationRequest[int], ...]:
        """Return request participants."""
        return self.interaction_unit.requests

    @property
    def proposals(self) -> tuple[Proposal[int], ...]:
        """Return proposal participants."""
        return self.interaction_unit.proposals

    @property
    def candidates(self) -> tuple[int, ...]:
        """Return candidate participants."""
        return self.interaction_unit.candidates

    @property
    def interaction_evaluation_spec(self) -> InteractionEvaluationSpec | None:
        """Return the interaction metadata attached to the source unit."""
        return self.interaction_unit.interaction_evaluation_spec


class MatchupProtocol(InteractionEvaluationProtocol[int, MatchupRecord]):
    """Minimal interaction-aware protocol over multiple requests."""

    @override
    def evaluate_interaction_unit(
        self,
        interaction_unit: InteractionEvaluationUnit[int],
    ) -> MatchupRecord:
        winner = max(interaction_unit.candidates)
        interaction_spec = interaction_unit.interaction_evaluation_spec
        arena = "default"
        if isinstance(interaction_spec, MatchupSpec):
            arena = interaction_spec.arena

        return MatchupRecord(
            interaction_unit=interaction_unit,
            winner=winner,
            arena=arena,
        )
