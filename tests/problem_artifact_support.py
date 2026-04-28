"""Shared support for problem and artifact tests."""

from dataclasses import dataclass

from typing_extensions import override

from variopt import (
    EvaluationProtocol,
    EvaluationRecord,
    EvaluationRequest,
    InteractionEvaluationProtocol,
    Objective,
    Observation,
    ObservationEvaluationProtocol,
    OptimizationDirection,
)
from variopt.artifacts import (
    InteractionEvaluationRecord,
    InteractionEvaluationSpec,
    InteractionEvaluationUnit,
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
    ) -> Observation[int]:
        candidate = request.candidate - 1
        return Observation.from_objective_value(
            request=request,
            candidate=candidate,
            value=float(candidate * candidate),
            direction=direction,
        )


@dataclass(frozen=True, slots=True)
class LabelRecord(EvaluationRecord[int]):
    """Simple non-scalar evaluation record for protocol-path tests."""

    label: str


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
class MatchupRecord(InteractionEvaluationRecord[int]):
    """Simple interaction-aware record used for sibling interaction tests."""

    winner: int
    arena: str


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
