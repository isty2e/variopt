"""Evaluation-record artifact definitions."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Generic, Protocol, runtime_checkable

import numpy as np
from typing_extensions import TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..direction import OptimizationDirection
from .requests import (
    EvaluationRequest,
    Proposal,
    ProposalEvaluationSpec,
    normalize_evaluation_request,
)

EvaluationRecordCandidateT = TypeVar("EvaluationRecordCandidateT")
RequestAlignedEvaluationRecordCandidateT = TypeVar(
    "RequestAlignedEvaluationRecordCandidateT",
    default=object,
    covariant=True,
)
ObservationRecordCandidateT = TypeVar("ObservationRecordCandidateT")
ObservationCandidateT = TypeVar("ObservationCandidateT")
ObjectiveVectorRecordCandidateT = TypeVar("ObjectiveVectorRecordCandidateT")
ObjectiveVectorCandidateT = TypeVar("ObjectiveVectorCandidateT")


def _normalize_scalar_record_float(
    value: object,
    *,
    field_name: str,
) -> float:
    """Return one finite canonical float for scalar record fields."""
    if type(value) is bool or isinstance(value, np.bool_):
        msg = f"{field_name} must be a real number"
        raise TypeError(msg)

    if not isinstance(value, Real):
        msg = f"{field_name} must be a real number"
        raise TypeError(msg)

    normalized_value = float(value)
    if not np.isfinite(normalized_value):
        msg = f"{field_name} must be finite"
        raise ValueError(msg)
    return normalized_value


@dataclass(frozen=True, slots=True)
class ObservationPayload(FrozenGenericSlotsCompat):
    """Request-free scalar objective payload.

    Parameters
    ----------
    value : float
        Raw scalar objective value.
    score : float
        Canonical minimization score derived from ``value`` and the optimization
        direction.
    elapsed_seconds : float | None, optional
        Optional wall-clock runtime for the evaluation.
    """

    value: float
    score: float
    elapsed_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate scalar payload values.

        Raises
        ------
        ValueError
            If ``value`` or ``score`` is non-finite, or if
            ``elapsed_seconds`` is negative.
        """
        if not np.isfinite(self.value):
            msg = "value must be finite"
            raise ValueError(msg)

        if not np.isfinite(self.score):
            msg = "score must be finite"
            raise ValueError(msg)

        if self.elapsed_seconds is not None:
            if not np.isfinite(self.elapsed_seconds):
                msg = "elapsed_seconds must be finite"
                raise ValueError(msg)
            if self.elapsed_seconds < 0.0:
                msg = "elapsed_seconds must be non-negative"
                raise ValueError(msg)

    @staticmethod
    def from_objective_value(
        *,
        value: float,
        direction: OptimizationDirection,
        elapsed_seconds: float | None = None,
    ) -> "ObservationPayload":
        """Build a scalar payload from a raw objective value.

        Parameters
        ----------
        value : float
            Raw scalar objective value.
        direction : OptimizationDirection
            Direction used to normalize ``value`` into the canonical
            minimization score.
        elapsed_seconds : float | None, optional
            Optional wall-clock runtime for the evaluation.

        Returns
        -------
        ObservationPayload
            Request-free scalar objective payload.
        """
        normalized_value = float(value)
        return ObservationPayload(
            value=normalized_value,
            score=direction.normalize_objective_value(normalized_value),
            elapsed_seconds=elapsed_seconds,
        )


@dataclass(frozen=True, slots=True, init=False)
class ObjectiveVectorPayload(FrozenGenericSlotsCompat):
    """Request-free multi-objective payload.

    Parameters
    ----------
    objective_values : Sequence[float]
        Raw objective values.
    objective_scores : Sequence[float]
        Canonical minimization scores aligned with ``objective_values``.
    elapsed_seconds : float | None, optional
        Optional wall-clock runtime for the evaluation.
    """

    objective_values: tuple[float, ...]
    objective_scores: tuple[float, ...]
    elapsed_seconds: float | None = None

    def __init__(
        self,
        *,
        objective_values: Sequence[float],
        objective_scores: Sequence[float],
        elapsed_seconds: float | None = None,
    ) -> None:
        """Create one request-free vector objective payload.

        Parameters
        ----------
        objective_values : Sequence[float]
            Raw objective values.
        objective_scores : Sequence[float]
            Canonical minimization scores aligned with ``objective_values``.
        elapsed_seconds : float | None, optional
            Optional wall-clock runtime for the evaluation.
        """
        normalized_objective_values = normalize_objective_vector(
            values=objective_values,
            field_name="objective_values",
        )
        normalized_objective_scores = normalize_objective_vector(
            values=objective_scores,
            field_name="objective_scores",
        )
        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "objective_values", normalized_objective_values)
        object.__setattr__(self, "objective_scores", normalized_objective_scores)
        object.__setattr__(self, "elapsed_seconds", elapsed_seconds)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate vector objective payload values.

        Raises
        ------
        ValueError
            If the objective vectors have different lengths or if
            ``elapsed_seconds`` is negative.
        """
        if len(self.objective_values) != len(self.objective_scores):
            msg = "objective_values and objective_scores must have the same length"
            raise ValueError(msg)

        if self.elapsed_seconds is not None:
            if not np.isfinite(self.elapsed_seconds):
                msg = "elapsed_seconds must be finite"
                raise ValueError(msg)
            if self.elapsed_seconds < 0.0:
                msg = "elapsed_seconds must be non-negative"
                raise ValueError(msg)

    @staticmethod
    def from_objective_values(
        *,
        objective_values: Sequence[float],
        directions: Sequence[OptimizationDirection],
        elapsed_seconds: float | None = None,
    ) -> "ObjectiveVectorPayload":
        """Build a vector payload from raw objective values.

        Parameters
        ----------
        objective_values : Sequence[float]
            Raw objective values.
        directions : Sequence[OptimizationDirection]
            Direction for each objective value.
        elapsed_seconds : float | None, optional
            Optional wall-clock runtime for the evaluation.

        Returns
        -------
        ObjectiveVectorPayload
            Request-free vector objective payload with normalized scores.

        Raises
        ------
        ValueError
            If ``directions`` does not align with ``objective_values``.
        """
        normalized_objective_values = normalize_objective_vector(
            values=objective_values,
            field_name="objective_values",
        )
        normalized_directions = tuple(directions)
        if len(normalized_directions) != len(normalized_objective_values):
            msg = "directions must align with objective_values"
            raise ValueError(msg)

        objective_scores = tuple(
            direction.normalize_objective_value(value)
            for value, direction in zip(
                normalized_objective_values,
                normalized_directions,
                strict=True,
            )
        )
        return ObjectiveVectorPayload(
            objective_values=normalized_objective_values,
            objective_scores=objective_scores,
            elapsed_seconds=elapsed_seconds,
        )


@runtime_checkable
class RequestAlignedEvaluationRecord(
    Protocol[RequestAlignedEvaluationRecordCandidateT]
):
    """Minimal request-aligned evaluation record contract.

    Notes
    -----
    This protocol captures the semantic boundary shared by
    :class:`EvaluationRecord` and its concrete subclasses: the record must own
    exactly one canonical request slot and the candidate evaluated for that
    slot. The request's proposal candidate and the evaluated candidate may differ
    when execution-side refinement occurred, so consumers that need the evaluated
    candidate must read :attr:`candidate`.
    """

    @property
    def request(
        self,
    ) -> EvaluationRequest[RequestAlignedEvaluationRecordCandidateT]:
        """Return the canonical request that produced the record."""
        ...

    @property
    def candidate(self) -> RequestAlignedEvaluationRecordCandidateT:
        """Return the candidate evaluated for the request."""
        ...


@dataclass(frozen=True, slots=True)
class EvaluationRecord(
    FrozenGenericSlotsCompat,
    Generic[EvaluationRecordCandidateT],
):
    """Immutable semantic record over one canonical request.

    Parameters
    ----------
    request : EvaluationRequest[EvaluationRecordCandidateT]
        Canonical request that produced the record.
    candidate : EvaluationRecordCandidateT
        Candidate evaluated for ``request``.
    """

    request: EvaluationRequest[EvaluationRecordCandidateT]
    candidate: EvaluationRecordCandidateT

    @property
    def proposal(self) -> Proposal[EvaluationRecordCandidateT]:
        """Return the proposal compatibility view.

        Returns
        -------
        Proposal[EvaluationRecordCandidateT]
            Proposal owned by :attr:`request`.
        """
        return self.request.proposal

    @property
    def proposal_evaluation_spec(self) -> ProposalEvaluationSpec | None:
        """Return the request-spec compatibility view.

        Returns
        -------
        ProposalEvaluationSpec | None
            Request-local metadata attached to :attr:`request`.
        """
        return self.request.proposal_evaluation_spec


@dataclass(frozen=True, slots=True, init=False)
class Observation(
    EvaluationRecord[ObservationRecordCandidateT],
    Generic[ObservationRecordCandidateT],
):
    """Immutable scalar evaluation result.

    Parameters
    ----------
    request : EvaluationRequest[ObservationRecordCandidateT] | None, optional
        Existing canonical request.
    proposal : Proposal[ObservationRecordCandidateT] | None, optional
        Proposal to lower into a canonical request.
    proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
        Optional request metadata used when lowering ``proposal``.
    candidate : ObservationRecordCandidateT
        Candidate evaluated by the request.
    value : float
        Raw scalar objective value.
    score : float
        Canonical minimization score derived from ``value`` and the optimization
        direction.
    elapsed_seconds : float | None, optional
        Optional wall-clock runtime for the evaluation.
    """

    request: EvaluationRequest[ObservationRecordCandidateT]
    candidate: ObservationRecordCandidateT
    value: float
    score: float
    elapsed_seconds: float | None = None

    def __init__(
        self,
        *,
        request: EvaluationRequest[ObservationRecordCandidateT] | None = None,
        proposal: Proposal[ObservationRecordCandidateT] | None = None,
        proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
        candidate: ObservationRecordCandidateT,
        value: float,
        score: float,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Create one canonical scalar observation.

        Parameters
        ----------
        request : EvaluationRequest[ObservationRecordCandidateT] | None, optional
            Existing canonical request.
        proposal : Proposal[ObservationRecordCandidateT] | None, optional
            Proposal to lower into a canonical request.
        proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
            Optional request metadata used when lowering ``proposal``.
        candidate : ObservationRecordCandidateT
            Candidate evaluated by the request.
        value : float
            Raw scalar objective value, normalized to a canonical Python
            ``float``.
        score : float
            Canonical minimization score used for ordering, normalized to a
            canonical Python ``float``.
        elapsed_seconds : float | None, optional
            Optional wall-clock runtime for the evaluation, normalized to a
            canonical Python ``float`` when present.
        """
        normalized_request = normalize_evaluation_request(
            request=request,
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )
        normalized_value = _normalize_scalar_record_float(
            value,
            field_name="value",
        )
        normalized_score = _normalize_scalar_record_float(
            score,
            field_name="score",
        )
        normalized_elapsed_seconds = (
            None
            if elapsed_seconds is None
            else _normalize_scalar_record_float(
                elapsed_seconds,
                field_name="elapsed_seconds",
            )
        )
        if (
            normalized_elapsed_seconds is not None
            and normalized_elapsed_seconds < 0.0
        ):
            msg = "elapsed_seconds must be non-negative"
            raise ValueError(msg)

        super(Observation, self).__init__(
            request=normalized_request,
            candidate=candidate,
        )
        object.__setattr__(self, "value", normalized_value)
        object.__setattr__(self, "score", normalized_score)
        object.__setattr__(self, "elapsed_seconds", normalized_elapsed_seconds)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate scalar observation payloads.

        Raises
        ------
        ValueError
            If ``value`` or ``score`` is non-finite, or if ``elapsed_seconds``
            is negative.
        """
        if not np.isfinite(self.value):
            msg = "value must be finite"
            raise ValueError(msg)

        if not np.isfinite(self.score):
            msg = "score must be finite"
            raise ValueError(msg)

        if self.elapsed_seconds is not None and self.elapsed_seconds < 0.0:
            msg = "elapsed_seconds must be non-negative"
            raise ValueError(msg)

    @staticmethod
    def from_objective_value(
        *,
        request: EvaluationRequest[ObservationCandidateT] | None = None,
        proposal: Proposal[ObservationCandidateT] | None = None,
        proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
        candidate: ObservationCandidateT,
        value: float,
        direction: OptimizationDirection,
        elapsed_seconds: float | None = None,
    ) -> "Observation[ObservationCandidateT]":
        """Build an observation from a raw scalar objective value.

        Parameters
        ----------
        request : EvaluationRequest[ObservationCandidateT] | None, optional
            Existing canonical request.
        proposal : Proposal[ObservationCandidateT] | None, optional
            Proposal to lower into a canonical request.
        proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
            Optional request metadata used when lowering ``proposal``.
        candidate : ObservationCandidateT
            Candidate evaluated by the request.
        value : float
            Raw scalar objective value.
        direction : OptimizationDirection
            Direction used to normalize ``value`` into the canonical
            minimization score.
        elapsed_seconds : float | None, optional
            Optional wall-clock runtime for the evaluation.

        Returns
        -------
        Observation[ObservationCandidateT]
            Scalar observation with both raw value and canonical score.
        """
        normalized_value = float(value)
        return Observation(
            request=request,
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
            candidate=candidate,
            value=normalized_value,
            score=direction.normalize_objective_value(normalized_value),
            elapsed_seconds=elapsed_seconds,
        )


def normalize_objective_vector(
    *,
    values: Sequence[float],
    field_name: str,
) -> tuple[float, ...]:
    """Normalize a float sequence into a canonical objective vector.

    Parameters
    ----------
    values : Sequence[float]
        Raw objective values or scores.
    field_name : str
        Human-readable field name used in validation errors.

    Returns
    -------
    tuple[float, ...]
        Finite non-empty tuple of floats.

    Raises
    ------
    ValueError
        If ``values`` is empty or contains a non-finite number.
    """
    normalized_values = tuple(float(value) for value in values)
    if len(normalized_values) == 0:
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)

    if not bool(np.all(np.isfinite(normalized_values))):
        msg = f"{field_name} must contain only finite values"
        raise ValueError(msg)

    return normalized_values


@dataclass(frozen=True, slots=True, init=False)
class ObjectiveVectorRecord(
    EvaluationRecord[ObjectiveVectorRecordCandidateT],
    Generic[ObjectiveVectorRecordCandidateT],
):
    """Immutable multi-objective evaluation record.

    Parameters
    ----------
    request : EvaluationRequest[ObjectiveVectorRecordCandidateT] | None, optional
        Existing canonical request.
    proposal : Proposal[ObjectiveVectorRecordCandidateT] | None, optional
        Proposal to lower into a canonical request.
    proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
        Optional request metadata used when lowering ``proposal``.
    candidate : ObjectiveVectorRecordCandidateT
        Candidate evaluated by the request.
    objective_values : Sequence[float]
        Raw objective values.
    objective_scores : Sequence[float]
        Canonical minimization scores aligned with ``objective_values``.
    elapsed_seconds : float | None, optional
        Optional wall-clock runtime for the evaluation.
    """

    request: EvaluationRequest[ObjectiveVectorRecordCandidateT]
    candidate: ObjectiveVectorRecordCandidateT
    objective_values: tuple[float, ...]
    objective_scores: tuple[float, ...]
    elapsed_seconds: float | None = None

    def __init__(
        self,
        *,
        request: EvaluationRequest[ObjectiveVectorRecordCandidateT] | None = None,
        proposal: Proposal[ObjectiveVectorRecordCandidateT] | None = None,
        proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
        candidate: ObjectiveVectorRecordCandidateT,
        objective_values: Sequence[float],
        objective_scores: Sequence[float],
        elapsed_seconds: float | None = None,
    ) -> None:
        """Create one canonical vector-valued evaluation record.

        Parameters
        ----------
        request : EvaluationRequest[ObjectiveVectorRecordCandidateT] | None, optional
            Existing canonical request.
        proposal : Proposal[ObjectiveVectorRecordCandidateT] | None, optional
            Proposal to lower into a canonical request.
        proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
            Optional request metadata used when lowering ``proposal``.
        candidate : ObjectiveVectorRecordCandidateT
            Candidate evaluated by the request.
        objective_values : Sequence[float]
            Raw objective values.
        objective_scores : Sequence[float]
            Canonical minimization scores aligned with ``objective_values``.
        elapsed_seconds : float | None, optional
            Optional wall-clock runtime for the evaluation.
        """
        normalized_request = normalize_evaluation_request(
            request=request,
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )
        normalized_objective_values = normalize_objective_vector(
            values=objective_values,
            field_name="objective_values",
        )
        normalized_objective_scores = normalize_objective_vector(
            values=objective_scores,
            field_name="objective_scores",
        )
        super(ObjectiveVectorRecord, self).__init__(
            request=normalized_request,
            candidate=candidate,
        )
        object.__setattr__(self, "objective_values", normalized_objective_values)
        object.__setattr__(self, "objective_scores", normalized_objective_scores)
        object.__setattr__(self, "elapsed_seconds", elapsed_seconds)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate multi-objective record payloads.

        Raises
        ------
        ValueError
            If the objective vectors have different lengths or if
            ``elapsed_seconds`` is negative.
        """
        if len(self.objective_values) != len(self.objective_scores):
            msg = "objective_values and objective_scores must have the same length"
            raise ValueError(msg)

        if self.elapsed_seconds is not None and self.elapsed_seconds < 0.0:
            msg = "elapsed_seconds must be non-negative"
            raise ValueError(msg)

    @staticmethod
    def from_objective_values(
        *,
        request: EvaluationRequest[ObjectiveVectorCandidateT] | None = None,
        proposal: Proposal[ObjectiveVectorCandidateT] | None = None,
        proposal_evaluation_spec: ProposalEvaluationSpec | None = None,
        candidate: ObjectiveVectorCandidateT,
        objective_values: Sequence[float],
        directions: Sequence[OptimizationDirection],
        elapsed_seconds: float | None = None,
    ) -> "ObjectiveVectorRecord[ObjectiveVectorCandidateT]":
        """Build a vector record from raw objective values.

        Parameters
        ----------
        request : EvaluationRequest[ObjectiveVectorCandidateT] | None, optional
            Existing canonical request.
        proposal : Proposal[ObjectiveVectorCandidateT] | None, optional
            Proposal to lower into a canonical request.
        proposal_evaluation_spec : ProposalEvaluationSpec | None, optional
            Optional request metadata used when lowering ``proposal``.
        candidate : ObjectiveVectorCandidateT
            Candidate evaluated by the request.
        objective_values : Sequence[float]
            Raw objective values.
        directions : Sequence[OptimizationDirection]
            Direction for each objective value.
        elapsed_seconds : float | None, optional
            Optional wall-clock runtime for the evaluation.

        Returns
        -------
        ObjectiveVectorRecord[ObjectiveVectorCandidateT]
            Vector-valued record with normalized objective scores.

        Raises
        ------
        ValueError
            If ``directions`` does not align with ``objective_values``.
        """
        normalized_objective_values = normalize_objective_vector(
            values=objective_values,
            field_name="objective_values",
        )
        normalized_directions = tuple(directions)
        if len(normalized_directions) != len(normalized_objective_values):
            msg = "directions must align with objective_values"
            raise ValueError(msg)

        objective_scores = tuple(
            direction.normalize_objective_value(value)
            for value, direction in zip(
                normalized_objective_values,
                normalized_directions,
                strict=True,
            )
        )
        return ObjectiveVectorRecord(
            request=request,
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
            candidate=candidate,
            objective_values=normalized_objective_values,
            objective_scores=objective_scores,
            elapsed_seconds=elapsed_seconds,
        )
