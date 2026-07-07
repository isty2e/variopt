"""Internal generic type variables for variopt interfaces."""

from typing import TypeVar

CandidateT = TypeVar("CandidateT")
EvaluationT = TypeVar("EvaluationT")
EvaluationRequestT = TypeVar("EvaluationRequestT")
InputT = TypeVar("InputT", contravariant=True)
ObservationT = TypeVar("ObservationT")
ProblemT = TypeVar("ProblemT")
ProposalT = TypeVar("ProposalT")
RunMethodStateT = TypeVar("RunMethodStateT")
