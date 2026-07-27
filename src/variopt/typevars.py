"""Internal generic type variables for variopt interfaces."""

from typing import TypeVar

CandidateT = TypeVar("CandidateT")
EvaluationT = TypeVar("EvaluationT")
EvaluationRequestT = TypeVar("EvaluationRequestT")
InputT_contra = TypeVar("InputT_contra", contravariant=True)
ObservationT = TypeVar("ObservationT")
ProblemT = TypeVar("ProblemT")
ProposalT = TypeVar("ProposalT")
RunMethodStateT = TypeVar("RunMethodStateT")
