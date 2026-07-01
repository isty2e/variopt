"""Candidate-refinement provenance artifact definitions."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..spaces import LeafPath
from ..typevars import CandidateT


def require_scalar_candidate_equality(
    *,
    record_candidate: object,
    refined_candidate: object,
    mismatch_message: str,
) -> None:
    """Validate that candidate equality is scalar and true.

    Parameters
    ----------
    record_candidate : object
        Candidate from the authoritative evaluation record.
    refined_candidate : object
        Candidate reported by candidate-refinement provenance.
    mismatch_message : str
        Error message used when equality is scalar but false.

    Raises
    ------
    TypeError
        If equality does not produce a scalar truth value.
    ValueError
        If the scalar equality result is false.
    """
    try:
        equality_result = record_candidate == refined_candidate
    except (TypeError, ValueError) as error:
        msg = "candidate equality must produce a scalar truth value"
        raise TypeError(msg) from error

    if type(equality_result) is not bool and getattr(equality_result, "shape", None) != ():
        msg = "candidate equality must produce a scalar truth value"
        raise TypeError(msg)

    try:
        candidates_match = bool(equality_result)
    except (TypeError, ValueError) as error:
        msg = "candidate equality must produce a scalar truth value"
        raise TypeError(msg) from error

    if not candidates_match:
        raise ValueError(mismatch_message)


def _normalize_changed_leaf_paths(
    changed_leaf_paths: Sequence[LeafPath],
) -> tuple[LeafPath, ...]:
    normalized_paths: list[LeafPath] = []
    for path in changed_leaf_paths:
        normalized_path = tuple(path)
        for segment in normalized_path:
            if type(segment) is not int and type(segment) is not str:
                msg = "changed_leaf_paths must contain only int or str path segments"
                raise TypeError(msg)
        normalized_paths.append(normalized_path)

    if len(set(normalized_paths)) != len(normalized_paths):
        msg = "changed_leaf_paths must not contain duplicate paths"
        raise ValueError(msg)

    return tuple(normalized_paths)


@dataclass(frozen=True, slots=True, init=False)
class CandidateRefinement(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Execution provenance for a candidate transformed before evaluation.

    Parameters
    ----------
    source_candidate : CandidateT
        Candidate requested by the caller or upstream search method before
        refinement.
    refined_candidate : CandidateT
        Candidate that was actually evaluated after refinement.
    changed_leaf_paths : Sequence[LeafPath], default=()
        Authoritative structured leaf paths whose canonical values changed
        during refinement. An empty sequence means the producer reports no
        changed structured leaf paths; absence of refinement should be
        represented by ``EvaluationOutcome.refinement is None``.
    """

    source_candidate: CandidateT
    refined_candidate: CandidateT
    changed_leaf_paths: tuple[LeafPath, ...] = ()

    def __init__(
        self,
        *,
        source_candidate: CandidateT,
        refined_candidate: CandidateT,
        changed_leaf_paths: Sequence[LeafPath] = (),
    ) -> None:
        """Create one canonical candidate-refinement payload.

        Parameters
        ----------
        source_candidate : CandidateT
            Candidate before execution-side refinement.
        refined_candidate : CandidateT
            Candidate after execution-side refinement.
        changed_leaf_paths : Sequence[LeafPath], default=()
            Authoritative structured leaf paths changed by refinement.

        Raises
        ------
        TypeError
            If any path segment is not a canonical ``int`` or ``str``.
        ValueError
            If ``changed_leaf_paths`` contains duplicate paths.
        """
        object.__setattr__(self, "source_candidate", source_candidate)
        object.__setattr__(self, "refined_candidate", refined_candidate)
        object.__setattr__(
            self,
            "changed_leaf_paths",
            _normalize_changed_leaf_paths(changed_leaf_paths),
        )
