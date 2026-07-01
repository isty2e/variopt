"""Candidate equality helpers for search-space contracts."""

from collections.abc import Callable
from typing import TypeAlias

from ..typevars import CandidateT

CandidateEquality: TypeAlias = Callable[[CandidateT, CandidateT], bool]


def scalar_candidate_equality(
    left_candidate: CandidateT,
    right_candidate: CandidateT,
) -> bool:
    """Return strict scalar equality for two candidates.

    Parameters
    ----------
    left_candidate : CandidateT
        Left candidate to compare.
    right_candidate : CandidateT
        Right candidate to compare.

    Returns
    -------
    bool
        Whether the candidates compare equal under scalar Python equality.

    Raises
    ------
    TypeError
        If candidate equality does not produce a scalar truth value.
    """
    try:
        equality_result = left_candidate == right_candidate
    except (TypeError, ValueError) as error:
        msg = "candidate equality must produce a scalar truth value"
        raise TypeError(msg) from error

    if type(equality_result) is not bool and getattr(equality_result, "shape", None) != ():
        msg = "candidate equality must produce a scalar truth value"
        raise TypeError(msg)

    try:
        return bool(equality_result)
    except (TypeError, ValueError) as error:
        msg = "candidate equality must produce a scalar truth value"
        raise TypeError(msg) from error


def require_candidate_match(
    *,
    left_candidate: CandidateT,
    right_candidate: CandidateT,
    mismatch_message: str,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> None:
    """Require two candidates to match under explicit equality semantics.

    Parameters
    ----------
    left_candidate : CandidateT
        Authoritative candidate to compare.
    right_candidate : CandidateT
        Candidate expected to match ``left_candidate``.
    mismatch_message : str
        Error message used when the candidates do not match.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate. When absent, strict scalar Python
        equality is used.

    Raises
    ------
    TypeError
        If the equality predicate does not return a canonical ``bool``.
    ValueError
        If the candidates do not match.
    """
    if candidate_equal is None:
        candidates_match = scalar_candidate_equality(left_candidate, right_candidate)
    else:
        candidates_match = candidate_equal(left_candidate, right_candidate)
        if type(candidates_match) is not bool:
            msg = "candidate equality predicate must return bool"
            raise TypeError(msg)

    if not candidates_match:
        raise ValueError(mismatch_message)
