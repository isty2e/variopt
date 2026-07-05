"""Candidate-refinement provenance artifact definitions."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeGuard

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..spaces import CandidateEquality, LeafPath
from ..spaces.equality import require_candidate_match
from ..typevars import CandidateT


def _is_leaf_path_container(value: object) -> TypeGuard[tuple[object, ...]]:
    return type(value) is tuple


def require_matching_refined_candidate(
    *,
    record_candidate: CandidateT,
    refined_candidate: CandidateT,
    mismatch_message: str,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> None:
    """Validate that refinement provenance matches an evaluated candidate.

    Parameters
    ----------
    record_candidate : CandidateT
        Candidate from the authoritative evaluation record.
    refined_candidate : CandidateT
        Candidate reported by candidate-refinement provenance.
    mismatch_message : str
        Error message used when candidates do not match.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate. When absent, strict scalar Python
        equality is used.

    Raises
    ------
    TypeError
        If equality does not produce a scalar truth value or if an explicit
        equality predicate does not return ``bool``.
    ValueError
        If the refined candidate does not match the evaluated record candidate.
    """
    require_candidate_match(
        left_candidate=record_candidate,
        right_candidate=refined_candidate,
        mismatch_message=mismatch_message,
        candidate_equal=candidate_equal,
    )


def _normalize_changed_leaf_paths(
    changed_leaf_paths: object,
) -> tuple[LeafPath, ...]:
    if isinstance(changed_leaf_paths, (str, bytes, bytearray)):
        msg = "changed_leaf_paths must be a sequence of leaf-path sequences"
        raise TypeError(msg)

    if not isinstance(changed_leaf_paths, Sequence):
        msg = "changed_leaf_paths must be a sequence of leaf-path sequences"
        raise TypeError(msg)

    path_sequence: Sequence[object] = changed_leaf_paths
    normalized_paths: list[LeafPath] = []
    for path in path_sequence:
        if not _is_leaf_path_container(path):
            msg = "changed_leaf_paths must contain only tuple leaf paths"
            raise TypeError(msg)

        normalized_path_segments: list[int | str] = []
        for segment in path:
            if type(segment) is not int and type(segment) is not str:
                msg = "changed_leaf_paths must contain only int or str path segments"
                raise TypeError(msg)
            normalized_path_segments.append(segment)
        normalized_paths.append(tuple(normalized_path_segments))

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
            If any leaf path is not a tuple or any path segment is not a
            canonical ``int`` or ``str``.
        ValueError
            If ``changed_leaf_paths`` contains duplicate paths.
        """
        object.__setattr__(self, "__orig_class__", None)
        object.__setattr__(self, "source_candidate", source_candidate)
        object.__setattr__(self, "refined_candidate", refined_candidate)
        object.__setattr__(
            self,
            "changed_leaf_paths",
            _normalize_changed_leaf_paths(changed_leaf_paths),
        )
