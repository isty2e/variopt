"""Structured search-space contract and path types."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import (
    Generic,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np

from .base import SearchSpace
from .types import SpaceCandidateValue, SpaceScalarValue

PathSegment: TypeAlias = int | str
LeafPath: TypeAlias = tuple[PathSegment, ...]

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)
SearchBoundaryT = TypeVar("SearchBoundaryT")
SearchCandidateT = TypeVar("SearchCandidateT")


def is_space_scalar_value(value: object) -> TypeGuard[SpaceScalarValue]:
    """Return whether one object is a canonical scalar structured value.

    Parameters
    ----------
    value : object
        Value to classify.

    Returns
    -------
    TypeGuard[SpaceScalarValue]
        ``True`` when ``value`` is one of the canonical scalar structured
        candidate variants.
    """
    return type(value) in {bool, int, float, str, bytes, bytearray}


def is_space_candidate_value(value: object) -> TypeGuard[SpaceCandidateValue]:
    """Return whether one object is a canonical structured candidate value.

    Parameters
    ----------
    value : object
        Value to classify.

    Returns
    -------
    TypeGuard[SpaceCandidateValue]
        ``True`` when ``value`` is a canonical scalar, tuple, or string-keyed
        mapping whose nested values are also canonical structured candidates.
    """
    if is_space_scalar_value(value):
        return True

    if isinstance(value, tuple):
        tuple_value = cast(tuple[object, ...], value)
        return all(is_space_candidate_value(item) for item in tuple_value)

    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[object, object], value)
        return all(
            isinstance(key, str) and is_space_candidate_value(item)
            for key, item in mapping_value.items()
        )

    return False


def require_space_candidate_value(
    value: object,
    *,
    operation: str,
) -> SpaceCandidateValue:
    """Return one canonical structured candidate value or raise.

    Parameters
    ----------
    value : object
        Value to validate.
    operation : str
        Human-readable operation name used in the error message.

    Returns
    -------
    SpaceCandidateValue
        Canonical structured candidate value.

    Raises
    ------
    TypeError
        If ``value`` is not a canonical structured candidate value.
    """
    if not is_space_candidate_value(value):
        msg = (
            f"{operation} requires a canonical structured candidate value "
            f"composed of scalars, tuples, and string-keyed mappings"
        )
        raise TypeError(msg)
    return value


def is_structured_candidate_space(
    space: SearchSpace[SearchBoundaryT, SearchCandidateT],
) -> "TypeGuard[StructuredSearchSpace[SearchBoundaryT, SpaceCandidateValue]]":
    """Return whether one generic search space exposes structured-candidate semantics.

    Parameters
    ----------
    space : SearchSpace[SearchBoundaryT, SearchCandidateT]
        Search space to classify.

    Returns
    -------
    TypeGuard[StructuredSearchSpace[SearchBoundaryT, SpaceCandidateValue]]
        ``True`` when ``space`` is a structured search space over canonical
        recursive candidate values.
    """
    return isinstance(space, StructuredSearchSpace)


@runtime_checkable
class StructuredLeafSpace(Protocol):
    """Leaf-space protocol used by structured candidate editors.

    Notes
    -----
    Structured traversal and replacement operate on leaf spaces through this
    minimal sampling contract. Concrete built-in leaf spaces, such as
    :class:`~variopt.spaces.RealSpace` and
    :class:`~variopt.spaces.IntegerSpace`, satisfy the protocol directly.
    """

    def sample(self, random_state: np.random.RandomState) -> SpaceCandidateValue:
        """Sample one canonical leaf value.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state object that owns all stochasticity for the sample.

        Returns
        -------
        SpaceCandidateValue
            Canonical leaf value.
        """
        ...


class StructuredSearchSpace(
    SearchSpace[BoundaryT, CandidateT],
    ABC,
    Generic[BoundaryT, CandidateT],
):
    """Candidate-preserving structured search-space contract.

    Notes
    -----
    Structured spaces expose editable leaf topology on top of the generic
    :class:`~variopt.spaces.SearchSpace` contract. They are the basis for
    structured local search, geometry compilation, and leafwise mutation.
    """

    def has_static_topology(self) -> bool:
        """Return whether active leaf topology is candidate-invariant.

        Returns
        -------
        bool
            ``True`` when every canonical candidate activates the same declared
            leaf set.

        Notes
        -----
        Built-in structured spaces are topology-static by default. Conditional
        or hierarchical spaces should override this method and return
        ``False``.
        """
        return True

    @abstractmethod
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return all declared editable leaf paths.

        Returns
        -------
        tuple[LeafPath, ...]
            Canonical topology order of editable leaf paths.
        """

    def active_leaf_paths(
        self,
        candidate: CandidateT,
    ) -> tuple[LeafPath, ...]:
        """Return candidate-conditioned active leaf paths.

        Parameters
        ----------
        candidate : CandidateT
            Canonical candidate whose active topology should be queried.

        Returns
        -------
        tuple[LeafPath, ...]
            Active leaf paths in canonical order.

        Notes
        -----
        The default structured-space basis is trivial: every declared path is
        active for every canonical candidate. Conditional or hierarchical
        structured spaces should override this method with topology-aware
        behavior.
        """
        self.validate(candidate)
        return self.leaf_paths()

    def is_active_leaf_path(
        self,
        candidate: CandidateT,
        path: LeafPath,
    ) -> bool:
        """Return whether a declared leaf path is active.

        Parameters
        ----------
        candidate : CandidateT
            Canonical candidate whose active topology should be queried.
        path : LeafPath
            Declared leaf path to test.

        Returns
        -------
        bool
            ``True`` when ``path`` is active for ``candidate``.
        """
        return path in set(self.active_leaf_paths(candidate))

    @abstractmethod
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the declared leaf space at a path.

        Parameters
        ----------
        path : LeafPath
            Declared leaf path.

        Returns
        -------
        StructuredLeafSpace
            Leaf-space object that owns values stored at ``path``.
        """

    @abstractmethod
    def leaf_value_at_path(
        self,
        candidate: CandidateT,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the canonical leaf value stored at a path.

        Parameters
        ----------
        candidate : CandidateT
            Canonical candidate to read from.
        path : LeafPath
            Declared leaf path to inspect.

        Returns
        -------
        SpaceCandidateValue
            Canonical leaf value stored at ``path``.
        """

    @abstractmethod
    def replace_leaf_values(
        self,
        candidate: CandidateT,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> CandidateT:
        """Return a candidate with leaf replacements applied.

        Parameters
        ----------
        candidate : CandidateT
            Canonical candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by declared leaf path.

        Returns
        -------
        CandidateT
            Canonical candidate with the supplied leaf values replaced.
        """
