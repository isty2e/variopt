"""Structured CSA variation-operator wrappers."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeGuard, TypeVar, cast

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....operators import VariationOperator
from .....spaces import LeafPath, SearchSpace, StructuredSearchSpace
from .....spaces.structured import require_space_candidate_value
from .....spaces.types import SpaceCandidateValue
from .crossover import uniform_crossover as _uniform_crossover
from .mutation import (
    bounded_mutation as _bounded_mutation,
)
from .mutation import (
    bounded_mutation_on_paths as _bounded_mutation_on_paths,
)
from .mutation import (
    random_reset_mutation as _random_reset_mutation,
)
from .mutation import (
    random_reset_mutation_on_paths as _random_reset_mutation_on_paths,
)
from .mutation import select_mutation_paths as _select_mutation_paths
from .validation import (
    require_parent_count,
    require_structured_space,
    validate_fraction,
)

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)


class StructuredPathMutationOperator(Protocol):
    """Capability protocol for structured unary operators over explicit leaf paths.

    Notes
    -----
    CSA ask-side planning only needs the operator's structured candidate-space
    view, its path-selection fraction cap, and the ability to apply one
    mutation on an explicit set of leaf paths. It does not need the concrete
    operator class or the exact structured candidate subtype.
    """

    @property
    def structured_candidate_space(
        self,
    ) -> StructuredSearchSpace[object, SpaceCandidateValue]:
        """Return the type-erased structured candidate space."""
        ...

    @property
    def max_selected_path_fraction(self) -> float:
        """Return the maximum fraction of active leaf paths that may be selected."""
        ...

    def select_validated_space_candidate_paths(
        self,
        candidate: SpaceCandidateValue,
        random_state: np.random.RandomState,
    ) -> tuple[LeafPath, ...]:
        """Select paths for an already validated candidate using native semantics."""
        ...

    def apply_space_candidate_on_paths(
        self,
        candidate: SpaceCandidateValue,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return a child by editing explicit paths on a canonical candidate value."""
        ...

    def apply_validated_space_candidate_on_paths(
        self,
        candidate: SpaceCandidateValue,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return a child by editing paths on an already validated candidate."""
        ...


class CovarianceGuidedStructuredMutationOperator(
    StructuredPathMutationOperator,
    Protocol,
):
    """Capability protocol for structured path mutation with numeric covariance guidance."""

    @property
    def max_coordinate_fraction(self) -> float:
        """Return the maximum coordinate-space fraction used by covariance guidance."""
        ...


class ValidatedParentVariationOperator(Protocol):
    """Capability protocol for applying built-in operators to validated parents."""

    @property
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        ...

    def apply_from_validated_parents(
        self,
        parents: Sequence[SpaceCandidateValue],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return a child without revalidating already validated parents."""
        ...


def is_structured_path_mutation_operator(
    operator: object,
) -> TypeGuard[StructuredPathMutationOperator]:
    """Return whether one exact built-in operator supports leaf-path editing."""
    operator_type = type(operator)
    return operator_type is RandomResetMutation or operator_type is BoundedMutation


def is_covariance_guided_structured_mutation_operator(
    operator: object,
) -> TypeGuard[CovarianceGuidedStructuredMutationOperator]:
    """Return whether one exact built-in operator supports covariance guidance."""
    return type(operator) is BoundedMutation


def is_validated_parent_variation_operator(
    operator: object,
) -> TypeGuard[ValidatedParentVariationOperator]:
    """Return whether one exact built-in operator supports validated-parent application."""
    operator_type = type(operator)
    return (
        operator_type is UniformCrossover
        or operator_type is RandomResetMutation
        or operator_type is BoundedMutation
    )


@dataclass(frozen=True, slots=True)
class UniformCrossover(
    FrozenGenericSlotsCompat,
    VariationOperator[StructuredCandidateT],
    Generic[BoundaryT, StructuredCandidateT],
):
    """Copy a random subset of partner leaves into the primary parent.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Search space used to validate parents and resulting children.
    max_exchange_fraction : float, default=0.5
        Maximum fraction of active leaves copied from the partner.
    """

    space: SearchSpace[BoundaryT, StructuredCandidateT]
    structured_space: StructuredSearchSpace[BoundaryT, StructuredCandidateT] = field(
        init=False,
        repr=False,
    )
    max_exchange_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Validate the crossover configuration."""
        validate_fraction(
            self.max_exchange_fraction,
            name="max_exchange_fraction",
        )
        object.__setattr__(
            self,
            "structured_space",
            require_structured_space(self.space),
        )

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 2

    @override
    def apply(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a crossover child built from two parent candidates.

        Parameters
        ----------
        parents : Sequence[StructuredCandidateT]
            Parent candidates in ``(primary, partner)`` order.
        random_state : np.random.RandomState
            Random state used for leaf-selection decisions.

        Returns
        -------
        StructuredCandidateT
            Child candidate after structured uniform crossover.

        Raises
        ------
        ValueError
            If ``parents`` does not match the required arity.
        TypeError
            If the configured search space is not structured.
        """
        require_parent_count(parents, arity=self.arity)
        child = _uniform_crossover(
            space=self.structured_space,
            primary_parent=parents[0],
            partner_parent=parents[1],
            max_exchange_fraction=self.max_exchange_fraction,
            random_state=random_state,
        )
        self.space.validate(child)
        return child

    def apply_from_validated_parents(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a crossover child for parents already validated by CSA state."""
        require_parent_count(parents, arity=self.arity)
        return _uniform_crossover(
            space=self.structured_space,
            primary_parent=parents[0],
            partner_parent=parents[1],
            max_exchange_fraction=self.max_exchange_fraction,
            random_state=random_state,
            validate_parents=False,
        )


@dataclass(frozen=True, slots=True)
class RandomResetMutation(
    FrozenGenericSlotsCompat,
    VariationOperator[StructuredCandidateT],
    Generic[BoundaryT, StructuredCandidateT],
):
    """Resample a random subset of candidate leaves from the declared space.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Search space used to validate parents and resulting children.
    max_exchange_fraction : float, default=0.2
        Maximum fraction of active leaves eligible for random reset.
    """

    space: SearchSpace[BoundaryT, StructuredCandidateT]
    structured_space: StructuredSearchSpace[BoundaryT, StructuredCandidateT] = field(
        init=False,
        repr=False,
    )
    max_exchange_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Validate the mutation configuration."""
        validate_fraction(
            self.max_exchange_fraction,
            name="max_exchange_fraction",
        )
        object.__setattr__(
            self,
            "structured_space",
            require_structured_space(self.space),
        )

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 1

    @override
    def apply(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a child formed by leaf-wise resampling.

        Parameters
        ----------
        parents : Sequence[StructuredCandidateT]
            Parent candidates. Exactly one parent is required.
        random_state : np.random.RandomState
            Random state used for path selection and resampling.

        Returns
        -------
        StructuredCandidateT
            Mutation child after random-reset resampling.

        Raises
        ------
        ValueError
            If ``parents`` does not match the required arity.
        TypeError
            If the configured search space is not structured.
        """
        require_parent_count(parents, arity=self.arity)
        child = _random_reset_mutation(
            space=self.structured_space,
            candidate=parents[0],
            max_exchange_fraction=self.max_exchange_fraction,
            random_state=random_state,
        )
        self.space.validate(child)
        return child

    def apply_from_validated_parents(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a reset child for parents already validated by CSA state."""
        require_parent_count(parents, arity=self.arity)
        selected_paths = self.select_validated_space_candidate_paths(
            parents[0],
            random_state,
        )
        if len(selected_paths) == 0:
            return parents[0]
        return _random_reset_mutation_on_paths(
            space=self.structured_space,
            candidate=parents[0],
            selected_paths=selected_paths,
            random_state=random_state,
            validate_candidate=False,
        )

    def apply_on_paths(
        self,
        candidate: StructuredCandidateT,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a mutation child over one explicit set of structured leaves.

        Parameters
        ----------
        candidate : StructuredCandidateT
            Candidate to mutate.
        selected_paths : Sequence[LeafPath]
            Explicit leaf paths to resample.
        random_state : np.random.RandomState
            Random state used for leaf-value resampling.

        Returns
        -------
        StructuredCandidateT
            Mutation child restricted to ``selected_paths``.
        """
        child = _random_reset_mutation_on_paths(
            space=self.structured_space,
            candidate=candidate,
            selected_paths=selected_paths,
            random_state=random_state,
        )
        self.space.validate(child)
        return child

    @property
    def structured_candidate_space(
        self,
    ) -> StructuredSearchSpace[object, SpaceCandidateValue]:
        """Return the structured candidate-space view used by ask-side planning."""
        return cast(
            StructuredSearchSpace[object, SpaceCandidateValue],
            self.structured_space,
        )

    @property
    def max_selected_path_fraction(self) -> float:
        """Return the maximum fraction of active leaf paths eligible for reset."""
        return self.max_exchange_fraction

    def select_validated_space_candidate_paths(
        self,
        candidate: SpaceCandidateValue,
        random_state: np.random.RandomState,
    ) -> tuple[LeafPath, ...]:
        """Select reset paths using the operator's native unweighted distribution."""
        return _select_mutation_paths(
            space=self.structured_candidate_space,
            candidate=candidate,
            max_exchange_fraction=self.max_exchange_fraction,
            random_state=random_state,
            validate_candidate=False,
        )

    def apply_space_candidate_on_paths(
        self,
        candidate: SpaceCandidateValue,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return one reset child from a type-erased structured candidate value."""
        candidate_value = require_space_candidate_value(
            candidate,
            operation="RandomResetMutation path editing",
        )
        structured_space = self.structured_candidate_space
        structured_space.validate(candidate_value)
        child = self.apply_validated_space_candidate_on_paths(
            candidate=candidate_value,
            selected_paths=selected_paths,
            random_state=random_state,
        )
        structured_space.validate(child)
        return child

    def apply_validated_space_candidate_on_paths(
        self,
        candidate: SpaceCandidateValue,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return one reset child from a validated structured candidate value."""
        return _random_reset_mutation_on_paths(
            space=self.structured_candidate_space,
            candidate=candidate,
            selected_paths=selected_paths,
            random_state=random_state,
            validate_candidate=False,
        )


@dataclass(frozen=True, slots=True)
class BoundedMutation(
    FrozenGenericSlotsCompat,
    VariationOperator[StructuredCandidateT],
    Generic[BoundaryT, StructuredCandidateT],
):
    """Apply bounded perturbations to a random subset of candidate leaves.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Search space used to validate parents and resulting children.
    max_perturbation_fraction : float, default=0.2
        Maximum fraction of each numeric leaf range used during perturbation.
    """

    space: SearchSpace[BoundaryT, StructuredCandidateT]
    structured_space: StructuredSearchSpace[BoundaryT, StructuredCandidateT] = field(
        init=False,
        repr=False,
    )
    max_perturbation_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Validate the mutation configuration."""
        validate_fraction(
            self.max_perturbation_fraction,
            name="max_perturbation_fraction",
        )
        object.__setattr__(
            self,
            "structured_space",
            require_structured_space(self.space),
        )

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 1

    @override
    def apply(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a child formed by bounded leaf-wise perturbations.

        Parameters
        ----------
        parents : Sequence[StructuredCandidateT]
            Parent candidates. Exactly one parent is required.
        random_state : np.random.RandomState
            Random state used for path selection and perturbation sampling.

        Returns
        -------
        StructuredCandidateT
            Mutation child after bounded perturbation.

        Raises
        ------
        ValueError
            If ``parents`` does not match the required arity.
        TypeError
            If the configured search space is not structured.
        """
        require_parent_count(parents, arity=self.arity)
        child = _bounded_mutation(
            space=self.structured_space,
            candidate=parents[0],
            max_perturbation_fraction=self.max_perturbation_fraction,
            random_state=random_state,
        )
        self.space.validate(child)
        return child

    def apply_from_validated_parents(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a bounded-mutation child for parents already validated by CSA state."""
        require_parent_count(parents, arity=self.arity)
        selected_paths = self.select_validated_space_candidate_paths(
            parents[0],
            random_state,
        )
        if len(selected_paths) == 0:
            return parents[0]
        return _bounded_mutation_on_paths(
            space=self.structured_space,
            candidate=parents[0],
            selected_paths=selected_paths,
            max_perturbation_fraction=self.max_perturbation_fraction,
            random_state=random_state,
            validate_candidate=False,
        )

    def apply_on_paths(
        self,
        candidate: StructuredCandidateT,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a mutation child over one explicit set of structured leaves.

        Parameters
        ----------
        candidate : StructuredCandidateT
            Candidate to mutate.
        selected_paths : Sequence[LeafPath]
            Explicit leaf paths to perturb.
        random_state : np.random.RandomState
            Random state used for perturbation sampling.

        Returns
        -------
        StructuredCandidateT
            Mutation child restricted to ``selected_paths``.
        """
        child = _bounded_mutation_on_paths(
            space=self.structured_space,
            candidate=candidate,
            selected_paths=selected_paths,
            max_perturbation_fraction=self.max_perturbation_fraction,
            random_state=random_state,
        )
        self.space.validate(child)
        return child

    @property
    def structured_candidate_space(
        self,
    ) -> StructuredSearchSpace[object, SpaceCandidateValue]:
        """Return the structured candidate-space view used by ask-side planning."""
        return cast(
            StructuredSearchSpace[object, SpaceCandidateValue],
            self.structured_space,
        )

    @property
    def max_selected_path_fraction(self) -> float:
        """Return the maximum fraction of active leaf paths eligible for perturbation."""
        return self.max_perturbation_fraction

    def select_validated_space_candidate_paths(
        self,
        candidate: SpaceCandidateValue,
        random_state: np.random.RandomState,
    ) -> tuple[LeafPath, ...]:
        """Select bounded-mutation paths using the native unweighted distribution."""
        return _select_mutation_paths(
            space=self.structured_candidate_space,
            candidate=candidate,
            max_exchange_fraction=self.max_perturbation_fraction,
            random_state=random_state,
            validate_candidate=False,
        )

    @property
    def max_coordinate_fraction(self) -> float:
        """Return the maximum coordinate-space fraction used for covariance guidance."""
        return self.max_perturbation_fraction

    def apply_space_candidate_on_paths(
        self,
        candidate: SpaceCandidateValue,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return one bounded-mutation child from a type-erased structured candidate value."""
        candidate_value = require_space_candidate_value(
            candidate,
            operation="BoundedMutation path editing",
        )
        structured_space = self.structured_candidate_space
        structured_space.validate(candidate_value)
        child = self.apply_validated_space_candidate_on_paths(
            candidate=candidate_value,
            selected_paths=selected_paths,
            random_state=random_state,
        )
        structured_space.validate(child)
        return child

    def apply_validated_space_candidate_on_paths(
        self,
        candidate: SpaceCandidateValue,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return one bounded child from a validated structured candidate value."""
        return _bounded_mutation_on_paths(
            space=self.structured_candidate_space,
            candidate=candidate,
            selected_paths=selected_paths,
            max_perturbation_fraction=self.max_perturbation_fraction,
            random_state=random_state,
            validate_candidate=False,
        )
