"""Neighborhood primitives shared by structured local-search kernels."""

from dataclasses import dataclass
from typing import Generic, Literal, TypeGuard, TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....spaces import (
    CategoricalSpace,
    IntegerSpace,
    LeafPath,
    SearchSpace,
    SpaceCandidateValue,
    SpaceScalarValue,
    StructuredLeafSpace,
    StructuredSearchSpace,
)

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)
DiscreteLeafSpace = IntegerSpace | CategoricalSpace[SpaceScalarValue]
StructuredVariableNeighborhoodStageKind = Literal[
    "leafwise_first_improvement",
    "sampled_leafwise_first_improvement",
    "scheduled_single_then_pair",
]


def is_integer_leaf_space(space: StructuredLeafSpace) -> TypeGuard[IntegerSpace]:
    """Return whether one structured leaf space is a canonical integer leaf.

    Parameters
    ----------
    space : StructuredLeafSpace
        Leaf space to classify.

    Returns
    -------
    TypeGuard[IntegerSpace]
        ``True`` when ``space`` is an integer leaf space.
    """
    return isinstance(space, IntegerSpace)


def is_categorical_leaf_space(
    space: StructuredLeafSpace,
) -> TypeGuard[CategoricalSpace[SpaceScalarValue]]:
    """Return whether one structured leaf space is a canonical categorical leaf.

    Parameters
    ----------
    space : StructuredLeafSpace
        Leaf space to classify.

    Returns
    -------
    TypeGuard[CategoricalSpace[SpaceScalarValue]]
        ``True`` when ``space`` is categorical.
    """
    return isinstance(space, CategoricalSpace)


def is_space_scalar_value(value: SpaceCandidateValue) -> TypeGuard[SpaceScalarValue]:
    """Return whether one canonical candidate value is a scalar leaf value.

    Parameters
    ----------
    value : SpaceCandidateValue
        Candidate value to classify.

    Returns
    -------
    TypeGuard[SpaceScalarValue]
        ``True`` when ``value`` is a scalar leaf value accepted by categorical
        neighborhoods.
    """
    return type(value) in {bool, int, float, str, bytes, bytearray}


@dataclass(frozen=True, slots=True)
class StructuredDiscreteNeighborhood(FrozenGenericSlotsCompat, Generic[BoundaryT, StructuredCandidateT]):
    """Canonical neighborhood metadata for discrete structured local search.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space that owns the neighborhood.
    leaf_paths : tuple[tuple[int | str, ...], ...]
        Editable leaf paths in canonical traversal order.
    leaf_spaces : tuple[DiscreteLeafSpace, ...]
        Discrete leaf spaces aligned with ``leaf_paths``.
    """

    space: StructuredSearchSpace[BoundaryT, StructuredCandidateT]
    leaf_paths: tuple[tuple[int | str, ...], ...]
    leaf_spaces: tuple[DiscreteLeafSpace, ...]

    @classmethod
    def from_space(
        cls,
        space: SearchSpace[BoundaryT, StructuredCandidateT],
    ) -> "StructuredDiscreteNeighborhood[BoundaryT, StructuredCandidateT]":
        """Normalize one search space into a discrete structured neighborhood.

        Parameters
        ----------
        space : SearchSpace[BoundaryT, StructuredCandidateT]
            Search space to normalize.

        Returns
        -------
        StructuredDiscreteNeighborhood[BoundaryT, StructuredCandidateT]
            Prepared discrete neighborhood metadata.

        Raises
        ------
        TypeError
            If ``space`` is not a static-topology structured space with only
            integer or categorical leaves.
        ValueError
            If ``space`` exposes no editable leaf paths.
        """
        if not isinstance(space, StructuredSearchSpace):
            msg = (
                "structured hill climber requires a structured search space "
                "with IntegerSpace or CategoricalSpace leaves"
            )
            raise TypeError(msg)

        if not space.has_static_topology():
            msg = (
                "structured hill climber requires a structured search space "
                "with static topology"
            )
            raise TypeError(msg)

        leaf_paths = space.leaf_paths()
        if len(leaf_paths) == 0:
            msg = "structured hill climber requires at least one editable discrete leaf"
            raise ValueError(msg)

        leaf_spaces: list[DiscreteLeafSpace] = []
        for path in leaf_paths:
            leaf_space = space.leaf_space_at_path(path)
            if is_integer_leaf_space(leaf_space):
                leaf_spaces.append(leaf_space)
                continue

            if is_categorical_leaf_space(leaf_space):
                leaf_spaces.append(leaf_space)
                continue

            msg = (
                "structured hill climber requires every structured leaf to be "
                "an IntegerSpace or CategoricalSpace"
            )
            raise TypeError(msg)

        return cls(
            space=space,
            leaf_paths=leaf_paths,
            leaf_spaces=tuple(leaf_spaces),
        )


def integer_leaf_neighbors(
    space: IntegerSpace,
    current_value: int,
) -> tuple[int, ...]:
    """Return deterministic one-step integer neighbors around one leaf value.

    Parameters
    ----------
    space : IntegerSpace
        Integer leaf space that owns ``current_value``.
    current_value : int
        Canonical integer leaf value.

    Returns
    -------
    tuple[int, ...]
        Neighboring integer values one step away within the declared bounds.
    """
    space.validate(current_value)
    neighbors: list[int] = []
    if current_value > space.low:
        neighbors.append(current_value - 1)
    if current_value < space.high:
        neighbors.append(current_value + 1)
    return tuple(neighbors)


def categorical_leaf_neighbors(
    space: CategoricalSpace[SpaceScalarValue],
    current_value: SpaceScalarValue,
) -> tuple[SpaceScalarValue, ...]:
    """Return all alternative categorical values in declaration order.

    Parameters
    ----------
    space : CategoricalSpace[SpaceScalarValue]
        Categorical leaf space that owns ``current_value``.
    current_value : SpaceScalarValue
        Current categorical leaf value.

    Returns
    -------
    tuple[SpaceScalarValue, ...]
        Alternative categorical values in declaration order.
    """
    return space.alternatives(current_value)


def discrete_leaf_neighbors(
    space: DiscreteLeafSpace,
    current_value: SpaceCandidateValue,
) -> tuple[SpaceCandidateValue, ...]:
    """Return deterministic local neighbors for one canonical discrete leaf.

    Parameters
    ----------
    space : DiscreteLeafSpace
        Discrete leaf space that owns ``current_value``.
    current_value : SpaceCandidateValue
        Canonical leaf value to perturb locally.

    Returns
    -------
    tuple[SpaceCandidateValue, ...]
        Deterministic local neighbors for the discrete leaf.

    Raises
    ------
    TypeError
        If ``current_value`` does not match the canonical type required by
        ``space``.
    """
    if isinstance(space, IntegerSpace):
        if type(current_value) is not int:
            msg = "integer leaf value must be a canonical integer"
            raise TypeError(msg)
        return integer_leaf_neighbors(space, current_value)

    if not is_space_scalar_value(current_value):
        msg = "categorical leaf value must be a canonical scalar"
        raise TypeError(msg)

    return categorical_leaf_neighbors(space, current_value)


@dataclass(frozen=True, slots=True)
class StructuredDiscreteMove:
    """One canonical single-leaf move in a structured discrete neighborhood.

    Parameters
    ----------
    path : LeafPath
        Leaf path to update.
    replacement : SpaceCandidateValue
        Canonical replacement value for that leaf.
    """

    path: LeafPath
    replacement: SpaceCandidateValue


@dataclass(frozen=True, slots=True)
class SampledStructuredNeighborhood:
    """Sampled single-leaf neighborhood and its coverage status.

    Parameters
    ----------
    moves : tuple[StructuredDiscreteMove, ...]
        Sampled single-leaf moves.
    covers_full_neighborhood : bool
        Whether the sample covers the full single-leaf neighborhood.
    """

    moves: tuple[StructuredDiscreteMove, ...]
    covers_full_neighborhood: bool


@dataclass(frozen=True, slots=True)
class StructuredKickPolicy:
    """Perturbation policy for structured iterated local search.

    Parameters
    ----------
    kick_leaf_count : int, default=2
        Number of distinct leaf positions perturbed in each kick.
    max_categorical_alternatives_per_leaf : int | None, default=None
        Optional cap on the number of categorical alternatives sampled per
        kicked leaf.
    """

    kick_leaf_count: int = 2
    max_categorical_alternatives_per_leaf: int | None = None

    def __post_init__(self) -> None:
        """Validate structured kick-policy metadata.

        Raises
        ------
        ValueError
            Raised when the kick size or categorical cap is invalid.
        """
        if self.kick_leaf_count <= 0:
            msg = "kick_leaf_count must be positive"
            raise ValueError(msg)

        if (
            self.max_categorical_alternatives_per_leaf is not None
            and self.max_categorical_alternatives_per_leaf <= 0
        ):
            msg = "max_categorical_alternatives_per_leaf must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StructuredVariableNeighborhoodStage:
    """One neighborhood stage in structured variable-neighborhood search.

    Parameters
    ----------
    kind : StructuredVariableNeighborhoodStageKind
        Neighborhood family attempted by this stage.
    max_neighbors_per_step : int | None, default=None
        Per-step neighbor cap for sampled leafwise stages.
    max_categorical_neighbors_per_leaf : int | None, default=None
        Optional categorical cap for sampled leafwise stages.
    pair_move_leaf_limit : int | None, default=None
        Number of leading scheduled leaves considered for pair moves in
        scheduled stages.
    """

    kind: StructuredVariableNeighborhoodStageKind
    max_neighbors_per_step: int | None = None
    max_categorical_neighbors_per_leaf: int | None = None
    pair_move_leaf_limit: int | None = None

    def __post_init__(self) -> None:
        """Validate stage metadata against the chosen neighborhood kind.

        Raises
        ------
        ValueError
            Raised when the configured metadata does not match the stage kind.
        """
        if self.kind == "leafwise_first_improvement":
            if self.max_neighbors_per_step is not None:
                msg = (
                    "leafwise_first_improvement stages must not set "
                    "max_neighbors_per_step"
                )
                raise ValueError(msg)
            if self.max_categorical_neighbors_per_leaf is not None:
                msg = (
                    "leafwise_first_improvement stages must not set "
                    "max_categorical_neighbors_per_leaf"
                )
                raise ValueError(msg)
            if self.pair_move_leaf_limit is not None:
                msg = (
                    "leafwise_first_improvement stages must not set "
                    "pair_move_leaf_limit"
                )
                raise ValueError(msg)
            return

        if self.kind == "sampled_leafwise_first_improvement":
            if self.max_neighbors_per_step is None:
                msg = (
                    "sampled_leafwise_first_improvement stages require "
                    "max_neighbors_per_step"
                )
                raise ValueError(msg)
            if self.max_neighbors_per_step <= 0:
                msg = "max_neighbors_per_step must be positive"
                raise ValueError(msg)
            if (
                self.max_categorical_neighbors_per_leaf is not None
                and self.max_categorical_neighbors_per_leaf <= 0
            ):
                msg = "max_categorical_neighbors_per_leaf must be positive"
                raise ValueError(msg)
            if self.pair_move_leaf_limit is not None:
                msg = (
                    "sampled_leafwise_first_improvement stages must not set "
                    "pair_move_leaf_limit"
                )
                raise ValueError(msg)
            return

        if self.kind == "scheduled_single_then_pair":
            if self.pair_move_leaf_limit is None:
                msg = "scheduled_single_then_pair stages require pair_move_leaf_limit"
                raise ValueError(msg)
            if self.pair_move_leaf_limit <= 0:
                msg = "pair_move_leaf_limit must be positive"
                raise ValueError(msg)
            if self.max_neighbors_per_step is not None:
                msg = (
                    "scheduled_single_then_pair stages must not set "
                    "max_neighbors_per_step"
                )
                raise ValueError(msg)
            if self.max_categorical_neighbors_per_leaf is not None:
                msg = (
                    "scheduled_single_then_pair stages must not set "
                    "max_categorical_neighbors_per_leaf"
                )
                raise ValueError(msg)
            return

        msg = f"unsupported structured variable-neighborhood stage: {self.kind!r}"
        raise ValueError(msg)

    @classmethod
    def leafwise_first_improvement(cls) -> "StructuredVariableNeighborhoodStage":
        """Build a deterministic single-leaf neighborhood stage.

        Returns
        -------
        StructuredVariableNeighborhoodStage
            Stage configured for deterministic leafwise first improvement.
        """
        return cls(kind="leafwise_first_improvement")

    @classmethod
    def sampled_leafwise_first_improvement(
        cls,
        *,
        max_neighbors_per_step: int,
        max_categorical_neighbors_per_leaf: int | None = None,
    ) -> "StructuredVariableNeighborhoodStage":
        """Build a bounded sampled single-leaf neighborhood stage.

        Parameters
        ----------
        max_neighbors_per_step : int
            Maximum number of sampled moves considered at each step.
        max_categorical_neighbors_per_leaf : int | None, default=None
            Optional cap on categorical alternatives sampled for each leaf.

        Returns
        -------
        StructuredVariableNeighborhoodStage
            Stage configured for bounded stochastic leafwise search.
        """
        return cls(
            kind="sampled_leafwise_first_improvement",
            max_neighbors_per_step=max_neighbors_per_step,
            max_categorical_neighbors_per_leaf=max_categorical_neighbors_per_leaf,
        )

    @classmethod
    def scheduled_single_then_pair(
        cls,
        *,
        pair_move_leaf_limit: int,
    ) -> "StructuredVariableNeighborhoodStage":
        """Build a scheduled single-then-pair neighborhood stage.

        Parameters
        ----------
        pair_move_leaf_limit : int
            Number of leading scheduled leaves considered for pair moves.

        Returns
        -------
        StructuredVariableNeighborhoodStage
            Stage configured for deterministic single-then-pair search.
        """
        return cls(
            kind="scheduled_single_then_pair",
            pair_move_leaf_limit=pair_move_leaf_limit,
        )
