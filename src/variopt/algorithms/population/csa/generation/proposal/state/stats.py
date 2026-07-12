"""Accumulated statistics for CSA proposal adaptation."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite

from typing_extensions import Self

from .......json_types import (
    JSONDict,
    JSONValue,
    require_json_field,
    require_json_finite_float,
    require_json_int,
    require_json_int_or_str,
    require_json_list,
    require_json_str,
)
from .......spaces import LeafPath
from .attribution import NumericSubspaceDisplacement
from .generation_evidence import (
    ProposalFamilyAdaptationSummary,
    ProposalLeafAdaptationSummary,
)


def _leaf_path_to_json(path: LeafPath) -> list[JSONValue]:
    return [segment for segment in path]


def _leaf_path_from_json(
    value: JSONValue, *, field_name: str = "leaf path"
) -> LeafPath:
    raw_segments = require_json_list(value, field_name=field_name)
    segments: list[int | str] = []
    for raw_position, raw_segment in enumerate(raw_segments):
        segments.append(
            require_json_int_or_str(
                raw_segment,
                field_name=f"{field_name}[{raw_position}]",
            ),
        )
    return tuple(segments)


def _leaf_paths_to_json(leaf_paths: tuple[LeafPath, ...]) -> list[JSONValue]:
    return [_leaf_path_to_json(path) for path in leaf_paths]


def _leaf_paths_from_json(
    value: JSONValue,
    *,
    field_name: str = "leaf path family",
) -> tuple[LeafPath, ...]:
    raw_paths = require_json_list(value, field_name=field_name)
    return tuple(
        _leaf_path_from_json(raw_path, field_name=f"{field_name}[{raw_position}]")
        for raw_position, raw_path in enumerate(raw_paths)
    )


@dataclass(frozen=True, slots=True)
class ProposalNumericSubspaceCovarianceStat:
    """Discounted covariance moments for one numeric structured leaf family.

    Parameters
    ----------
    leaf_paths : tuple[LeafPath, ...]
        Structured leaf family represented by this covariance estimate.
    observation_count : int, default=0
        Number of successful displacements accumulated into the statistic.
    discounted_weight : float, default=0.0
        Lazily decayed effective sample weight.
    discounted_displacement_sum : tuple[float, ...], default=()
        Lazily decayed first-moment accumulator.
    discounted_outer_product_sum : tuple[tuple[float, ...], ...], default=()
        Lazily decayed second-moment accumulator.
    last_update_index : int, default=0
        Reducer update index at which the accumulators were last materialized.
    """

    leaf_paths: tuple[LeafPath, ...]
    observation_count: int = 0
    discounted_weight: float = 0.0
    discounted_displacement_sum: tuple[float, ...] = ()
    discounted_outer_product_sum: tuple[tuple[float, ...], ...] = ()
    last_update_index: int = 0

    def __post_init__(self) -> None:
        """Normalize one canonical covariance-stat record."""
        normalized_leaf_paths = tuple(tuple(path) for path in self.leaf_paths)
        object.__setattr__(self, "leaf_paths", normalized_leaf_paths)
        object.__setattr__(
            self, "discounted_displacement_sum", tuple(self.discounted_displacement_sum)
        )
        object.__setattr__(
            self,
            "discounted_outer_product_sum",
            tuple(tuple(row) for row in self.discounted_outer_product_sum),
        )
        if type(self.observation_count) is not int:
            msg = "observation_count must be an int"
            raise TypeError(msg)
        if self.observation_count < 0:
            msg = "observation_count must be non-negative"
            raise ValueError(msg)
        if type(self.discounted_weight) is not float:
            msg = "discounted_weight must be a float"
            raise TypeError(msg)
        if not isfinite(self.discounted_weight):
            msg = "discounted_weight must be finite"
            raise ValueError(msg)
        if not 0.0 <= self.discounted_weight <= self.observation_count:
            msg = "discounted_weight must be bounded by observation_count"
            raise ValueError(msg)
        if type(self.last_update_index) is not int:
            msg = "last_update_index must be an int"
            raise TypeError(msg)
        if self.last_update_index < 0:
            msg = "last_update_index must be non-negative"
            raise ValueError(msg)
        if len(normalized_leaf_paths) == 0:
            msg = "numeric covariance stats require at least one leaf path"
            raise ValueError(msg)
        if len(self.discounted_displacement_sum) not in {0, len(normalized_leaf_paths)}:
            msg = "discounted_displacement_sum dimensions must match leaf_paths"
            raise ValueError(msg)
        if any(type(value) is not float for value in self.discounted_displacement_sum):
            msg = "discounted_displacement_sum values must be floats"
            raise TypeError(msg)
        if any(not isfinite(value) for value in self.discounted_displacement_sum):
            msg = "discounted_displacement_sum values must be finite"
            raise ValueError(msg)
        if len(self.discounted_outer_product_sum) not in {
            0,
            len(normalized_leaf_paths),
        }:
            msg = "discounted_outer_product_sum dimensions must match leaf_paths"
            raise ValueError(msg)
        for row in self.discounted_outer_product_sum:
            if len(row) != len(normalized_leaf_paths):
                msg = "discounted_outer_product_sum must be square"
                raise ValueError(msg)
            if any(type(value) is not float for value in row):
                msg = "discounted_outer_product_sum values must be floats"
                raise TypeError(msg)
            if any(not isfinite(value) for value in row):
                msg = "discounted_outer_product_sum values must be finite"
                raise ValueError(msg)
        if self.observation_count > 0 and self.discounted_weight == 0.0:
            msg = "observed covariance stats must have positive discounted_weight"
            raise ValueError(msg)
        if self.discounted_weight > 0.0 and (
            len(self.discounted_displacement_sum) != len(normalized_leaf_paths)
            or len(self.discounted_outer_product_sum) != len(normalized_leaf_paths)
        ):
            msg = "positive covariance weight requires complete moment accumulators"
            raise ValueError(msg)
        if self.discounted_weight == 0.0 and (
            any(value != 0.0 for value in self.discounted_displacement_sum)
            or any(
                value != 0.0
                for row in self.discounted_outer_product_sum
                for value in row
            )
        ):
            msg = "zero covariance weight requires zero moment accumulators"
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        """Return the numeric subspace dimension."""
        return len(self.leaf_paths)

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the covariance statistic.

        Returns
        -------
        JSONDict
            JSON-safe covariance-stat snapshot.
        """
        return {
            "leaf_paths": _leaf_paths_to_json(self.leaf_paths),
            "observation_count": self.observation_count,
            "discounted_weight": self.discounted_weight,
            "discounted_displacement_sum": list(self.discounted_displacement_sum),
            "discounted_outer_product_sum": [
                list(row) for row in self.discounted_outer_product_sum
            ],
            "last_update_index": self.last_update_index,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
    ) -> Self:
        """Build a covariance statistic from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe covariance-stat snapshot.

        Returns
        -------
        Self
            Reconstructed covariance statistic.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        raw_leaf_paths = require_json_field(data, "leaf_paths")
        observation_count = require_json_int(
            require_json_field(data, "observation_count"),
            field_name="observation_count",
        )
        discounted_weight = require_json_finite_float(
            require_json_field(data, "discounted_weight"),
            field_name="discounted_weight",
        )
        raw_displacement_sum = require_json_list(
            require_json_field(data, "discounted_displacement_sum"),
            field_name="discounted_displacement_sum",
        )
        raw_outer_product_sum = require_json_list(
            require_json_field(data, "discounted_outer_product_sum"),
            field_name="discounted_outer_product_sum",
        )
        last_update_index = require_json_int(
            require_json_field(data, "last_update_index"),
            field_name="last_update_index",
        )

        displacement_sum: list[float] = []
        for raw_index, raw_value in enumerate(raw_displacement_sum):
            displacement_sum.append(
                require_json_finite_float(
                    raw_value,
                    field_name=f"discounted_displacement_sum[{raw_index}]",
                ),
            )

        outer_product_sum: list[tuple[float, ...]] = []
        for raw_row_index, raw_row in enumerate(raw_outer_product_sum):
            row_values = require_json_list(
                raw_row,
                field_name=f"discounted_outer_product_sum[{raw_row_index}]",
            )
            row: list[float] = []
            for raw_column_index, raw_value in enumerate(row_values):
                row.append(
                    require_json_finite_float(
                        raw_value,
                        field_name=(
                            "discounted_outer_product_sum"
                            f"[{raw_row_index}][{raw_column_index}]"
                        ),
                    ),
                )
            outer_product_sum.append(tuple(row))

        return cls(
            leaf_paths=_leaf_paths_from_json(raw_leaf_paths, field_name="leaf_paths"),
            observation_count=observation_count,
            discounted_weight=discounted_weight,
            discounted_displacement_sum=tuple(displacement_sum),
            discounted_outer_product_sum=tuple(outer_product_sum),
            last_update_index=last_update_index,
        )

    def effective_weight(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> float:
        """Return the lazily decayed effective covariance weight.

        Parameters
        ----------
        current_update_index : int
            Reducer update index at which to materialize the decay.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        float
            Effective covariance weight after lazy decay.

        Raises
        ------
        ValueError
            If ``current_update_index`` is earlier than ``last_update_index``.
        """
        if current_update_index < self.last_update_index:
            msg = "current_update_index must not go backwards"
            raise ValueError(msg)
        elapsed_updates = current_update_index - self.last_update_index
        return self.discounted_weight * (adaptation_decay**elapsed_updates)

    def effective_displacement_sum(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> tuple[float, ...]:
        """Return the lazily decayed displacement sum.

        Parameters
        ----------
        current_update_index : int
            Reducer update index at which to materialize the decay.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        tuple[float, ...]
            Effective first-moment accumulator after lazy decay.
        """
        decay_factor = adaptation_decay ** max(
            0, current_update_index - self.last_update_index
        )
        return tuple(value * decay_factor for value in self.discounted_displacement_sum)

    def effective_outer_product_sum(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> tuple[tuple[float, ...], ...]:
        """Return the lazily decayed outer-product sum.

        Parameters
        ----------
        current_update_index : int
            Reducer update index at which to materialize the decay.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        tuple[tuple[float, ...], ...]
            Effective second-moment accumulator after lazy decay.
        """
        decay_factor = adaptation_decay ** max(
            0, current_update_index - self.last_update_index
        )
        return tuple(
            tuple(value * decay_factor for value in row)
            for row in self.discounted_outer_product_sum
        )

    def effective_mean(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> tuple[float, ...]:
        """Return the lazily decayed mean displacement vector.

        Parameters
        ----------
        current_update_index : int
            Reducer update index at which to materialize the decay.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        tuple[float, ...]
            Effective mean displacement vector for this numeric subspace.
        """
        effective_weight = self.effective_weight(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        if effective_weight == 0.0:
            return tuple(0.0 for _ in range(self.dimension))

        effective_displacement_sum = self.effective_displacement_sum(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        return tuple(value / effective_weight for value in effective_displacement_sum)

    def effective_covariance(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> tuple[tuple[float, ...], ...]:
        """Return the lazily decayed covariance matrix.

        Parameters
        ----------
        current_update_index : int
            Reducer update index at which to materialize the decay.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        tuple[tuple[float, ...], ...]
            Effective covariance matrix for this numeric subspace.
        """
        effective_weight = self.effective_weight(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        if effective_weight == 0.0:
            return tuple(
                tuple(0.0 for _ in range(self.dimension)) for _ in range(self.dimension)
            )

        effective_mean = self.effective_mean(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        effective_outer_product_sum = self.effective_outer_product_sum(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        return tuple(
            tuple(
                (
                    effective_outer_product_sum[row_index][column_index]
                    / effective_weight
                )
                - (effective_mean[row_index] * effective_mean[column_index])
                for column_index in range(self.dimension)
            )
            for row_index in range(self.dimension)
        )

    def record_successful_displacement(
        self,
        displacement: NumericSubspaceDisplacement,
        *,
        survival_efficiency: float,
        current_update_index: int,
        adaptation_decay: float,
    ) -> Self:
        """Return a covariance stat with one additional successful displacement.

        Parameters
        ----------
        displacement : NumericSubspaceDisplacement
            Successful numeric displacement to accumulate.
        survival_efficiency : float
            Proposal survival efficiency used as the sample weight.
        current_update_index : int
            Reducer update index associated with the displacement.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        Self
            Updated covariance statistic with the displacement incorporated.

        Raises
        ------
        ValueError
            If ``displacement`` is keyed to a different leaf family.
        """
        if displacement.leaf_paths != self.leaf_paths:
            msg = "numeric covariance displacement leaf paths must match the stat key"
            raise ValueError(msg)
        if type(survival_efficiency) is not float:
            msg = "survival_efficiency must be a float"
            raise TypeError(msg)
        if not 0.0 < survival_efficiency <= 1.0:
            msg = "survival_efficiency must lie within (0, 1]"
            raise ValueError(msg)

        decayed_weight = self.effective_weight(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        decayed_displacement_sum = self.effective_displacement_sum(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        decayed_outer_product_sum = self.effective_outer_product_sum(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        next_displacement_sum = tuple(
            decayed_value + (survival_efficiency * displacement_value)
            for decayed_value, displacement_value in zip(
                decayed_displacement_sum,
                displacement.displacement_coordinates,
                strict=True,
            )
        )
        next_outer_product_sum = tuple(
            tuple(
                decayed_outer_product_sum[row_index][column_index]
                + survival_efficiency
                * (
                    displacement.displacement_coordinates[row_index]
                    * displacement.displacement_coordinates[column_index]
                )
                for column_index in range(self.dimension)
            )
            for row_index in range(self.dimension)
        )
        if any(not isfinite(value) for value in next_displacement_sum) or any(
            not isfinite(value) for row in next_outer_product_sum for value in row
        ):
            msg = (
                "numeric covariance moment accumulation must remain finite; "
                "normalize displacement coordinates"
            )
            raise ValueError(msg)
        return replace(
            self,
            observation_count=self.observation_count + 1,
            discounted_weight=decayed_weight + survival_efficiency,
            discounted_displacement_sum=next_displacement_sum,
            discounted_outer_product_sum=next_outer_product_sum,
            last_update_index=current_update_index,
        )


@dataclass(frozen=True, slots=True)
class ProposalFamilyStat:
    """Discounted survival-efficiency history for one proposal family.

    Parameters
    ----------
    family_key : str
        Canonical proposal family identifier.
    observation_count : int, default=0
        Number of proposal outcomes recorded for this family.
    discounted_survival_efficiency : float, default=0.0
        Lazily decayed sum of bounded proposal survival efficiencies.
    discounted_observation_weight : float, default=0.0
        Lazily decayed number of represented outcome observations.
    last_update_index : int, default=0
        Reducer update index at which the statistic was last materialized.
    """

    family_key: str
    observation_count: int = 0
    discounted_survival_efficiency: float = 0.0
    discounted_observation_weight: float = 0.0
    last_update_index: int = 0

    def __post_init__(self) -> None:
        """Reject invalid family-stat state."""
        if self.family_key == "":
            msg = "family_key must not be empty"
            raise ValueError(msg)

        if type(self.observation_count) is not int:
            msg = "observation_count must be an int"
            raise TypeError(msg)
        if self.observation_count < 0:
            msg = "observation_count must be non-negative"
            raise ValueError(msg)
        if type(self.discounted_survival_efficiency) is not float:
            msg = "discounted_survival_efficiency must be a float"
            raise TypeError(msg)
        if not isfinite(self.discounted_survival_efficiency):
            msg = "discounted_survival_efficiency must be finite"
            raise ValueError(msg)
        if type(self.discounted_observation_weight) is not float:
            msg = "discounted_observation_weight must be a float"
            raise TypeError(msg)
        if not isfinite(self.discounted_observation_weight):
            msg = "discounted_observation_weight must be finite"
            raise ValueError(msg)
        if not (
            0.0
            <= self.discounted_survival_efficiency
            <= self.discounted_observation_weight
            <= self.observation_count
        ):
            msg = (
                "discounted family survival efficiency and weight must be "
                "bounded by observations"
            )
            raise ValueError(msg)
        if type(self.last_update_index) is not int:
            msg = "last_update_index must be an int"
            raise TypeError(msg)
        if self.last_update_index < 0:
            msg = "last_update_index must be non-negative"
            raise ValueError(msg)

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the family statistic.

        Returns
        -------
        JSONDict
            JSON-safe family-stat snapshot.
        """
        return {
            "family_key": self.family_key,
            "observation_count": self.observation_count,
            "discounted_survival_efficiency": self.discounted_survival_efficiency,
            "discounted_observation_weight": self.discounted_observation_weight,
            "last_update_index": self.last_update_index,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
    ) -> Self:
        """Build a family statistic from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe family-stat snapshot.

        Returns
        -------
        Self
            Reconstructed family statistic.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        family_key = require_json_str(
            require_json_field(data, "family_key"),
            field_name="family_key",
        )
        observation_count = require_json_int(
            require_json_field(data, "observation_count"),
            field_name="observation_count",
        )
        discounted_survival_efficiency = require_json_finite_float(
            require_json_field(data, "discounted_survival_efficiency"),
            field_name="discounted_survival_efficiency",
        )
        discounted_observation_weight = require_json_finite_float(
            require_json_field(data, "discounted_observation_weight"),
            field_name="discounted_observation_weight",
        )
        last_update_index = require_json_int(
            require_json_field(data, "last_update_index"),
            field_name="last_update_index",
        )
        return cls(
            family_key=family_key,
            observation_count=observation_count,
            discounted_survival_efficiency=discounted_survival_efficiency,
            discounted_observation_weight=discounted_observation_weight,
            last_update_index=last_update_index,
        )

    def effective_survival_efficiency_sum(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> float:
        """Return the lazily decayed family survival-efficiency sum.

        Parameters
        ----------
        current_update_index : int
            Reducer update index at which to materialize the decay.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        float
            Effective family survival-efficiency sum after lazy decay.

        Raises
        ------
        ValueError
            If ``current_update_index`` is earlier than ``last_update_index``.
        """
        if current_update_index < self.last_update_index:
            msg = "current_update_index must not go backwards"
            raise ValueError(msg)
        elapsed_updates = current_update_index - self.last_update_index
        return self.discounted_survival_efficiency * (adaptation_decay**elapsed_updates)

    def effective_observation_weight(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> float:
        """Return the lazily decayed family-observation weight."""
        if current_update_index < self.last_update_index:
            msg = "current_update_index must not go backwards"
            raise ValueError(msg)
        elapsed_updates = current_update_index - self.last_update_index
        return self.discounted_observation_weight * (adaptation_decay**elapsed_updates)

    def effective_survival_efficiency(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> float:
        """Return decayed survival efficiency per family observation."""
        effective_weight = self.effective_observation_weight(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        if effective_weight == 0.0:
            return 0.0
        return (
            self.effective_survival_efficiency_sum(
                current_update_index=current_update_index,
                adaptation_decay=adaptation_decay,
            )
            / effective_weight
        )

    def record_generation(
        self,
        summary: ProposalFamilyAdaptationSummary,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> Self:
        """Return this family stat updated by one generation summary.

        Parameters
        ----------
        summary : ProposalFamilyAdaptationSummary
            Canonical family observations and bounded total survival efficiency.
        current_update_index : int
            Reducer update index associated with the generation.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        Self
            Updated family statistic with the generation incorporated.
        """
        if summary.family_key != self.family_key:
            msg = "family adaptation summary key must match the stat key"
            raise ValueError(msg)
        return replace(
            self,
            observation_count=self.observation_count + summary.observation_count,
            discounted_survival_efficiency=(
                self.effective_survival_efficiency_sum(
                    current_update_index=current_update_index,
                    adaptation_decay=adaptation_decay,
                )
                + summary.total_survival_efficiency
            ),
            discounted_observation_weight=(
                self.effective_observation_weight(
                    current_update_index=current_update_index,
                    adaptation_decay=adaptation_decay,
                )
                + float(summary.observation_count)
            ),
            last_update_index=current_update_index,
        )


@dataclass(frozen=True, slots=True)
class ProposalLeafStat:
    """Discounted survival-efficiency history for one leaf association.

    Parameters
    ----------
    path : LeafPath
        Structured leaf path keyed by this statistic.
    observation_count : int, default=0
        Number of observed outcomes recorded for the leaf.
    discounted_survival_efficiency : float, default=0.0
        Lazily decayed sum of bounded survival-efficiency shares.
    discounted_observation_weight : float, default=0.0
        Lazily decayed number of represented leaf associations.
    last_update_index : int, default=0
        Reducer update index at which the statistic was last materialized.
    recent_failure_streak : int, default=0
        Number of consecutive observed generations without positive efficiency.
    """

    path: LeafPath
    observation_count: int = 0
    discounted_survival_efficiency: float = 0.0
    discounted_observation_weight: float = 0.0
    last_update_index: int = 0
    recent_failure_streak: int = 0

    def __post_init__(self) -> None:
        """Normalize one canonical leaf-stat record."""
        object.__setattr__(self, "path", tuple(self.path))
        if type(self.observation_count) is not int:
            msg = "observation_count must be an int"
            raise TypeError(msg)
        if self.observation_count < 0:
            msg = "observation_count must be non-negative"
            raise ValueError(msg)
        if type(self.discounted_survival_efficiency) is not float:
            msg = "discounted_survival_efficiency must be a float"
            raise TypeError(msg)
        if not isfinite(self.discounted_survival_efficiency):
            msg = "discounted_survival_efficiency must be finite"
            raise ValueError(msg)
        if type(self.discounted_observation_weight) is not float:
            msg = "discounted_observation_weight must be a float"
            raise TypeError(msg)
        if not isfinite(self.discounted_observation_weight):
            msg = "discounted_observation_weight must be finite"
            raise ValueError(msg)
        if not (
            0.0
            <= self.discounted_survival_efficiency
            <= self.discounted_observation_weight
            <= self.observation_count
        ):
            msg = (
                "discounted leaf survival efficiency and weight must be bounded "
                "by observations"
            )
            raise ValueError(msg)
        if type(self.last_update_index) is not int:
            msg = "last_update_index must be an int"
            raise TypeError(msg)
        if self.last_update_index < 0:
            msg = "last_update_index must be non-negative"
            raise ValueError(msg)
        if type(self.recent_failure_streak) is not int:
            msg = "recent_failure_streak must be an int"
            raise TypeError(msg)
        if self.recent_failure_streak < 0:
            msg = "recent_failure_streak must be non-negative"
            raise ValueError(msg)

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the leaf statistic.

        Returns
        -------
        JSONDict
            JSON-safe leaf-stat snapshot.
        """
        return {
            "path": _leaf_path_to_json(self.path),
            "observation_count": self.observation_count,
            "discounted_survival_efficiency": self.discounted_survival_efficiency,
            "discounted_observation_weight": self.discounted_observation_weight,
            "last_update_index": self.last_update_index,
            "recent_failure_streak": self.recent_failure_streak,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
    ) -> Self:
        """Build a leaf statistic from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe leaf-stat snapshot.

        Returns
        -------
        Self
            Reconstructed leaf statistic.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        observation_count = require_json_int(
            require_json_field(data, "observation_count"),
            field_name="observation_count",
        )
        discounted_survival_efficiency = require_json_finite_float(
            require_json_field(data, "discounted_survival_efficiency"),
            field_name="discounted_survival_efficiency",
        )
        discounted_observation_weight = require_json_finite_float(
            require_json_field(data, "discounted_observation_weight"),
            field_name="discounted_observation_weight",
        )
        last_update_index = require_json_int(
            require_json_field(data, "last_update_index"),
            field_name="last_update_index",
        )
        recent_failure_streak = require_json_int(
            require_json_field(data, "recent_failure_streak"),
            field_name="recent_failure_streak",
        )
        return cls(
            path=_leaf_path_from_json(
                require_json_field(data, "path"),
                field_name="path",
            ),
            observation_count=observation_count,
            discounted_survival_efficiency=discounted_survival_efficiency,
            discounted_observation_weight=discounted_observation_weight,
            last_update_index=last_update_index,
            recent_failure_streak=recent_failure_streak,
        )

    def effective_survival_efficiency_sum(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> float:
        """Return the lazily decayed leaf survival-efficiency sum.

        Parameters
        ----------
        current_update_index : int
            Reducer update index at which to materialize the decay.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        float
            Effective leaf survival-efficiency sum after lazy decay.

        Raises
        ------
        ValueError
            If ``current_update_index`` is earlier than ``last_update_index``.
        """
        if current_update_index < self.last_update_index:
            msg = "current_update_index must not go backwards"
            raise ValueError(msg)
        elapsed_updates = current_update_index - self.last_update_index
        return self.discounted_survival_efficiency * (adaptation_decay**elapsed_updates)

    def effective_observation_weight(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> float:
        """Return the lazily decayed leaf-observation weight."""
        if current_update_index < self.last_update_index:
            msg = "current_update_index must not go backwards"
            raise ValueError(msg)
        elapsed_updates = current_update_index - self.last_update_index
        return self.discounted_observation_weight * (adaptation_decay**elapsed_updates)

    def effective_survival_efficiency(
        self,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> float:
        """Return decayed survival efficiency per leaf association."""
        effective_weight = self.effective_observation_weight(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        )
        if effective_weight == 0.0:
            return 0.0
        return (
            self.effective_survival_efficiency_sum(
                current_update_index=current_update_index,
                adaptation_decay=adaptation_decay,
            )
            / effective_weight
        )

    def record_generation(
        self,
        summary: ProposalLeafAdaptationSummary,
        *,
        current_update_index: int,
        adaptation_decay: float,
    ) -> Self:
        """Return this leaf stat updated by one generation summary.

        Parameters
        ----------
        summary : ProposalLeafAdaptationSummary
            Canonical leaf associations and bounded total survival efficiency.
        current_update_index : int
            Reducer update index associated with the generation.
        adaptation_decay : float
            Multiplicative decay factor applied per update step.

        Returns
        -------
        Self
            Updated leaf statistic with refreshed efficiency and failure streak.
        """
        if summary.path != self.path:
            msg = "leaf adaptation summary path must match the stat key"
            raise ValueError(msg)
        next_discounted_survival_efficiency = (
            self.effective_survival_efficiency_sum(
                current_update_index=current_update_index,
                adaptation_decay=adaptation_decay,
            )
            + summary.total_survival_efficiency
        )
        next_discounted_observation_weight = self.effective_observation_weight(
            current_update_index=current_update_index,
            adaptation_decay=adaptation_decay,
        ) + float(summary.observation_count)
        next_failure_streak = self.recent_failure_streak + 1
        if summary.total_survival_efficiency > 0.0:
            next_failure_streak = 0

        return replace(
            self,
            observation_count=self.observation_count + summary.observation_count,
            discounted_survival_efficiency=next_discounted_survival_efficiency,
            discounted_observation_weight=next_discounted_observation_weight,
            last_update_index=current_update_index,
            recent_failure_streak=next_failure_streak,
        )
