"""Private adaptive-potential runtime helpers for CSA score shaping."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Generic

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....distance import require_valid_distance
from .....diversity import DiversityMetric
from .....json_types import JSONDict, JSONValue
from .....typevars import CandidateT
from .model import CSAAdaptivePotential

AdaptiveBinIndex = tuple[int, ...] | None


def _potential_to_json_value(
    potential: NDArray[np.float64],
    *,
    shape: tuple[int, ...],
) -> JSONValue:
    value_count = prod(shape)
    values = tuple(float(potential.item(index)) for index in range(value_count))
    nested_value, next_offset = _nested_potential_json_value(
        values=values,
        shape=shape,
        axis=0,
        offset=0,
    )
    if next_offset != len(values):
        msg = "potential serialization did not consume all values"
        raise RuntimeError(msg)
    return nested_value


def _nested_potential_json_value(
    *,
    values: tuple[float, ...],
    shape: tuple[int, ...],
    axis: int,
    offset: int,
) -> tuple[JSONValue, int]:
    axis_length = shape[axis]
    if axis == len(shape) - 1:
        next_offset = offset + axis_length
        return list(values[offset:next_offset]), next_offset

    items: list[JSONValue] = []
    next_offset = offset
    for _ in range(axis_length):
        item, next_offset = _nested_potential_json_value(
            values=values,
            shape=shape,
            axis=axis + 1,
            offset=next_offset,
        )
        items.append(item)
    return items, next_offset


@dataclass(frozen=True, slots=True)
class AdaptivePotentialState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical adaptive-potential state.

    Parameters
    ----------
    model : CSAAdaptivePotential[CandidateT]
        Adaptive-potential configuration defining bins and energies.
    potential : NDArray[np.float64]
        Dense potential grid aligned with ``model.axes``.
    """

    model: CSAAdaptivePotential[CandidateT]
    potential: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Reject potential arrays whose shape does not match the configured axes."""
        expected_shape = tuple(axis.bin_count for axis in self.model.axes)
        if self.potential.shape != expected_shape:
            msg = "potential shape must match the configured adaptive-potential axes"
            raise ValueError(msg)

    def score_candidate(
        self,
        *,
        candidate: CandidateT,
        diversity_metric: DiversityMetric[CandidateT],
    ) -> tuple[float, AdaptiveBinIndex]:
        """Return the adaptive-potential energy and bin index for one candidate.

        Parameters
        ----------
        candidate : CandidateT
            Candidate to score.
        diversity_metric : DiversityMetric[CandidateT]
            Diversity metric used to measure candidate-to-axis distances.

        Returns
        -------
        tuple[float, AdaptiveBinIndex]
            Energy contribution and resolved bin index.
        """
        bin_index = self.bin_index(
            candidate=candidate,
            diversity_metric=diversity_metric,
        )
        if bin_index is None:
            return self.model.overflow_energy, None

        return float(self.potential.item(bin_index)), bin_index

    def bin_index(
        self,
        *,
        candidate: CandidateT,
        diversity_metric: DiversityMetric[CandidateT],
    ) -> AdaptiveBinIndex:
        """Return the adaptive-potential bin index for one candidate, if in range.

        Parameters
        ----------
        candidate : CandidateT
            Candidate to locate.
        diversity_metric : DiversityMetric[CandidateT]
            Diversity metric used to measure candidate-to-axis distances.

        Returns
        -------
        AdaptiveBinIndex
            Adaptive bin index, or ``None`` when the candidate falls outside
            the configured axis ranges.
        """
        index_parts: list[int] = []
        for axis in self.model.axes:
            distance = require_valid_distance(
                diversity_metric.distance(candidate, axis.reference_candidate),
            )
            scaled_position = (
                float(axis.bin_count)
                / (axis.maximum_distance - axis.minimum_distance)
            ) * (distance - axis.minimum_distance)
            scaled_index = int(float(scaled_position))
            if scaled_index < 0 or scaled_index >= axis.bin_count:
                return None

            index_parts.append(scaled_index)

        return tuple(index_parts)

    def increment(self, bin_index: AdaptiveBinIndex) -> Self:
        """Return a copy with one adaptive-potential bin incremented.

        Parameters
        ----------
        bin_index : AdaptiveBinIndex
            Adaptive bin to increment. ``None`` leaves the state unchanged.

        Returns
        -------
        Self
            Updated adaptive-potential state.
        """
        if bin_index is None:
            return self

        updated_potential: NDArray[np.float64] = np.array(
            self.potential,
            copy=True,
            dtype=np.float64,
        )
        updated_potential[bin_index] += self.model.increment
        return type(self)(
            model=self.model,
            potential=updated_potential,
        )

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the adaptive-potential state.

        Returns
        -------
        JSONDict
            JSON-safe adaptive-potential snapshot.
        """
        return {
            "potential": _potential_to_json_value(
                self.potential,
                shape=tuple(axis.bin_count for axis in self.model.axes),
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        model: CSAAdaptivePotential[CandidateT],
    ) -> Self:
        """Build an adaptive-potential state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe adaptive-potential snapshot.
        model : CSAAdaptivePotential[CandidateT]
            Adaptive-potential model that owns the reconstructed state.

        Returns
        -------
        Self
            Reconstructed adaptive-potential state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        raw_potential = data.get("potential")
        if not isinstance(raw_potential, list):
            msg = "adaptive-potential snapshot requires potential list"
            raise TypeError(msg)
        return cls(
            model=model,
            potential=np.asarray(raw_potential, dtype=np.float64),
        )


def build_adaptive_potential_state(
    model: CSAAdaptivePotential[CandidateT] | None,
) -> AdaptivePotentialState[CandidateT] | None:
    """Return the canonical adaptive-potential state implied by one model.

    Parameters
    ----------
    model : CSAAdaptivePotential[CandidateT] | None
        Adaptive-potential model to instantiate.

    Returns
    -------
    AdaptivePotentialState[CandidateT] | None
        Fresh adaptive-potential state, or ``None`` when adaptive potential is
        disabled.
    """
    if model is None:
        return None

    shape = tuple(axis.bin_count for axis in model.axes)
    return AdaptivePotentialState(
        model=model,
        potential=np.zeros(shape, dtype=np.float64),
    )
