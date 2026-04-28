"""Scalar search spaces."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, log
from typing import Generic, Literal, TypeVar

import numpy as np
from typing_extensions import override

from ..randomness import random_state_randint
from .structured import LeafPath, StructuredLeafSpace, StructuredSearchSpace
from .types import SpaceCandidateValue, SpaceScalarValue

CategoricalT = TypeVar("CategoricalT", bound=SpaceScalarValue)


def has_duplicate_choices(choices: Sequence[CategoricalT]) -> bool:
    """Return whether a categorical declaration contains duplicate values.

    Parameters
    ----------
    choices : Sequence[CategoricalT]
        Declared categorical choices to inspect.

    Returns
    -------
    bool
        Whether any value appears more than once in ``choices``.
    """
    for index, choice in enumerate(choices):
        for other_choice in choices[index + 1 :]:
            if choice == other_choice:
                return True

    return False


@dataclass(frozen=True)
class RealSpace(StructuredSearchSpace[float | int, float]):
    """Bounded real-valued search space.

    Parameters
    ----------
    low : float
        Inclusive lower bound.
    high : float
        Inclusive upper bound.
    scale : {"linear", "log"}, default="linear"
        Coordinate system used for continuous transforms and local search.
        ``"log"`` means values are optimized in log coordinates while remaining
        positive in value space.
    """

    low: float
    high: float
    scale: Literal["linear", "log"] = "linear"

    def __post_init__(self) -> None:
        """Validate real-space metadata and normalize bounds.

        Raises
        ------
        ValueError
            If bounds are non-finite, out of order, incompatible with the
            declared scale, or if ``scale`` is not supported.
        """
        low = float(self.low)
        high = float(self.high)

        if not np.isfinite(low) or not np.isfinite(high):
            msg = "RealSpace bounds must be finite"
            raise ValueError(msg)

        if low > high:
            msg = "RealSpace low must be less than or equal to high"
            raise ValueError(msg)

        if self.scale not in {"linear", "log"}:
            msg = "RealSpace scale must be 'linear' or 'log'"
            raise ValueError(msg)

        if self.scale == "log" and (low <= 0.0 or high <= 0.0):
            msg = "log-scaled RealSpace bounds must both be positive"
            raise ValueError(msg)

        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @override
    def normalize(self, raw_candidate: float | int) -> float:
        """Normalize a scalar boundary value into canonical float form.

        Parameters
        ----------
        raw_candidate : float | int
            Boundary-level numeric value.

        Returns
        -------
        float
            Canonical floating-point candidate within the declared bounds.
        """
        if type(raw_candidate) not in {float, int}:
            msg = "real candidate must be an int or float"
            raise TypeError(msg)

        candidate = float(raw_candidate)
        self.validate(candidate)
        return candidate

    @override
    def validate(self, candidate: float) -> None:
        """Validate a canonical real-valued candidate.

        Parameters
        ----------
        candidate : float
            Canonical floating-point candidate to validate.
        """
        if type(candidate) is not float:
            msg = "real candidate must be canonical float"
            raise TypeError(msg)

        if not np.isfinite(candidate):
            msg = "real candidate must be finite"
            raise ValueError(msg)

        if candidate < self.low or candidate > self.high:
            msg = "real candidate is outside the declared bounds"
            raise ValueError(msg)

    @override
    def sample(self, random_state: np.random.RandomState) -> float:
        """Sample a canonical real-valued candidate.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state object that owns all stochasticity for the sample.

        Returns
        -------
        float
            Canonical sampled candidate.
        """
        if self.low == self.high:
            return self.low

        if self.scale == "log":
            coordinate_low, coordinate_high = self.coordinate_bounds()
            return self.project_coordinate(
                random_state.uniform(coordinate_low, coordinate_high),
            )

        return float(random_state.uniform(self.low, self.high))

    def coordinate_bounds(self) -> tuple[float, float]:
        """Return the coordinate-space bounds for this space.

        Returns
        -------
        tuple[float, float]
            Linear bounds for ``scale="linear"`` and log-domain bounds for
            ``scale="log"``.
        """
        if self.scale == "log":
            return (log(self.low), log(self.high))
        return (self.low, self.high)

    def to_coordinate(self, value: float) -> float:
        """Map a canonical value into coordinate space.

        Parameters
        ----------
        value : float
            Canonical candidate value.

        Returns
        -------
        float
            Coordinate-space representation of ``value``.
        """
        self.validate(value)
        if self.scale == "log":
            return log(value)
        return value

    def project_coordinate(self, coordinate: float) -> float:
        """Project a coordinate-space value back into value space.

        Parameters
        ----------
        coordinate : float
            Coordinate-space value to project.

        Returns
        -------
        float
            Canonical value clamped into the declared bounds.
        """
        if not np.isfinite(coordinate):
            msg = "coordinate must be finite"
            raise ValueError(msg)

        coordinate_low, coordinate_high = self.coordinate_bounds()
        clamped_coordinate = min(coordinate_high, max(coordinate_low, coordinate))
        if self.scale == "log":
            value = exp(clamped_coordinate)
        else:
            value = clamped_coordinate
        return float(min(self.high, max(self.low, value)))

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return the single editable leaf path.

        Returns
        -------
        tuple[LeafPath, ...]
            Singleton tuple containing the root leaf path ``()``.
        """
        return ((),)

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the leaf space at the supplied path.

        Parameters
        ----------
        path : LeafPath
            Leaf path to inspect.

        Returns
        -------
        StructuredLeafSpace
            This space itself when ``path == ()``.
        """
        if path != ():
            msg = f"path {path!r} is invalid for a real-valued leaf space"
            raise TypeError(msg)
        return self

    @override
    def leaf_value_at_path(
        self,
        candidate: float,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the leaf value stored at the supplied path.

        Parameters
        ----------
        candidate : float
            Canonical candidate to inspect.
        path : LeafPath
            Leaf path to inspect.

        Returns
        -------
        SpaceCandidateValue
            Canonical scalar value stored at the root leaf.
        """
        if path != ():
            msg = f"path {path!r} is invalid for a real-valued leaf candidate"
            raise TypeError(msg)
        self.validate(candidate)
        return candidate

    @override
    def replace_leaf_values(
        self,
        candidate: float,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> float:
        """Return a candidate with the root leaf replaced.

        Parameters
        ----------
        candidate : float
            Canonical candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by leaf path.

        Returns
        -------
        float
            Updated canonical candidate.
        """
        self.validate(candidate)
        if () not in replacements:
            return candidate

        replacement = replacements[()]
        if isinstance(replacement, bool) or not isinstance(replacement, (float, int)):
            msg = "real leaf replacement must be numeric"
            raise TypeError(msg)
        return self.normalize(replacement)


@dataclass(frozen=True)
class IntegerSpace(StructuredSearchSpace[int, int]):
    """Bounded integer-valued search space.

    Parameters
    ----------
    low : int
        Inclusive lower bound.
    high : int
        Inclusive upper bound.
    scale : {"linear", "log"}, default="linear"
        Coordinate system used for scalar transforms and sampled moves.
    """

    low: int
    high: int
    scale: Literal["linear", "log"] = "linear"

    def __post_init__(self) -> None:
        """Validate integer-space metadata.

        Raises
        ------
        TypeError
            If bounds are not canonical integers.
        ValueError
            If bounds are out of order or incompatible with the declared scale.
        """
        if type(self.low) is not int or type(self.high) is not int:
            msg = "IntegerSpace bounds must be canonical integers"
            raise TypeError(msg)

        if self.scale not in {"linear", "log"}:
            msg = "IntegerSpace scale must be 'linear' or 'log'"
            raise ValueError(msg)

        if self.scale == "log" and (self.low <= 0 or self.high <= 0):
            msg = "log-scaled IntegerSpace bounds must both be positive"
            raise ValueError(msg)

        if self.low > self.high:
            msg = "IntegerSpace low must be less than or equal to high"
            raise ValueError(msg)

    @override
    def normalize(self, raw_candidate: int) -> int:
        """Normalize a scalar boundary value into canonical integer form.

        Parameters
        ----------
        raw_candidate : int
            Boundary-level integer value.

        Returns
        -------
        int
            Canonical integer candidate within the declared bounds.
        """
        if type(raw_candidate) is not int:
            msg = "integer candidate must be an int"
            raise TypeError(msg)

        self.validate(raw_candidate)
        return raw_candidate

    @override
    def validate(self, candidate: int) -> None:
        """Validate a canonical integer candidate.

        Parameters
        ----------
        candidate : int
            Canonical integer candidate to validate.
        """
        if type(candidate) is not int:
            msg = "integer candidate must be canonical int"
            raise TypeError(msg)

        if candidate < self.low or candidate > self.high:
            msg = "integer candidate is outside the declared bounds"
            raise ValueError(msg)

    @override
    def sample(self, random_state: np.random.RandomState) -> int:
        """Sample a canonical integer candidate.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state object that owns all stochasticity for the sample.

        Returns
        -------
        int
            Canonical sampled integer candidate.
        """
        if self.low == self.high:
            return self.low

        if self.scale == "log":
            coordinate_low, coordinate_high = self.coordinate_bounds()
            coordinate = random_state.uniform(coordinate_low, coordinate_high)
            return self.project_coordinate(coordinate)

        return random_state_randint(random_state, self.low, self.high + 1)

    def coordinate_bounds(self) -> tuple[float, float]:
        """Return the coordinate-space bounds for this space.

        Returns
        -------
        tuple[float, float]
            Linear bounds for ``scale="linear"`` and log-domain bounds for
            ``scale="log"``.
        """
        if self.scale == "log":
            return (log(float(self.low)), log(float(self.high)))
        return (float(self.low), float(self.high))

    def to_coordinate(self, value: int) -> float:
        """Map a canonical integer into coordinate space.

        Parameters
        ----------
        value : int
            Canonical integer value.

        Returns
        -------
        float
            Coordinate-space representation of ``value``.
        """
        self.validate(value)
        if self.scale == "log":
            return log(float(value))
        return float(value)

    def project_coordinate(self, coordinate: float) -> int:
        """Project a coordinate-space value back into value space.

        Parameters
        ----------
        coordinate : float
            Coordinate-space value to project.

        Returns
        -------
        int
            Canonical integer clamped into the declared bounds.
        """
        if not np.isfinite(coordinate):
            msg = "coordinate must be finite"
            raise ValueError(msg)

        coordinate_low, coordinate_high = self.coordinate_bounds()
        clamped_coordinate = min(coordinate_high, max(coordinate_low, coordinate))
        if self.scale == "log":
            value = exp(clamped_coordinate)
        else:
            value = clamped_coordinate
        rounded_value = int(round(value))
        return min(self.high, max(self.low, rounded_value))

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return the single editable leaf path.

        Returns
        -------
        tuple[LeafPath, ...]
            Singleton tuple containing the root leaf path ``()``.
        """
        return ((),)

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the leaf space at the supplied path.

        Parameters
        ----------
        path : LeafPath
            Leaf path to inspect.

        Returns
        -------
        StructuredLeafSpace
            This space itself when ``path == ()``.
        """
        if path != ():
            msg = f"path {path!r} is invalid for an integer leaf space"
            raise TypeError(msg)
        return self

    @override
    def leaf_value_at_path(
        self,
        candidate: int,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the leaf value stored at the supplied path.

        Parameters
        ----------
        candidate : int
            Canonical candidate to inspect.
        path : LeafPath
            Leaf path to inspect.

        Returns
        -------
        SpaceCandidateValue
            Canonical scalar value stored at the root leaf.
        """
        if path != ():
            msg = f"path {path!r} is invalid for an integer leaf candidate"
            raise TypeError(msg)
        self.validate(candidate)
        return candidate

    @override
    def replace_leaf_values(
        self,
        candidate: int,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> int:
        """Return a candidate with the root leaf replaced.

        Parameters
        ----------
        candidate : int
            Canonical candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by leaf path.

        Returns
        -------
        int
            Updated canonical candidate.
        """
        self.validate(candidate)
        if () not in replacements:
            return candidate

        replacement = replacements[()]
        if type(replacement) is not int:
            msg = "integer leaf replacement must be a canonical integer"
            raise TypeError(msg)
        return self.normalize(replacement)


@dataclass(frozen=True)
class CategoricalSpace(
    StructuredSearchSpace[CategoricalT, CategoricalT],
    Generic[CategoricalT],
):
    """Finite categorical search space.

    Parameters
    ----------
    choices : Sequence[CategoricalT]
        Declared categorical values in stable order.
    """

    choices: tuple[CategoricalT, ...]

    def __init__(self, choices: Sequence[CategoricalT]) -> None:
        """Create a categorical space from declared choices.

        Parameters
        ----------
        choices : Sequence[CategoricalT]
            Declared categorical values in stable order.

        Raises
        ------
        ValueError
            If ``choices`` is empty or contains duplicates.
        """
        if len(choices) == 0:
            msg = "CategoricalSpace requires at least one choice"
            raise ValueError(msg)

        if has_duplicate_choices(choices):
            msg = "CategoricalSpace choices must be unique"
            raise ValueError(msg)

        object.__setattr__(self, "choices", tuple(choices))

    @override
    def normalize(self, raw_candidate: CategoricalT) -> CategoricalT:
        """Normalize a categorical boundary value.

        Parameters
        ----------
        raw_candidate : CategoricalT
            Boundary-level categorical value.

        Returns
        -------
        CategoricalT
            Canonical categorical value.
        """
        self.validate(raw_candidate)
        return raw_candidate

    @override
    def validate(self, candidate: CategoricalT) -> None:
        """Validate a canonical categorical candidate.

        Parameters
        ----------
        candidate : CategoricalT
            Canonical categorical value to validate.
        """
        if not any(candidate == choice for choice in self.choices):
            msg = "categorical candidate is not in the declared choices"
            raise ValueError(msg)

    @override
    def sample(self, random_state: np.random.RandomState) -> CategoricalT:
        """Sample a canonical categorical value.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state object that owns all stochasticity for the sample.

        Returns
        -------
        CategoricalT
            Canonical sampled categorical value.
        """
        index_space = IntegerSpace(0, len(self.choices) - 1)
        index = index_space.sample(random_state)
        return self.choices[index]

    def alternatives(
        self,
        candidate: SpaceScalarValue,
    ) -> tuple[CategoricalT, ...]:
        """Return all declared choices except the supplied value.

        Parameters
        ----------
        candidate : SpaceScalarValue
            Canonical categorical value already present in the space.

        Returns
        -------
        tuple[CategoricalT, ...]
            Alternative categorical values in declaration order.
        """
        if not any(candidate == choice for choice in self.choices):
            msg = "categorical candidate is not in the declared choices"
            raise ValueError(msg)

        return tuple(choice for choice in self.choices if choice != candidate)

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        """Return the single editable leaf path.

        Returns
        -------
        tuple[LeafPath, ...]
            Singleton tuple containing the root leaf path ``()``.
        """
        return ((),)

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        """Return the leaf space at the supplied path.

        Parameters
        ----------
        path : LeafPath
            Leaf path to inspect.

        Returns
        -------
        StructuredLeafSpace
            This space itself when ``path == ()``.
        """
        if path != ():
            msg = f"path {path!r} is invalid for a categorical leaf space"
            raise TypeError(msg)
        return self

    @override
    def leaf_value_at_path(
        self,
        candidate: CategoricalT,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        """Return the leaf value stored at the supplied path.

        Parameters
        ----------
        candidate : CategoricalT
            Canonical candidate to inspect.
        path : LeafPath
            Leaf path to inspect.

        Returns
        -------
        SpaceCandidateValue
            Canonical scalar value stored at the root leaf.
        """
        if path != ():
            msg = f"path {path!r} is invalid for a categorical leaf candidate"
            raise TypeError(msg)
        self.validate(candidate)
        return candidate

    @override
    def replace_leaf_values(
        self,
        candidate: CategoricalT,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> CategoricalT:
        """Return a candidate with the root leaf replaced.

        Parameters
        ----------
        candidate : CategoricalT
            Canonical candidate to update.
        replacements : Mapping[LeafPath, SpaceCandidateValue]
            Replacement mapping keyed by leaf path.

        Returns
        -------
        CategoricalT
            Updated canonical categorical value.
        """
        self.validate(candidate)
        if () not in replacements:
            return candidate

        replacement = replacements[()]
        for choice in self.choices:
            if replacement == choice:
                return choice

        msg = "categorical leaf replacement is not in the declared choices"
        raise ValueError(msg)
