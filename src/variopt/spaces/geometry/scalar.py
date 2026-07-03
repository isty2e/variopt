"""Built-in scalar structured-space geometry implementations."""

from dataclasses import dataclass, field
from math import log

from ..scalar import (
    CategoricalSpace,
    IntegerSpace,
    RealSpace,
)
from ..types import SpaceCandidateValue, SpaceScalarValue
from .leaf import (
    require_integer_candidate,
    require_real_candidate,
)
from .parts import StructuredDistanceParts

# Categorical geometry keys keep the exact runtime type next to the value so
# choices such as ``1`` and ``1.0`` remain distinct in membership caches.
HashableCategoricalValue = bool | int | float | str | bytes
HashableCategoricalType = type[bool] | type[int] | type[float] | type[str] | type[bytes]
CategoricalChoiceKey = tuple[HashableCategoricalType, HashableCategoricalValue]

_CATEGORICAL_MATCH_DISTANCE_PARTS = StructuredDistanceParts(
    overlap_squared_distance=0.0,
    shared_leaf_count=1,
)
_CATEGORICAL_MISMATCH_DISTANCE_PARTS = StructuredDistanceParts(
    overlap_squared_distance=1.0,
    shared_leaf_count=1,
)


@dataclass(frozen=True, slots=True)
class RealSpaceGeometry:
    """Fast geometry for one real-valued leaf space.

    Parameters
    ----------
    space : RealSpace
        Real-valued leaf space whose distance law is implemented.
    """

    space: RealSpace

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return normalized squared distance parts for one real leaf.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left leaf value.
        right : SpaceCandidateValue
            Right leaf value.

        Returns
        -------
        StructuredDistanceParts
            Normalized squared-distance contribution for the leaf.

        Raises
        ------
        TypeError
            If either value is not a canonical real candidate.
        ValueError
            If either value lies outside the declared bounds.
        """
        overlap_squared_distance, shared_leaf_count, topology_mismatch_leaf_count = (
            self.distance_part_values(left, right)
        )
        return StructuredDistanceParts(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for one real leaf."""
        return (self.squared_distance(left, right), 1, 0)

    def squared_distance(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> float:
        """Return one normalized squared distance for one real leaf."""
        left_value = require_real_candidate(
            value=left,
            message="real-space diversity requires numeric left leaf values",
        )
        right_value = require_real_candidate(
            value=right,
            message="real-space diversity requires numeric right leaf values",
        )
        if left_value < self.space.low or left_value > self.space.high:
            msg = "real candidate is outside the declared bounds"
            raise ValueError(msg)
        if right_value < self.space.low or right_value > self.space.high:
            msg = "real candidate is outside the declared bounds"
            raise ValueError(msg)

        if self.space.low == self.space.high:
            return 0.0

        if self.space.scale == "log":
            coordinate_span = log(self.space.high) - log(self.space.low)
            leaf_distance = abs(log(left_value) - log(right_value)) / coordinate_span
        else:
            coordinate_span = self.space.high - self.space.low
            leaf_distance = abs(left_value - right_value) / coordinate_span

        return leaf_distance * leaf_distance


@dataclass(frozen=True, slots=True)
class IntegerSpaceGeometry:
    """Fast geometry for one integer-valued leaf space.

    Parameters
    ----------
    space : IntegerSpace
        Integer-valued leaf space whose distance law is implemented.
    """

    space: IntegerSpace

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return normalized squared distance parts for one integer leaf.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left leaf value.
        right : SpaceCandidateValue
            Right leaf value.

        Returns
        -------
        StructuredDistanceParts
            Normalized squared-distance contribution for the leaf.

        Raises
        ------
        TypeError
            If either value is not a canonical integer candidate.
        ValueError
            If either value lies outside the declared bounds.
        """
        overlap_squared_distance, shared_leaf_count, topology_mismatch_leaf_count = (
            self.distance_part_values(left, right)
        )
        return StructuredDistanceParts(
            overlap_squared_distance=overlap_squared_distance,
            shared_leaf_count=shared_leaf_count,
            topology_mismatch_leaf_count=topology_mismatch_leaf_count,
        )

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for one integer leaf."""
        return (self.squared_distance(left, right), 1, 0)

    def squared_distance(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> float:
        """Return one normalized squared distance for one integer leaf."""
        left_value = require_integer_candidate(
            value=left,
            message="integer-space diversity requires canonical integer left leaf values",
        )
        right_value = require_integer_candidate(
            value=right,
            message="integer-space diversity requires canonical integer right leaf values",
        )
        if left_value < self.space.low or left_value > self.space.high:
            msg = "integer candidate is outside the declared bounds"
            raise ValueError(msg)
        if right_value < self.space.low or right_value > self.space.high:
            msg = "integer candidate is outside the declared bounds"
            raise ValueError(msg)

        if self.space.low == self.space.high:
            return 0.0

        if self.space.scale == "log":
            coordinate_span = log(float(self.space.high)) - log(float(self.space.low))
            leaf_distance = abs(log(float(left_value)) - log(float(right_value))) / coordinate_span
        else:
            coordinate_span = float(self.space.high - self.space.low)
            leaf_distance = abs(float(left_value - right_value)) / coordinate_span

        return leaf_distance * leaf_distance


@dataclass(frozen=True, slots=True)
class CategoricalSpaceGeometry:
    """Fast geometry for one categorical leaf space.

    Parameters
    ----------
    space : CategoricalSpace[SpaceScalarValue]
        Categorical leaf space whose distance law is implemented.
    """

    space: CategoricalSpace[SpaceScalarValue]
    choice_keys: frozenset[CategoricalChoiceKey] | None = field(
        init=False,
        repr=False,
    )
    equal_choice_values: frozenset[HashableCategoricalValue] | None = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Cache hashable categorical choices for repeated distance checks."""
        choice_keys: list[CategoricalChoiceKey] = []
        equal_choice_values: list[HashableCategoricalValue] = []
        for choice in self.space.choices:
            key = categorical_choice_key(choice)
            if key is None:
                object.__setattr__(self, "choice_keys", None)
                object.__setattr__(self, "equal_choice_values", None)
                return
            choice_keys.append(key)
            equal_choice_values.append(key[1])

        object.__setattr__(self, "choice_keys", frozenset(choice_keys))
        object.__setattr__(self, "equal_choice_values", frozenset(equal_choice_values))

    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        """Return one match-or-mismatch distance part for one categorical leaf.

        Parameters
        ----------
        left : SpaceCandidateValue
            Left categorical choice.
        right : SpaceCandidateValue
            Right categorical choice.

        Returns
        -------
        StructuredDistanceParts
            Zero distance for a match, unit distance for a mismatch.

        Raises
        ------
        ValueError
            If either value is not a declared categorical choice.
        """
        if self.squared_distance(left, right) == 0.0:
            return _CATEGORICAL_MATCH_DISTANCE_PARTS
        return _CATEGORICAL_MISMATCH_DISTANCE_PARTS

    def squared_distance(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> float:
        """Return the match-or-mismatch squared distance for one categorical leaf."""
        choice_keys = self.choice_keys
        equal_choice_values = self.equal_choice_values
        if choice_keys is not None and equal_choice_values is not None:
            # This mirrors require_geometry_categorical_choice for the all-hashable
            # cache path to avoid two function calls per categorical leaf.
            left_key = categorical_choice_key(left)
            right_key = categorical_choice_key(right)
            if left_key is not None and right_key is not None:
                if left_key not in choice_keys:
                    if left_key[1] in equal_choice_values:
                        msg = "categorical candidate must use the declared choice type"
                        raise TypeError(msg)
                    msg = "categorical candidate is not in the declared choices"
                    raise ValueError(msg)
                if right_key not in choice_keys:
                    if right_key[1] in equal_choice_values:
                        msg = "categorical candidate must use the declared choice type"
                        raise TypeError(msg)
                    msg = "categorical candidate is not in the declared choices"
                    raise ValueError(msg)
                if left_key == right_key:
                    return 0.0
                return 1.0

        require_geometry_categorical_choice(
            space=self.space,
            choice_keys=choice_keys,
            equal_choice_values=equal_choice_values,
            value=left,
        )
        require_geometry_categorical_choice(
            space=self.space,
            choice_keys=choice_keys,
            equal_choice_values=equal_choice_values,
            value=right,
        )
        if left == right:
            return 0.0
        return 1.0

    def distance_part_values(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> tuple[float, int, int]:
        """Return raw distance-part values for one categorical leaf."""
        return (self.squared_distance(left, right), 1, 0)


def categorical_choice_key(value: SpaceCandidateValue) -> CategoricalChoiceKey | None:
    """Return a hashable exact-type choice key, if ``value`` supports hashing."""
    if type(value) is bool:
        return (bool, value)
    if type(value) is int:
        return (int, value)
    if type(value) is float:
        return (float, value)
    if type(value) is str:
        return (str, value)
    if type(value) is bytes:
        return (bytes, value)
    return None


def require_geometry_categorical_choice(
    *,
    space: CategoricalSpace[SpaceScalarValue],
    choice_keys: frozenset[CategoricalChoiceKey] | None,
    equal_choice_values: frozenset[HashableCategoricalValue] | None,
    value: SpaceCandidateValue,
) -> None:
    """Validate one categorical geometry leaf without re-entering space methods."""
    if type(value) is bool:
        key: CategoricalChoiceKey | None = (bool, value)
    elif type(value) is int:
        key = (int, value)
    elif type(value) is float:
        key = (float, value)
    elif type(value) is str:
        key = (str, value)
    elif type(value) is bytes:
        key = (bytes, value)
    elif type(value) is bytearray:
        key = None
    else:
        msg = "categorical candidate must be scalar"
        raise TypeError(msg)

    if key is not None and choice_keys is not None and equal_choice_values is not None:
        if key in choice_keys:
            return
        if key[1] in equal_choice_values:
            msg = "categorical candidate must use the declared choice type"
            raise TypeError(msg)
        msg = "categorical candidate is not in the declared choices"
        raise ValueError(msg)

    value_type = type(value)
    for choice in space.choices:
        if value == choice:
            if value_type is not type(choice):
                msg = "categorical candidate must use the declared choice type"
                raise TypeError(msg)
            return

    msg = "categorical candidate is not in the declared choices"
    raise ValueError(msg)
