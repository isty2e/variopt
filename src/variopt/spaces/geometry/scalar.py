"""Built-in scalar structured-space geometry implementations."""

from dataclasses import dataclass
from math import log

from ..scalar import CategoricalSpace, IntegerSpace, RealSpace
from ..types import SpaceCandidateValue, SpaceScalarValue
from .leaf import (
    require_integer_candidate,
    require_real_candidate,
    validate_categorical_choice,
)
from .parts import StructuredDistanceParts


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
            return StructuredDistanceParts(
                overlap_squared_distance=0.0,
                shared_leaf_count=1,
            )

        if self.space.scale == "log":
            coordinate_span = log(self.space.high) - log(self.space.low)
            leaf_distance = abs(log(left_value) - log(right_value)) / coordinate_span
        else:
            coordinate_span = self.space.high - self.space.low
            leaf_distance = abs(left_value - right_value) / coordinate_span

        return StructuredDistanceParts(
            overlap_squared_distance=leaf_distance * leaf_distance,
            shared_leaf_count=1,
        )


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
            return StructuredDistanceParts(
                overlap_squared_distance=0.0,
                shared_leaf_count=1,
            )

        if self.space.scale == "log":
            coordinate_span = log(float(self.space.high)) - log(float(self.space.low))
            leaf_distance = abs(log(float(left_value)) - log(float(right_value))) / coordinate_span
        else:
            coordinate_span = float(self.space.high - self.space.low)
            leaf_distance = abs(float(left_value - right_value)) / coordinate_span

        return StructuredDistanceParts(
            overlap_squared_distance=leaf_distance * leaf_distance,
            shared_leaf_count=1,
        )


@dataclass(frozen=True, slots=True)
class CategoricalSpaceGeometry:
    """Fast geometry for one categorical leaf space.

    Parameters
    ----------
    space : CategoricalSpace[SpaceScalarValue]
        Categorical leaf space whose distance law is implemented.
    """

    space: CategoricalSpace[SpaceScalarValue]

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
        validate_categorical_choice(self.space, left)
        validate_categorical_choice(self.space, right)
        squared_distance = 0.0 if left == right else 1.0
        return StructuredDistanceParts(
            overlap_squared_distance=squared_distance,
            shared_leaf_count=1,
        )
