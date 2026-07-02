"""Serialization regressions for CSA variation operators."""

import pickle
from typing import TypeVar, cast

import numpy as np

from variopt import PermutationSpace
from variopt.algorithms.population.csa.operators import MixtureVariation
from variopt.algorithms.population.permutation import SwapMutation

PickleRoundTripT = TypeVar("PickleRoundTripT")


def pickle_round_trip(value: PickleRoundTripT) -> PickleRoundTripT:
    """Return one pickle round-trip result with the input type restored."""
    return cast(PickleRoundTripT, pickle.loads(pickle.dumps(value)))


class CSAOperatorSerializationTests:
    """Regression tests for process-safe CSA operator records."""

    def test_mixture_variation_pickle_round_trips(self) -> None:
        space = PermutationSpace(size=4)
        variation = MixtureVariation((SwapMutation(space),))

        restored = pickle_round_trip(variation)
        child = restored.apply(
            ((0, 1, 2, 3),),
            np.random.RandomState(0),
        )

        space.validate(child)
        assert restored.weights == (1.0,)
