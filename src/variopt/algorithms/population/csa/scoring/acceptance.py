"""Public CSA acceptance-policy configuration objects."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class CSAAcceptancePolicy:
    """CSA-specific probabilistic acceptance policy.

    Parameters
    ----------
    initial_temperature : float, default=0.0
        Initial acceptance temperature.
    reduction_factor : float, default=0.999
        Multiplicative decay or recovery factor.
    minimum_temperature : float, default=0.0
        Lower temperature bound.
    boltzmann_constant : float, default=1.9872066e-3
        Boltzmann constant used to convert temperature into inverse
        acceptance strength.
    recover : bool, default=False
        Whether the schedule recovers instead of cooling on each advance.
    """

    initial_temperature: float = 0.0
    reduction_factor: float = 0.999
    minimum_temperature: float = 0.0
    boltzmann_constant: float = 1.9872066e-3
    recover: bool = False

    def __post_init__(self) -> None:
        """Reject invalid acceptance-policy configuration."""
        if (
            not isfinite(self.initial_temperature)
            or self.initial_temperature < 0.0
        ):
            msg = "initial_temperature must be a finite non-negative float"
            raise ValueError(msg)

        if (
            not isfinite(self.reduction_factor)
            or self.reduction_factor <= 0.0
        ):
            msg = "reduction_factor must be a finite positive float"
            raise ValueError(msg)

        if (
            not isfinite(self.minimum_temperature)
            or self.minimum_temperature < 0.0
        ):
            msg = "minimum_temperature must be a finite non-negative float"
            raise ValueError(msg)

        if (
            not isfinite(self.boltzmann_constant)
            or self.boltzmann_constant <= 0.0
        ):
            msg = "boltzmann_constant must be a finite positive float"
            raise ValueError(msg)
