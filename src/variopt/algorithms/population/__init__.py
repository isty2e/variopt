"""Population-family optimizer facades.

This namespace is the canonical package tier for population-family optimizer
entry points. Family-specific policy, operator, and runtime details remain in
their subpackages such as :mod:`variopt.algorithms.population.csa`.
"""

from .clearing_ga import ClearingGAProfile, ClearingGeneticAlgorithmOptimizer
from .csa import CSAOptimizer, CSAProfile
from .de import DEProfile, DifferentialEvolutionOptimizer
from .ga import GAProfile, GeneticAlgorithmOptimizer
from .permutation import InversionMutation, OrderCrossover, SwapMutation
from .restricted_tournament_ga import (
    RestrictedTournamentGAProfile,
    RestrictedTournamentGeneticAlgorithmOptimizer,
)
from .species_ga import (
    SpeciesConservingGeneticAlgorithmOptimizer,
    SpeciesGAProfile,
)

__all__ = [
    "ClearingGAProfile",
    "ClearingGeneticAlgorithmOptimizer",
    "CSAOptimizer",
    "CSAProfile",
    "DEProfile",
    "DifferentialEvolutionOptimizer",
    "GAProfile",
    "GeneticAlgorithmOptimizer",
    "InversionMutation",
    "OrderCrossover",
    "RestrictedTournamentGAProfile",
    "RestrictedTournamentGeneticAlgorithmOptimizer",
    "SpeciesConservingGeneticAlgorithmOptimizer",
    "SpeciesGAProfile",
    "SwapMutation",
]
