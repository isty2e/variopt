"""Regression tests for the population-family facade exports."""


import variopt.algorithms as root_algorithms
import variopt.algorithms.population as population_algorithms
import variopt.algorithms.population.clearing_ga as clearing_ga_algorithms
import variopt.algorithms.population.csa as csa_algorithms
import variopt.algorithms.population.de as de_algorithms
import variopt.algorithms.population.ga as ga_algorithms
import variopt.algorithms.population.permutation as permutation_algorithms
import variopt.algorithms.population.restricted_tournament_ga as restricted_tournament_ga_algorithms
import variopt.algorithms.population.species_ga as species_ga_algorithms


class PopulationFacadeExportTests:
    """Lock the canonical population-family facade surface."""

    def test_population_facade_reexports_population_family_entry_points(self) -> None:
        assert population_algorithms.CSAOptimizer is csa_algorithms.CSAOptimizer
        assert population_algorithms.CSAProfile is csa_algorithms.CSAProfile
        assert population_algorithms.DEProfile is de_algorithms.DEProfile
        assert population_algorithms.DifferentialEvolutionOptimizer is de_algorithms.DifferentialEvolutionOptimizer
        assert population_algorithms.GAProfile is ga_algorithms.GAProfile
        assert population_algorithms.GeneticAlgorithmOptimizer is ga_algorithms.GeneticAlgorithmOptimizer
        assert population_algorithms.InversionMutation is permutation_algorithms.InversionMutation
        assert population_algorithms.OrderCrossover is permutation_algorithms.OrderCrossover
        assert population_algorithms.SwapMutation is permutation_algorithms.SwapMutation
        assert population_algorithms.SpeciesGAProfile is species_ga_algorithms.SpeciesGAProfile
        assert population_algorithms.SpeciesConservingGeneticAlgorithmOptimizer is species_ga_algorithms.SpeciesConservingGeneticAlgorithmOptimizer
        assert population_algorithms.ClearingGAProfile is clearing_ga_algorithms.ClearingGAProfile
        assert population_algorithms.ClearingGeneticAlgorithmOptimizer is clearing_ga_algorithms.ClearingGeneticAlgorithmOptimizer
        assert population_algorithms.RestrictedTournamentGAProfile is restricted_tournament_ga_algorithms.RestrictedTournamentGAProfile
        assert population_algorithms.RestrictedTournamentGeneticAlgorithmOptimizer is restricted_tournament_ga_algorithms.RestrictedTournamentGeneticAlgorithmOptimizer

    def test_root_algorithms_facade_remains_convenience_reexport(self) -> None:
        assert root_algorithms.DEProfile is population_algorithms.DEProfile
        assert root_algorithms.DifferentialEvolutionOptimizer is population_algorithms.DifferentialEvolutionOptimizer
        assert root_algorithms.GAProfile is population_algorithms.GAProfile
        assert root_algorithms.GeneticAlgorithmOptimizer is population_algorithms.GeneticAlgorithmOptimizer
        assert root_algorithms.SpeciesGAProfile is population_algorithms.SpeciesGAProfile
        assert root_algorithms.SpeciesConservingGeneticAlgorithmOptimizer is population_algorithms.SpeciesConservingGeneticAlgorithmOptimizer
        assert root_algorithms.ClearingGAProfile is population_algorithms.ClearingGAProfile
        assert root_algorithms.ClearingGeneticAlgorithmOptimizer is population_algorithms.ClearingGeneticAlgorithmOptimizer
        assert root_algorithms.RestrictedTournamentGAProfile is population_algorithms.RestrictedTournamentGAProfile
        assert root_algorithms.RestrictedTournamentGeneticAlgorithmOptimizer is population_algorithms.RestrictedTournamentGeneticAlgorithmOptimizer
