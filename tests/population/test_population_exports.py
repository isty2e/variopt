"""Regression tests for the population-family facade exports."""

import variopt.algorithms as root_algorithms
import variopt.algorithms.population as population_algorithms
import variopt.algorithms.population.clearing_ga as clearing_ga_algorithms
import variopt.algorithms.population.csa as csa_algorithms
import variopt.algorithms.population.csa.progression.cutoff as csa_cutoff
import variopt.algorithms.population.de as de_algorithms
import variopt.algorithms.population.ga as ga_algorithms
import variopt.algorithms.population.generational_ga as generational_ga_algorithms
import variopt.algorithms.population.generational_ga.state as generational_ga_state
import variopt.algorithms.population.permutation as permutation_algorithms
import variopt.algorithms.population.restricted_tournament_ga as restricted_tournament_ga_algorithms
import variopt.algorithms.population.species_ga as species_ga_algorithms

EXPECTED_POPULATION_ALL = (
    "ClearingGAProfile",
    "ClearingGeneticAlgorithmOptimizer",
    "CSAOptimizer",
    "CSAProfile",
    "DEProfile",
    "DifferentialEvolutionOptimizer",
    "GAProfile",
    "GeneticAlgorithmOptimizer",
    "GenerationalGAMemberBuffer",
    "GenerationalGAOptimizerState",
    "GenerationalGAPopulationMember",
    "GenerationalGAVariant",
    "InversionMutation",
    "OrderCrossover",
    "RestrictedTournamentGAProfile",
    "RestrictedTournamentGeneticAlgorithmOptimizer",
    "SpeciesConservingGeneticAlgorithmOptimizer",
    "SpeciesGAProfile",
    "SwapMutation",
)

EXPECTED_CSA_ALL = (
    "Bank",
    "BoundedMutation",
    "CSAAcceptancePolicy",
    "CSAAdaptivePotential",
    "CSAAdaptivePotentialAxis",
    "CSACutoffObservation",
    "CSACutoffSchedule",
    "CSALocalRouteCutoffSchedule",
    "CSADefaultComponents",
    "CSAClusteringPolicy",
    "CSABankUpdatePolicy",
    "CSABiasedPotential",
    "CSABankGrowthPolicy",
    "CSANicheQualityPolicy",
    "CSAOptimizer",
    "CSAProfile",
    "CSAProposalPolicy",
    "CSAPerturbationSchedule",
    "CSAPerturbationSpec",
    "CSARefreshPolicy",
    "CSAScoreModel",
    "DifferentialEvolutionVariation",
    "MixtureVariation",
    "RandomResetMutation",
    "UniformCrossover",
    "derive_csa_defaults",
)

EXPECTED_ROOT_ALGORITHMS_ALL = (
    "AlgorithmProfile",
    "ClearingGeneticAlgorithmOptimizer",
    "ClearingGAProfile",
    "DEProfile",
    "DifferentialEvolutionOptimizer",
    "GAProfile",
    "GeneticAlgorithmOptimizer",
    "ScipyMinimizeKernel",
    "ScipyMinimizeMethod",
    "RestrictedTournamentGAProfile",
    "RestrictedTournamentGeneticAlgorithmOptimizer",
    "SpeciesConservingGeneticAlgorithmOptimizer",
    "SpeciesGAProfile",
    "StructuredHillClimbKernel",
    "StructuredIteratedLocalSearchKernel",
    "StructuredKickPolicy",
    "StructuredStochasticNeighborhoodKernel",
    "StructuredScheduledLocalSearchKernel",
    "StructuredVariableNeighborhoodKernel",
    "StructuredVariableNeighborhoodStage",
)


class PopulationFacadeExportTests:
    """Lock the canonical population-family facade surface."""

    def test_population_facade_reexports_population_family_entry_points(self) -> None:
        assert tuple(population_algorithms.__all__) == EXPECTED_POPULATION_ALL
        assert population_algorithms.CSAOptimizer is csa_algorithms.CSAOptimizer
        assert population_algorithms.CSAProfile is csa_algorithms.CSAProfile
        assert population_algorithms.DEProfile is de_algorithms.DEProfile
        assert (
            population_algorithms.DifferentialEvolutionOptimizer
            is de_algorithms.DifferentialEvolutionOptimizer
        )
        assert population_algorithms.GAProfile is ga_algorithms.GAProfile
        assert (
            population_algorithms.GeneticAlgorithmOptimizer
            is ga_algorithms.GeneticAlgorithmOptimizer
        )
        assert (
            population_algorithms.InversionMutation
            is permutation_algorithms.InversionMutation
        )
        assert (
            population_algorithms.OrderCrossover
            is permutation_algorithms.OrderCrossover
        )
        assert population_algorithms.SwapMutation is permutation_algorithms.SwapMutation
        assert (
            population_algorithms.SpeciesGAProfile
            is species_ga_algorithms.SpeciesGAProfile
        )
        assert (
            population_algorithms.SpeciesConservingGeneticAlgorithmOptimizer
            is species_ga_algorithms.SpeciesConservingGeneticAlgorithmOptimizer
        )
        assert (
            population_algorithms.ClearingGAProfile
            is clearing_ga_algorithms.ClearingGAProfile
        )
        assert (
            population_algorithms.ClearingGeneticAlgorithmOptimizer
            is clearing_ga_algorithms.ClearingGeneticAlgorithmOptimizer
        )
        assert (
            population_algorithms.RestrictedTournamentGAProfile
            is restricted_tournament_ga_algorithms.RestrictedTournamentGAProfile
        )
        assert (
            population_algorithms.RestrictedTournamentGeneticAlgorithmOptimizer
            is restricted_tournament_ga_algorithms.RestrictedTournamentGeneticAlgorithmOptimizer
        )
        assert (
            population_algorithms.GenerationalGAMemberBuffer
            is generational_ga_state.GenerationalGAMemberBuffer
        )
        assert (
            population_algorithms.GenerationalGAOptimizerState
            is generational_ga_state.GenerationalGAOptimizerState
        )
        assert (
            population_algorithms.GenerationalGAPopulationMember
            is generational_ga_state.GenerationalGAPopulationMember
        )
        assert (
            population_algorithms.GenerationalGAVariant
            is generational_ga_state.GenerationalGAVariant
        )

        internal_names = (
            "GENERATIONAL_GA_EXECUTION_MODELS",
            "GenerationalGAGenerationCommit",
            "ask_generational_ga",
            "create_initial_generational_ga_state",
            "sort_generational_ga_population",
            "tell_generational_ga",
        )
        assert not hasattr(generational_ga_algorithms, "__all__")
        assert all(name not in population_algorithms.__all__ for name in internal_names)
        assert all(not hasattr(population_algorithms, name) for name in internal_names)

    def test_root_algorithms_facade_remains_convenience_reexport(self) -> None:
        assert tuple(root_algorithms.__all__) == EXPECTED_ROOT_ALGORITHMS_ALL
        assert root_algorithms.DEProfile is population_algorithms.DEProfile
        assert (
            root_algorithms.DifferentialEvolutionOptimizer
            is population_algorithms.DifferentialEvolutionOptimizer
        )
        assert root_algorithms.GAProfile is population_algorithms.GAProfile
        assert (
            root_algorithms.GeneticAlgorithmOptimizer
            is population_algorithms.GeneticAlgorithmOptimizer
        )
        assert (
            root_algorithms.SpeciesGAProfile is population_algorithms.SpeciesGAProfile
        )
        assert (
            root_algorithms.SpeciesConservingGeneticAlgorithmOptimizer
            is population_algorithms.SpeciesConservingGeneticAlgorithmOptimizer
        )
        assert (
            root_algorithms.ClearingGAProfile is population_algorithms.ClearingGAProfile
        )
        assert (
            root_algorithms.ClearingGeneticAlgorithmOptimizer
            is population_algorithms.ClearingGeneticAlgorithmOptimizer
        )
        assert (
            root_algorithms.RestrictedTournamentGAProfile
            is population_algorithms.RestrictedTournamentGAProfile
        )
        assert (
            root_algorithms.RestrictedTournamentGeneticAlgorithmOptimizer
            is population_algorithms.RestrictedTournamentGeneticAlgorithmOptimizer
        )


class CSAFacadeExportTests:
    """Lock the advanced CSA policy facade surface."""

    def test_csa_facade_reexports_cutoff_contracts(self) -> None:
        assert tuple(csa_algorithms.__all__) == EXPECTED_CSA_ALL
        assert csa_algorithms.CSACutoffObservation is csa_cutoff.CSACutoffObservation
        assert csa_algorithms.CSACutoffSchedule is csa_cutoff.CSACutoffSchedule
        assert (
            csa_algorithms.CSALocalRouteCutoffSchedule
            is csa_cutoff.CSALocalRouteCutoffSchedule
        )
