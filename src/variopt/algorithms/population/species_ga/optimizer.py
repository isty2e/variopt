"""Species-conserving genetic algorithm optimizer."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Observation, Proposal
from ....diversity import DiversityMetric
from ....execution import (
    ExecutionModel,
)
from ....methods import RunMethod
from ....operators import VariationOperator
from ....randomness import RandomSeed, RandomStateSnapshot
from ....sampling import CandidateSampler, SearchSpaceSampler
from ....spaces import PermutationSpace, SearchSpace
from ....typevars import CandidateT
from ..generational_ga.lifecycle import (
    GENERATIONAL_GA_EXECUTION_MODELS,
    GenerationalGAGenerationCommit,
    ask_generational_ga,
    create_initial_generational_ga_state,
    sort_generational_ga_population,
    tell_generational_ga,
)
from ..generational_ga.state import (
    GenerationalGAOptimizerState,
    GenerationalGAPopulationMember,
    GenerationalGAVariant,
)
from ..permutation.defaults import derive_permutation_variation_defaults
from .profile import SpeciesGAProfile, SpeciesGAResolvedProfile

BoundaryT = TypeVar("BoundaryT")


@dataclass(frozen=True, slots=True)
class SpeciesConservingGeneticAlgorithmOptimizer(FrozenGenericSlotsCompat,
    RunMethod[
        GenerationalGAOptimizerState[CandidateT],
        Proposal[CandidateT],
        Observation[CandidateT],
    ],
    Generic[BoundaryT, CandidateT],
):
    """Single-objective GA with species-conserving survival.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, CandidateT]
        Search space used to validate and sample candidates.
    population_size : int
        Number of members maintained in each generation.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used to form and protect species.
    crossover_operator : VariationOperator[CandidateT] | None, optional
        Optional crossover operator applied before mutation.
    mutation_operator : VariationOperator[CandidateT] | None, optional
        Optional mutation operator applied after crossover or cloning.
    profile : SpeciesGAProfile, optional
        Boundary-level profile controlling species radius and capacity.
    sampler : CandidateSampler[CandidateT] | None, optional
        Optional sampler for initial population generation.
    random_state : RandomSeed, optional
        Seed or random-state object used to initialize optimizer randomness.
    """

    space: SearchSpace[BoundaryT, CandidateT]
    population_size: int
    diversity_metric: DiversityMetric[CandidateT]
    crossover_operator: VariationOperator[CandidateT] | None = field(default=None, kw_only=True)
    mutation_operator: VariationOperator[CandidateT] | None = field(default=None, kw_only=True)
    profile: SpeciesGAProfile = field(default_factory=SpeciesGAProfile, kw_only=True)
    sampler: CandidateSampler[CandidateT] | None = field(default=None, kw_only=True)
    random_state: RandomSeed = None
    resolved_profile: SpeciesGAResolvedProfile = field(init=False, repr=False)
    resolved_sampler: CandidateSampler[CandidateT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve and validate the canonical species-GA configuration.

        Raises
        ------
        ValueError
            Raised when population size, operator arities, or profile settings
            are inconsistent.
        """
        if self.population_size <= 0:
            msg = "population_size must be positive"
            raise ValueError(msg)

        resolved_profile = self.profile.resolve()
        object.__setattr__(self, "resolved_profile", resolved_profile)
        object.__setattr__(
            self,
            "resolved_sampler",
            SearchSpaceSampler(self.space) if self.sampler is None else self.sampler,
        )

        if self.crossover_operator is None and self.mutation_operator is None:
            msg = "at least one variation operator must be provided"
            raise ValueError(msg)

        if self.crossover_operator is not None and self.crossover_operator.arity < 2:
            msg = "crossover_operator arity must be at least 2"
            raise ValueError(msg)

        if self.mutation_operator is not None and self.mutation_operator.arity != 1:
            msg = "mutation_operator arity must be exactly 1"
            raise ValueError(msg)

        if resolved_profile.tournament_size > self.population_size:
            msg = "tournament_size must not exceed population_size"
            raise ValueError(msg)

    @staticmethod
    def from_permutation_space_defaults(
        *,
        space: PermutationSpace,
        population_size: int,
        profile: SpeciesGAProfile | None = None,
        sampler: CandidateSampler[tuple[int, ...]] | None = None,
        diversity_metric: DiversityMetric[tuple[int, ...]] | None = None,
        crossover_operator: VariationOperator[tuple[int, ...]] | None = None,
        mutation_operator: VariationOperator[tuple[int, ...]] | None = None,
        random_state: RandomSeed = None,
    ) -> "SpeciesConservingGeneticAlgorithmOptimizer[Sequence[int], tuple[int, ...]]":
        """Build a permutation-specialized species-conserving GA.

        Parameters
        ----------
        space : PermutationSpace
            Permutation search space optimized by the returned GA.
        population_size : int
            Number of members maintained in each generation.
        profile : SpeciesGAProfile | None, default=None
            Optional profile override.
        sampler : CandidateSampler[tuple[int, ...]] | None, default=None
            Optional initial-population sampler override.
        diversity_metric : DiversityMetric[tuple[int, ...]] | None, default=None
            Optional diversity metric override.
        crossover_operator : VariationOperator[tuple[int, ...]] | None, default=None
            Optional crossover operator override.
        mutation_operator : VariationOperator[tuple[int, ...]] | None, default=None
            Optional mutation operator override.
        random_state : RandomSeed, optional
            Seed or random-state object used to initialize optimizer
            randomness.

        Returns
        -------
        SpeciesConservingGeneticAlgorithmOptimizer[Sequence[int], tuple[int, ...]]
            Species-conserving GA configured for permutation-safe defaults.
        """
        defaults = derive_permutation_variation_defaults(space)
        return SpeciesConservingGeneticAlgorithmOptimizer(
            space=space,
            population_size=population_size,
            diversity_metric=(
                defaults.diversity_metric
                if diversity_metric is None
                else diversity_metric
            ),
            crossover_operator=(
                defaults.crossover_operator
                if crossover_operator is None
                else crossover_operator
            ),
            mutation_operator=(
                defaults.mutation_operator
                if mutation_operator is None
                else mutation_operator
            ),
            profile=SpeciesGAProfile() if profile is None else profile,
            sampler=sampler,
            random_state=random_state,
        )

    @override
    def create_initial_state(self) -> GenerationalGAOptimizerState[CandidateT]:
        """Create the initial immutable optimizer state.

        Returns
        -------
        GenerationalGAOptimizerState[CandidateT]
            State with initialized randomness and no observed population.
        """
        return create_initial_generational_ga_state(
            self.random_state,
            variant=GenerationalGAVariant.SPECIES,
        )

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        """Return supported execution models.

        Returns
        -------
        frozenset[ExecutionModel]
            Execution models accepted by the species-GA ask/tell contract.
        """
        return GENERATIONAL_GA_EXECUTION_MODELS

    @override
    def is_exhausted(self, state: GenerationalGAOptimizerState[CandidateT]) -> bool:
        """Report whether the optimizer can emit more proposals.

        Parameters
        ----------
        state : GenerationalGAOptimizerState[CandidateT]
            Optimizer state to inspect.

        Returns
        -------
        bool
            Always ``False`` for the current unbounded implementation.
        """
        _ = state
        return False

    @override
    def ask(
        self,
        state: GenerationalGAOptimizerState[CandidateT],
        batch_size: int = 1,
    ) -> tuple[
        tuple[Proposal[CandidateT], ...],
        GenerationalGAOptimizerState[CandidateT],
    ]:
        """Emit the next proposal batch and advanced optimizer state.

        Parameters
        ----------
        state : GenerationalGAOptimizerState[CandidateT]
            Current immutable optimizer state.
        batch_size : int, default=1
            Maximum number of proposals to emit.

        Returns
        -------
        tuple[tuple[Proposal[CandidateT], ...], GenerationalGAOptimizerState[CandidateT]]
            Proposal batch together with the advanced immutable state.

        Raises
        ------
        ValueError
            Raised when ``batch_size`` is not positive.
        RuntimeError
            Raised when outstanding proposals have not yet been observed.
        """
        return ask_generational_ga(
            self,
            state,
            batch_size=batch_size,
            proposal_id_prefix="species-ga-",
            variant=GenerationalGAVariant.SPECIES,
        )

    @override
    def tell(
        self,
        state: GenerationalGAOptimizerState[CandidateT],
        observations: Sequence[Observation[CandidateT]],
    ) -> GenerationalGAOptimizerState[CandidateT]:
        """Advance the optimizer state with one observed proposal batch.

        Parameters
        ----------
        state : GenerationalGAOptimizerState[CandidateT]
            Current immutable optimizer state.
        observations : Sequence[Observation[CandidateT]]
            Observations aligned with the currently pending proposals.

        Returns
        -------
        GenerationalGAOptimizerState[CandidateT]
            Updated immutable optimizer state after buffering or committing the
            observed members.

        Raises
        ------
        ValueError
            Raised when observation count or ordering does not match the
            pending proposals.
        """
        return tell_generational_ga(
            state,
            observations,
            population_size=self.population_size,
            build_next_population=self._build_next_population,
            variant=GenerationalGAVariant.SPECIES,
        )

    def _build_next_population(
        self,
        *,
        parents: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        offspring: tuple[GenerationalGAPopulationMember[CandidateT], ...],
        random_state: RandomStateSnapshot,
    ) -> GenerationalGAGenerationCommit[CandidateT]:
        _ = random_state
        candidate_pool = sort_generational_ga_population(parents + offspring)
        species_members: list[
            list[tuple[int, GenerationalGAPopulationMember[CandidateT]]]
        ] = []
        for member_index, member in enumerate(candidate_pool):
            assigned = False
            for members in species_members:
                seed_member = members[0][1]
                if (
                    self.diversity_metric.distance(
                        member.candidate,
                        seed_member.candidate,
                    )
                    < self.resolved_profile.species_radius
                ):
                    if len(members) < self.resolved_profile.species_capacity:
                        members.append((member_index, member))
                    assigned = True
                    break
            if assigned:
                continue

            species_members.append([(member_index, member)])

        protected_entries = tuple(
            (member_index, species_member_index == 0, member)
            for members in species_members
            for species_member_index, (member_index, member) in enumerate(members)
        )
        if len(protected_entries) >= self.population_size:
            seed_prioritized_population = tuple(
                member
                for _, _, member in sorted(
                    protected_entries,
                    key=lambda entry: (not entry[1], entry[2].score),
                )[: self.population_size]
            )
            return GenerationalGAGenerationCommit(
                population=sort_generational_ga_population(
                    seed_prioritized_population,
                ),
            )

        protected_indices = frozenset(
            member_index for member_index, _, _ in protected_entries
        )
        protected_population = tuple(member for _, _, member in protected_entries)
        backfill_members = tuple(
            member
            for member_index, member in enumerate(candidate_pool)
            if member_index not in protected_indices
        )
        next_population = sort_generational_ga_population(
            protected_population
            + backfill_members[: self.population_size - len(protected_population)]
        )

        return GenerationalGAGenerationCommit(
            population=next_population,
        )
