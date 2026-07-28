"""Tests for resolved CSA optimizer configuration projection."""

from collections.abc import Sequence
from dataclasses import fields, replace

import numpy as np
import pytest
from typing_extensions import override

from variopt.algorithms.population.csa import (
    BoundedMutation,
    CSAAcceptancePolicy,
    CSAAdaptivePotential,
    CSAAdaptivePotentialAxis,
    CSABankGrowthPolicy,
    CSABankUpdatePolicy,
    CSABiasedPotential,
    CSAClusteringPolicy,
    CSACutoffSchedule,
    CSALocalRouteCutoffSchedule,
    CSANicheQualityPolicy,
    CSAOptimizer,
    CSAPerturbationSchedule,
    CSAPerturbationSpec,
    CSAProfile,
    CSAProposalPolicy,
    CSARefreshPolicy,
    CSAScoreModel,
    DifferentialEvolutionVariation,
    MixtureVariation,
    RandomResetMutation,
    UniformCrossover,
    derive_csa_defaults,
)
from variopt.algorithms.population.csa.manifest import (
    CSAComponentDescriptor,
    CSAConfigurationResolutionError,
)
from variopt.algorithms.population.csa.profile import CSAResolvedProfile
from variopt.algorithms.population.permutation import (
    InversionMutation,
    OrderCrossover,
    SwapMutation,
)
from variopt.diversity import DiversityMetric, StructuredSpaceDiversityMetric
from variopt.json_types import require_json_list, require_json_mapping
from variopt.operators import VariationOperator
from variopt.sampling import CandidateSampler, SearchSpaceSampler
from variopt.spaces import (
    ArraySpace,
    CategoricalSpace,
    IntegerSpace,
    PermutationSpace,
    RealSpace,
    RecordSpace,
    SearchSpace,
    TupleSpace,
)


def component_descriptor(
    identifier: str = "org.example.component",
) -> CSAComponentDescriptor:
    """Return one reusable caller-owned custom component descriptor."""
    return CSAComponentDescriptor(
        identifier=identifier,
        version=1,
        configuration={"mode": "test"},
    )


def integer_optimizer(
    *,
    profile: CSAProfile[int] | None = None,
    random_state: int | None = 7,
) -> CSAOptimizer[int, int]:
    """Return one exact built-in integer CSA optimizer."""
    return CSAOptimizer.from_space_defaults(
        space=IntegerSpace(-10, 10),
        bank_capacity=4,
        profile=profile,
        random_state=random_state,
    )


class CustomSpace(SearchSpace[int, int]):
    """Opaque integer search space used to exercise custom projection."""

    @override
    def normalize(self, raw_candidate: int) -> int:
        return raw_candidate

    @override
    def validate(self, candidate: int) -> None:
        _ = candidate

    @override
    def sample(self, random_state: np.random.RandomState) -> int:
        _ = random_state
        return 0


class CustomSampler(CandidateSampler[int]):
    """Opaque sampler used to exercise custom projection."""

    @override
    def sample(self, random_state: np.random.RandomState) -> int:
        _ = random_state
        return 0


class CustomMetric(DiversityMetric[int]):
    """Opaque metric used to exercise custom projection."""

    @override
    def distance(self, left: int, right: int) -> float:
        return float(abs(left - right))


class CustomMutation(VariationOperator[int]):
    """Opaque mutation used to exercise custom projection."""

    @property
    @override
    def arity(self) -> int:
        return 1

    @override
    def apply(
        self,
        parents: Sequence[int],
        random_state: np.random.RandomState,
    ) -> int:
        _ = random_state
        return parents[0]


class IntegerSpaceSubclass(IntegerSpace):
    """Integer-space subclass that must enter the custom descriptor path."""


class CustomPerturbationSchedule(CSAPerturbationSchedule[int]):
    """Perturbation-schedule subclass treated as one custom component."""


class CustomCutoffSchedule(CSACutoffSchedule):
    """Cutoff-schedule subclass treated as one custom component."""


class OpaqueInteger(int):
    """Integer subclass requiring an opaque configured-value descriptor."""


class DescriptorSubclass(CSAComponentDescriptor):
    """Descriptor subclass rejected by the exact manifest boundary."""


class PathString(str):
    """String subclass rejected as a semantic path segment."""


class PathTuple(tuple[str | int, ...]):
    """Tuple subclass rejected as a semantic component path."""


class CSAConfigurationProjectionTests:
    """Exercise exact built-in projection and semantic identity."""

    def test_projects_nested_builtin_configuration_without_descriptors(self) -> None:
        space = RecordSpace(
            count=IntegerSpace(1, 5, scale="log"),
            ratio=RealSpace(0.1, 2.0, scale="log"),
            category=CategoricalSpace(
                (
                    True,
                    2,
                    1.5,
                    "x",
                    b"\x00",
                    bytearray(b"\x01"),
                ),
            ),
            pair=TupleSpace(IntegerSpace(0, 2), RealSpace(-1.0, 1.0)),
        )
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            random_state=13,
        )

        manifest = optimizer.configuration_manifest()
        configuration = manifest.configuration
        projected_space = require_json_mapping(
            configuration["space"],
            field_name="configuration.space",
        )
        projected_space_configuration = require_json_mapping(
            projected_space["configuration"],
            field_name="configuration.space.configuration",
        )
        projected_fields = require_json_list(
            projected_space_configuration["fields"],
            field_name="configuration.space.configuration.fields",
        )
        categorical_field = require_json_mapping(
            projected_fields[2],
            field_name="configuration.space.configuration.fields[2]",
        )
        categorical_space = require_json_mapping(
            categorical_field["space"],
            field_name="configuration.space.configuration.fields[2].space",
        )
        categorical_configuration = require_json_mapping(
            categorical_space["configuration"],
            field_name=(
                "configuration.space.configuration.fields[2].space.configuration"
            ),
        )

        assert projected_space["identifier"] == "variopt.space.record"
        assert categorical_configuration["choices"] == [
            {"type": "boolean", "value": True},
            {"type": "integer", "value": 2},
            {"type": "float", "value": 1.5},
            {"type": "string", "value": "x"},
            {"type": "bytes", "hex": "00"},
            {"type": "bytearray", "hex": "01"},
        ]
        assert configuration["random_initialization"] == {
            "mode": "seeded",
            "seed": 13,
        }

        array_optimizer = CSAOptimizer.from_space_defaults(
            space=ArraySpace(IntegerSpace(-2, 2), length=3),
            bank_capacity=4,
            random_state=13,
        )
        assert (
            "variopt.space.array"
            in array_optimizer.configuration_manifest().canonical_json()
        )

    def test_permutation_defaults_project_exact_builtin_operators(self) -> None:
        optimizer = CSAOptimizer.from_space_defaults(
            space=PermutationSpace(size=6),
            bank_capacity=4,
            random_state=3,
        )

        manifest = optimizer.configuration_manifest()

        assert "variopt.operator.order-crossover" in manifest.canonical_json()
        assert "variopt.operator.inversion-mutation" in manifest.canonical_json()
        assert "variopt.operator.swap-mutation" in manifest.canonical_json()

    def test_equivalent_resolved_profiles_ignore_boundary_preset_shape(self) -> None:
        default_optimizer = integer_optimizer()
        resolved = default_optimizer.resolved_profile
        explicit_profile = CSAProfile(
            perturbation_schedule=resolved.perturbation_schedule,
            proposal_policy=resolved.proposal_policy,
            seed_count=resolved.seed_count,
            initial_new_bank_cut=resolved.initial_new_bank_cut,
            random_seed_mode=resolved.random_seed_mode,
            weighted_partner_selection=resolved.weighted_partner_selection,
            max_bank_capacity=resolved.max_bank_capacity,
            cutoff_schedule=resolved.cutoff_schedule,
            acceptance_policy=resolved.acceptance_policy,
            clustering_policy=resolved.clustering_policy,
            growth_policy=resolved.growth_policy,
            refresh_policy=resolved.refresh_policy,
            restart_lite=resolved.restart_lite,
            cycle_limit=resolved.cycle_limit,
            update_policy=resolved.update_policy,
            score_model=resolved.score_model,
        )
        explicit_optimizer = integer_optimizer(profile=explicit_profile)

        assert (
            explicit_optimizer.configuration_manifest()
            == default_optimizer.configuration_manifest()
        )

    def test_none_sampler_normalizes_to_the_space_sampler_semantics(self) -> None:
        space = IntegerSpace(-10, 10)
        defaults = derive_csa_defaults(space)
        profile = CSAProfile(
            perturbation_schedule=defaults.perturbation_schedule,
        )
        implicit_sampler = CSAOptimizer(
            space=space,
            diversity_metric=defaults.diversity_metric,
            bank_capacity=4,
            profile=profile,
            sampler=None,
            random_state=5,
        )
        explicit_sampler = replace(
            implicit_sampler,
            sampler=SearchSpaceSampler(space=space),
        )

        assert (
            implicit_sampler.configuration_manifest()
            == explicit_sampler.configuration_manifest()
        )

    def test_object_aliasing_does_not_affect_manifest_content(self) -> None:
        shared_space = IntegerSpace(-10, 10)
        aliased = CSAOptimizer.from_space_defaults(
            space=shared_space,
            bank_capacity=4,
            random_state=5,
        )

        independent_root_space = IntegerSpace(-10, 10)
        independent_schedule = CSAPerturbationSchedule(
            regular_family=(
                CSAPerturbationSpec(
                    UniformCrossover(space=IntegerSpace(-10, 10)),
                    count=2,
                ),
            ),
            initial_family=(
                CSAPerturbationSpec(
                    UniformCrossover(space=IntegerSpace(-10, 10)),
                    count=2,
                ),
            ),
            mutation_family=(
                CSAPerturbationSpec(
                    BoundedMutation(space=IntegerSpace(-10, 10)),
                    count=2,
                ),
                CSAPerturbationSpec(
                    RandomResetMutation(space=IntegerSpace(-10, 10)),
                    count=1,
                ),
            ),
        )
        independent = CSAOptimizer(
            space=independent_root_space,
            diversity_metric=StructuredSpaceDiversityMetric(
                space=independent_root_space,
            ),
            bank_capacity=4,
            profile=CSAProfile(
                perturbation_schedule=independent_schedule,
            ),
            sampler=SearchSpaceSampler(space=IntegerSpace(-10, 10)),
            random_state=5,
        )

        assert aliased.configuration_manifest() == independent.configuration_manifest()

    def test_every_resolved_profile_axis_affects_fingerprint(self) -> None:
        base_optimizer = integer_optimizer()
        base_profile = base_optimizer.profile
        schedule = base_optimizer.resolved_profile.perturbation_schedule
        modified_profiles: tuple[CSAProfile[int], ...] = (
            replace(
                base_profile,
                perturbation_schedule=replace(schedule, shuffle_children=False),
            ),
            replace(base_profile, proposal_policy=CSAProposalPolicy(enabled=True)),
            replace(base_profile, seed_count=6),
            replace(base_profile, initial_new_bank_cut=2),
            replace(base_profile, random_seed_mode=1),
            replace(base_profile, weighted_partner_selection=True),
            replace(base_profile, max_bank_capacity=12),
            replace(
                base_profile,
                cutoff_schedule=CSACutoffSchedule(reduction_factor=0.9),
            ),
            replace(
                base_profile,
                acceptance_policy=CSAAcceptancePolicy(initial_temperature=1.0),
            ),
            replace(
                base_profile,
                clustering_policy=CSAClusteringPolicy(enabled=True),
            ),
            replace(
                base_profile,
                growth_policy=CSABankGrowthPolicy(energy_gap_update_factor=2.0),
            ),
            replace(
                base_profile,
                refresh_policy=CSARefreshPolicy(mode="adaptive_refresh"),
            ),
            replace(base_profile, restart_lite=True),
            replace(base_profile, cycle_limit=11),
            replace(
                base_profile,
                update_policy=CSABankUpdatePolicy(
                    minimum_significant_score_gap_ratio=0.1,
                ),
            ),
            replace(
                base_profile,
                score_model=CSAScoreModel(
                    biased_potential=CSABiasedPotential(maximum_bias=12.0),
                ),
            ),
        )
        base_fingerprint = base_optimizer.configuration_manifest().fingerprint

        modified_fingerprints = {
            integer_optimizer(profile=profile).configuration_manifest().fingerprint
            for profile in modified_profiles
        }

        assert base_fingerprint not in modified_fingerprints
        assert len(modified_fingerprints) == 16

    def test_nested_policy_knobs_independently_affect_fingerprint(self) -> None:
        baseline = integer_optimizer()
        base_profile = baseline.profile
        variants: tuple[CSAProfile[int], ...] = (
            replace(
                base_profile,
                proposal_policy=CSAProposalPolicy(numeric_covariance_ridge=2e-6),
            ),
            replace(
                base_profile,
                cutoff_schedule=CSACutoffSchedule(
                    recover_steps=1,
                    recover_mode="score_gap_increase",
                ),
            ),
            replace(
                base_profile,
                cutoff_schedule=CSALocalRouteCutoffSchedule(
                    target_local_route_fraction=0.3,
                ),
            ),
            replace(
                base_profile,
                acceptance_policy=CSAAcceptancePolicy(
                    boltzmann_constant=0.002,
                ),
            ),
            replace(
                base_profile,
                clustering_policy=CSAClusteringPolicy(
                    update_mode="current_cluster",
                ),
            ),
            replace(
                base_profile,
                growth_policy=CSABankGrowthPolicy(
                    maximum_growth_per_generation=10,
                ),
            ),
            replace(
                base_profile,
                refresh_policy=CSARefreshPolicy(newcomer_first_round=False),
            ),
            replace(
                base_profile,
                update_policy=CSABankUpdatePolicy(crowding_penalty_ratio=0.5),
            ),
            replace(
                base_profile,
                update_policy=CSABankUpdatePolicy(
                    niche_quality_policy=CSANicheQualityPolicy(
                        mode="mean",
                        ratio=0.2,
                    ),
                ),
            ),
            replace(
                base_profile,
                score_model=CSAScoreModel(
                    biased_potential=CSABiasedPotential(sigma=0.2),
                ),
            ),
        )
        baseline_fingerprint = baseline.configuration_manifest().fingerprint

        variant_fingerprints = {
            integer_optimizer(profile=variant).configuration_manifest().fingerprint
            for variant in variants
        }

        assert baseline_fingerprint not in variant_fingerprints
        assert len(variant_fingerprints) == len(variants)

    def test_operator_order_count_bank_capacity_and_seed_affect_fingerprint(
        self,
    ) -> None:
        space = IntegerSpace(-10, 10)
        first_schedule = CSAPerturbationSchedule(
            mutation_family=(
                CSAPerturbationSpec(BoundedMutation(space=space), count=1),
                CSAPerturbationSpec(RandomResetMutation(space=space), count=2),
            ),
        )
        reordered_schedule = CSAPerturbationSchedule(
            mutation_family=tuple(reversed(first_schedule.mutation_family)),
        )
        recounted_schedule = replace(
            first_schedule,
            mutation_family=(
                replace(first_schedule.mutation_family[0], count=3),
                first_schedule.mutation_family[1],
            ),
        )

        def manifest_fingerprint(
            *,
            schedule: CSAPerturbationSchedule[int],
            bank_capacity: int = 4,
            random_state: int | None = 7,
        ) -> str:
            optimizer = CSAOptimizer(
                space=space,
                diversity_metric=StructuredSpaceDiversityMetric(space=space),
                bank_capacity=bank_capacity,
                profile=CSAProfile(perturbation_schedule=schedule),
                sampler=SearchSpaceSampler(space=space),
                random_state=random_state,
            )
            return optimizer.configuration_manifest().fingerprint

        fingerprints = {
            manifest_fingerprint(schedule=first_schedule),
            manifest_fingerprint(schedule=reordered_schedule),
            manifest_fingerprint(schedule=recounted_schedule),
            manifest_fingerprint(schedule=first_schedule, bank_capacity=5),
            manifest_fingerprint(schedule=first_schedule, random_state=None),
        }

        assert len(fingerprints) == 5

    def test_ordered_record_fields_and_categorical_binary_types_affect_identity(
        self,
    ) -> None:
        ordered = CSAOptimizer.from_space_defaults(
            space=RecordSpace(
                first=IntegerSpace(0, 2),
                second=RealSpace(0.0, 1.0),
            ),
            bank_capacity=4,
            random_state=1,
        )
        reordered = CSAOptimizer.from_space_defaults(
            space=RecordSpace(
                second=RealSpace(0.0, 1.0),
                first=IntegerSpace(0, 2),
            ),
            bank_capacity=4,
            random_state=1,
        )
        bytes_space = CSAOptimizer.from_space_defaults(
            space=CategoricalSpace((b"x",)),
            bank_capacity=4,
            random_state=1,
        )
        bytearray_space = CSAOptimizer.from_space_defaults(
            space=CategoricalSpace((bytearray(b"x"),)),
            bank_capacity=4,
            random_state=1,
        )

        assert (
            ordered.configuration_manifest().fingerprint
            != reordered.configuration_manifest().fingerprint
        )
        assert (
            bytes_space.configuration_manifest().fingerprint
            != bytearray_space.configuration_manifest().fingerprint
        )

    def test_nondeterministic_initialization_is_explicit_and_stable(self) -> None:
        optimizer = integer_optimizer(random_state=None)

        first_manifest = optimizer.configuration_manifest()
        second_manifest = optimizer.configuration_manifest()

        assert first_manifest == second_manifest
        assert first_manifest.configuration["random_initialization"] == {
            "mode": "nondeterministic",
        }

    def test_projects_local_route_adaptation_and_nested_score_models(self) -> None:
        space = IntegerSpace(-10, 10)
        regular_operator = MixtureVariation(
            (
                UniformCrossover(space=space, max_exchange_fraction=0.4),
                DifferentialEvolutionVariation(
                    space=space,
                    mutation_range=(0.4, 0.8),
                    recombination_probability=0.6,
                    n_cross=1,
                ),
            ),
            weights=(1.0, 2.0),
        )
        profile = CSAProfile(
            perturbation_schedule=CSAPerturbationSchedule(
                regular_family=(CSAPerturbationSpec(regular_operator),),
                mutation_family=(CSAPerturbationSpec(BoundedMutation(space=space)),),
            ),
            cutoff_schedule=CSALocalRouteCutoffSchedule(
                target_local_route_fraction=0.3,
                response=3.0,
            ),
            update_policy=CSABankUpdatePolicy(
                niche_quality_policy=CSANicheQualityPolicy(
                    mode="mean",
                    ratio=0.25,
                ),
            ),
            score_model=CSAScoreModel(
                biased_potential=CSABiasedPotential(
                    maximum_bias=50.0,
                    sigma=0.2,
                    sigma_reference="constant",
                ),
                adaptive_potential=CSAAdaptivePotential(
                    axes=(
                        CSAAdaptivePotentialAxis(
                            reference_candidate=0,
                            minimum_distance=0.0,
                            maximum_distance=1.0,
                            bin_count=5,
                        ),
                    ),
                    increment=0.2,
                    overflow_energy=200.0,
                ),
            ),
        )
        optimizer = CSAOptimizer(
            space=space,
            diversity_metric=StructuredSpaceDiversityMetric(space=space),
            bank_capacity=4,
            profile=profile,
            sampler=SearchSpaceSampler(space=space),
            random_state=5,
        )

        canonical_json = optimizer.configuration_manifest().canonical_json()

        assert "variopt.operator.mixture" in canonical_json
        assert "variopt.operator.differential-evolution" in canonical_json
        assert "variopt.csa.cutoff-schedule.local-route" in canonical_json
        assert "variopt.csa.niche-quality-policy" in canonical_json
        assert "variopt.csa.adaptive-potential-axis" in canonical_json

    @pytest.mark.parametrize("random_state", [True, -1, 2**32])
    def test_rejects_random_initialization_that_execution_cannot_materialize(
        self,
        random_state: int,
    ) -> None:
        optimizer = integer_optimizer(random_state=random_state)

        with pytest.raises((TypeError, ValueError)):
            optimizer.configuration_manifest()

    def test_accepts_the_maximum_numpy_uint32_seed(self) -> None:
        manifest = integer_optimizer(
            random_state=2**32 - 1,
        ).configuration_manifest()

        assert manifest.configuration["random_initialization"] == {
            "mode": "seeded",
            "seed": 2**32 - 1,
        }

    def test_each_builtin_space_topology_axis_affects_fingerprint(self) -> None:
        real_bound_left = CSAOptimizer.from_space_defaults(
            space=RealSpace(0.0, 1.0),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        real_bound_right = CSAOptimizer.from_space_defaults(
            space=RealSpace(0.0, 2.0),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        real_linear = CSAOptimizer.from_space_defaults(
            space=RealSpace(1.0, 2.0, scale="linear"),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        real_log = CSAOptimizer.from_space_defaults(
            space=RealSpace(1.0, 2.0, scale="log"),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        category_forward = CSAOptimizer.from_space_defaults(
            space=CategoricalSpace(("a", "b")),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        category_reverse = CSAOptimizer.from_space_defaults(
            space=CategoricalSpace(("b", "a")),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        array_short = CSAOptimizer.from_space_defaults(
            space=ArraySpace(IntegerSpace(0, 2), length=2),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        array_long = CSAOptimizer.from_space_defaults(
            space=ArraySpace(IntegerSpace(0, 2), length=3),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        tuple_forward = CSAOptimizer.from_space_defaults(
            space=TupleSpace(IntegerSpace(0, 2), RealSpace(0.0, 1.0)),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        tuple_reverse = CSAOptimizer.from_space_defaults(
            space=TupleSpace(RealSpace(0.0, 1.0), IntegerSpace(0, 2)),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        permutation_short = CSAOptimizer.from_space_defaults(
            space=PermutationSpace(4),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()
        permutation_long = CSAOptimizer.from_space_defaults(
            space=PermutationSpace(5),
            bank_capacity=4,
            random_state=1,
        ).configuration_manifest()

        assert real_bound_left.fingerprint != real_bound_right.fingerprint
        assert real_linear.fingerprint != real_log.fingerprint
        assert category_forward.fingerprint != category_reverse.fingerprint
        assert array_short.fingerprint != array_long.fingerprint
        assert tuple_forward.fingerprint != tuple_reverse.fingerprint
        assert permutation_short.fingerprint != permutation_long.fingerprint

    def test_manifest_projection_does_not_advance_or_replace_engine_state(self) -> None:
        optimizer = integer_optimizer(random_state=19)
        state_before = optimizer.create_initial_state()

        _ = optimizer.configuration_manifest()
        state_after = optimizer.create_initial_state()

        assert state_after == state_before


class CSAConfigurationCustomResolutionTests:
    """Exercise custom occurrence resolution and aggregate diagnostics."""

    def test_collects_missing_and_unused_paths_before_failing(self) -> None:
        optimizer = CSAOptimizer(
            space=CustomSpace(),
            diversity_metric=CustomMetric(),
            bank_capacity=4,
            profile=CSAProfile(
                perturbation_schedule=CSAPerturbationSchedule(
                    mutation_family=(CSAPerturbationSpec(CustomMutation()),),
                ),
            ),
            sampler=CustomSampler(),
            random_state=0,
        )

        with pytest.raises(CSAConfigurationResolutionError) as exc_info:
            optimizer.configuration_manifest(
                custom_component_descriptors={
                    ("unused",): component_descriptor("org.example.unused"),
                },
            )

        assert exc_info.value.missing_component_paths == (
            ("diversity_metric",),
            (
                "resolved_profile",
                "perturbation_schedule",
                "mutation_family",
                0,
                "operator",
            ),
            ("sampler",),
            ("space",),
        )
        assert exc_info.value.unused_component_paths == (("unused",),)

    def test_custom_configuration_succeeds_when_every_occurrence_is_described(
        self,
    ) -> None:
        optimizer = CSAOptimizer(
            space=CustomSpace(),
            diversity_metric=CustomMetric(),
            bank_capacity=4,
            profile=CSAProfile(
                perturbation_schedule=CSAPerturbationSchedule(
                    mutation_family=(CSAPerturbationSpec(CustomMutation()),),
                ),
            ),
            sampler=CustomSampler(),
            random_state=0,
        )
        descriptor = component_descriptor()

        manifest = optimizer.configuration_manifest(
            custom_component_descriptors={
                ("space",): descriptor,
                ("sampler",): descriptor,
                ("diversity_metric",): descriptor,
                (
                    "resolved_profile",
                    "perturbation_schedule",
                    "mutation_family",
                    0,
                    "operator",
                ): descriptor,
            },
        )

        assert manifest.configuration["space"] == descriptor.to_dict()
        assert "org.example.component" in manifest.canonical_json()

    def test_custom_descriptor_version_changes_projected_fingerprint(self) -> None:
        optimizer = CSAOptimizer(
            space=CustomSpace(),
            diversity_metric=CustomMetric(),
            bank_capacity=4,
            profile=CSAProfile(
                perturbation_schedule=CSAPerturbationSchedule(
                    mutation_family=(CSAPerturbationSpec(CustomMutation()),),
                ),
            ),
            sampler=CustomSampler(),
            random_state=0,
        )
        paths = (
            ("space",),
            ("sampler",),
            ("diversity_metric",),
            (
                "resolved_profile",
                "perturbation_schedule",
                "mutation_family",
                0,
                "operator",
            ),
        )
        version_one = CSAComponentDescriptor(
            identifier="org.example.component",
            version=1,
            configuration={},
        )
        version_two = CSAComponentDescriptor(
            identifier="org.example.component",
            version=2,
            configuration={},
        )

        first_manifest = optimizer.configuration_manifest(
            custom_component_descriptors={path: version_one for path in paths},
        )
        second_manifest = optimizer.configuration_manifest(
            custom_component_descriptors={path: version_two for path in paths},
        )

        assert first_manifest.fingerprint != second_manifest.fingerprint

    def test_custom_parent_consumes_only_its_own_semantic_location(self) -> None:
        space = IntegerSpace(-10, 10)
        schedule = CustomPerturbationSchedule(
            mutation_family=(CSAPerturbationSpec(BoundedMutation(space=space)),),
        )
        optimizer = CSAOptimizer(
            space=space,
            diversity_metric=StructuredSpaceDiversityMetric(space=space),
            bank_capacity=4,
            profile=CSAProfile(perturbation_schedule=schedule),
            random_state=0,
        )
        schedule_path = ("resolved_profile", "perturbation_schedule")
        nested_operator_path = (
            "resolved_profile",
            "perturbation_schedule",
            "mutation_family",
            0,
            "operator",
        )

        with pytest.raises(CSAConfigurationResolutionError) as exc_info:
            optimizer.configuration_manifest(
                custom_component_descriptors={
                    nested_operator_path: component_descriptor(),
                },
            )

        assert exc_info.value.missing_component_paths == (schedule_path,)
        assert exc_info.value.unused_component_paths == (nested_operator_path,)

        manifest = optimizer.configuration_manifest(
            custom_component_descriptors={
                schedule_path: component_descriptor("org.example.schedule"),
            },
        )
        assert "org.example.schedule" in manifest.canonical_json()

    def test_custom_policy_subclass_does_not_leak_builtin_fields(self) -> None:
        profile = CSAProfile[int](
            cutoff_schedule=CustomCutoffSchedule(),
        )
        optimizer = integer_optimizer(profile=profile)
        path = ("resolved_profile", "cutoff_schedule")

        with pytest.raises(CSAConfigurationResolutionError) as exc_info:
            optimizer.configuration_manifest()

        assert exc_info.value.missing_component_paths == (path,)

        manifest = optimizer.configuration_manifest(
            custom_component_descriptors={
                path: component_descriptor("org.example.cutoff"),
            },
        )
        assert "org.example.cutoff" in manifest.canonical_json()

    def test_nested_mixture_reports_the_exact_missing_child_operator(self) -> None:
        space = IntegerSpace(-10, 10)
        mixture = MixtureVariation[int](
            (
                BoundedMutation(space=space),
                CustomMutation(),
            ),
        )
        profile = CSAProfile(
            perturbation_schedule=CSAPerturbationSchedule(
                regular_family=(CSAPerturbationSpec(mixture),),
            ),
        )
        optimizer = CSAOptimizer(
            space=space,
            diversity_metric=StructuredSpaceDiversityMetric(space=space),
            bank_capacity=4,
            profile=profile,
            random_state=0,
        )
        path = (
            "resolved_profile",
            "perturbation_schedule",
            "regular_family",
            0,
            "operator",
            "operators",
            1,
        )

        with pytest.raises(CSAConfigurationResolutionError) as exc_info:
            optimizer.configuration_manifest()

        assert exc_info.value.missing_component_paths == (path,)

    def test_space_subclass_requires_a_descriptor_at_each_semantic_occurrence(
        self,
    ) -> None:
        space = IntegerSpaceSubclass(-10, 10)
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            random_state=0,
        )

        with pytest.raises(CSAConfigurationResolutionError) as exc_info:
            optimizer.configuration_manifest()

        assert set(exc_info.value.missing_component_paths) == {
            ("space",),
            ("sampler", "space"),
            ("diversity_metric", "space"),
            (
                "resolved_profile",
                "perturbation_schedule",
                "regular_family",
                0,
                "operator",
                "space",
            ),
            (
                "resolved_profile",
                "perturbation_schedule",
                "initial_family",
                0,
                "operator",
                "space",
            ),
            (
                "resolved_profile",
                "perturbation_schedule",
                "mutation_family",
                0,
                "operator",
                "space",
            ),
            (
                "resolved_profile",
                "perturbation_schedule",
                "mutation_family",
                1,
                "operator",
                "space",
            ),
        }

        descriptor = component_descriptor("org.example.integer-space")
        manifest = optimizer.configuration_manifest(
            custom_component_descriptors={
                path: descriptor for path in exc_info.value.missing_component_paths
            },
        )

        assert manifest.configuration["space"] == descriptor.to_dict()

    def test_record_field_names_do_not_leak_into_component_path_segments(self) -> None:
        space = RecordSpace(**{"": IntegerSpaceSubclass(0, 3)})
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            random_state=0,
        )

        with pytest.raises(CSAConfigurationResolutionError) as exc_info:
            optimizer.configuration_manifest()

        assert ("space", "fields", 0, "space") in (
            exc_info.value.missing_component_paths
        )
        assert all("" not in path for path in exc_info.value.missing_component_paths)

        descriptor = component_descriptor("org.example.record-child")
        manifest = optimizer.configuration_manifest(
            custom_component_descriptors={
                path: descriptor for path in exc_info.value.missing_component_paths
            },
        )
        assert "org.example.record-child" in manifest.canonical_json()

    def test_descriptor_for_exact_builtin_or_unknown_path_is_unused(self) -> None:
        descriptor = component_descriptor()

        with pytest.raises(CSAConfigurationResolutionError) as exc_info:
            integer_optimizer().configuration_manifest(
                custom_component_descriptors={
                    ("space",): descriptor,
                    ("unknown",): descriptor,
                },
            )

        assert exc_info.value.missing_component_paths == ()
        assert exc_info.value.unused_component_paths == (
            ("space",),
            ("unknown",),
        )

    def test_mixed_component_paths_sort_integer_segments_numerically(self) -> None:
        descriptor = component_descriptor()

        with pytest.raises(CSAConfigurationResolutionError) as exc_info:
            integer_optimizer().configuration_manifest(
                custom_component_descriptors={
                    ("unknown", 10**100): descriptor,
                    ("unknown", 2): descriptor,
                },
            )

        assert exc_info.value.unused_component_paths == (
            ("unknown", 2),
            ("unknown", 10**100),
        )

    def test_opaque_adaptive_reference_requires_its_own_descriptor(self) -> None:
        axis: CSAAdaptivePotentialAxis[int] = CSAAdaptivePotentialAxis(
            reference_candidate=OpaqueInteger(0),
            minimum_distance=0.0,
            maximum_distance=1.0,
            bin_count=2,
        )
        adaptive_potential: CSAAdaptivePotential[int] = CSAAdaptivePotential(
            axes=(axis,),
        )
        score_model: CSAScoreModel[int] = CSAScoreModel(
            adaptive_potential=adaptive_potential,
        )
        profile: CSAProfile[int] = CSAProfile(
            score_model=score_model,
        )
        optimizer = integer_optimizer(profile=profile)
        reference_path = (
            "resolved_profile",
            "score_model",
            "adaptive_potential",
            "axes",
            0,
            "reference_candidate",
        )

        with pytest.raises(CSAConfigurationResolutionError) as exc_info:
            optimizer.configuration_manifest()

        assert exc_info.value.missing_component_paths == (reference_path,)

        manifest = optimizer.configuration_manifest(
            custom_component_descriptors={
                reference_path: component_descriptor("org.example.reference"),
            },
        )
        assert "org.example.reference" in manifest.canonical_json()

    @pytest.mark.parametrize(
        "path",
        [
            (),
            ("",),
            (-1,),
        ],
    )
    def test_rejects_invalid_path_values(
        self,
        path: tuple[str | int, ...],
    ) -> None:
        with pytest.raises(ValueError):
            integer_optimizer().configuration_manifest(
                custom_component_descriptors={
                    path: component_descriptor(),
                },
            )

    def test_rejects_bool_and_scalar_subclass_path_segments(self) -> None:
        for path in ((True,), (PathString("space"),)):
            with pytest.raises(TypeError, match="exact strings or integers"):
                integer_optimizer().configuration_manifest(
                    custom_component_descriptors={
                        path: component_descriptor(),
                    },
                )

    def test_rejects_path_strings_that_are_not_valid_utf8(self) -> None:
        with pytest.raises(ValueError, match="valid UTF-8 text"):
            integer_optimizer().configuration_manifest(
                custom_component_descriptors={
                    ("\ud800",): component_descriptor(),
                },
            )

    def test_rejects_path_and_descriptor_subclasses(self) -> None:
        with pytest.raises(TypeError, match="exact built-in tuples"):
            integer_optimizer().configuration_manifest(
                custom_component_descriptors={
                    PathTuple(("space",)): component_descriptor(),
                },
            )

        with pytest.raises(TypeError, match="must use CSAComponentDescriptor"):
            integer_optimizer().configuration_manifest(
                custom_component_descriptors={
                    ("space",): DescriptorSubclass(
                        identifier="org.example.subclass",
                        version=1,
                        configuration={},
                    ),
                },
            )


class CSAConfigurationProjectionCoverageTests:
    """Lock every manually projected built-in dataclass field."""

    def test_space_and_component_dataclass_fields_are_explicitly_accounted_for(
        self,
    ) -> None:
        # Compatibility slots and compiled caches are deliberately listed here
        # even though projection excludes them from execution configuration.
        assert {field.name for field in fields(RealSpace)} == {"low", "high", "scale"}
        assert {field.name for field in fields(IntegerSpace)} == {
            "low",
            "high",
            "scale",
        }
        assert {field.name for field in fields(CategoricalSpace)} == {"choices"}
        assert {field.name for field in fields(PermutationSpace)} == {
            "size",
            "_index_space",
        }
        assert {field.name for field in fields(ArraySpace)} == {
            "element_space",
            "length",
        }
        assert {field.name for field in fields(SearchSpaceSampler)} == {
            "__orig_class__",
            "space",
        }
        assert {field.name for field in fields(StructuredSpaceDiversityMetric)} == {
            "__orig_class__",
            "space",
            "geometry",
            "part_values_geometry",
            "validated_part_values_geometry",
            "compiled_geometry_plan",
        }
        assert {field.name for field in fields(UniformCrossover)} == {
            "__orig_class__",
            "space",
            "structured_space",
            "max_exchange_fraction",
        }
        assert {field.name for field in fields(RandomResetMutation)} == {
            "__orig_class__",
            "space",
            "structured_space",
            "max_exchange_fraction",
        }
        assert {field.name for field in fields(BoundedMutation)} == {
            "__orig_class__",
            "space",
            "structured_space",
            "max_perturbation_fraction",
        }
        assert {field.name for field in fields(DifferentialEvolutionVariation)} == {
            "__orig_class__",
            "space",
            "structured_space",
            "mutation_range",
            "recombination_probability",
            "n_cross",
        }
        assert {field.name for field in fields(MixtureVariation)} == {
            "__orig_class__",
            "operators",
            "weights",
        }
        assert {field.name for field in fields(OrderCrossover)} == {
            "space",
            "max_segment_fraction",
        }
        assert {field.name for field in fields(SwapMutation)} == {
            "space",
            "max_swap_fraction",
        }
        assert {field.name for field in fields(InversionMutation)} == {
            "space",
            "max_inversion_fraction",
        }

    def test_resolved_profile_dataclass_fields_are_explicitly_accounted_for(
        self,
    ) -> None:
        assert {field.name for field in fields(CSAResolvedProfile)} == {
            "__orig_class__",
            "perturbation_schedule",
            "proposal_policy",
            "seed_count",
            "initial_new_bank_cut",
            "random_seed_mode",
            "weighted_partner_selection",
            "max_bank_capacity",
            "cutoff_schedule",
            "acceptance_policy",
            "clustering_policy",
            "growth_policy",
            "refresh_policy",
            "restart_lite",
            "cycle_limit",
            "update_policy",
            "score_model",
        }
        assert {field.name for field in fields(CSAPerturbationSchedule)} == {
            "__orig_class__",
            "regular_family",
            "initial_family",
            "mutation_family",
            "shuffle_children",
        }
        assert {field.name for field in fields(CSAPerturbationSpec)} == {
            "__orig_class__",
            "operator",
            "count",
        }
        assert {field.name for field in fields(CSAProposalPolicy)} == {
            "enabled",
            "family_bias_strength",
            "leaf_bias_strength",
            "local_displacement_leaf_bias_strength",
            "adaptation_decay",
            "minimum_family_weight",
            "minimum_leaf_weight",
            "numeric_covariance_strength",
            "numeric_covariance_min_observations",
            "numeric_covariance_ridge",
            "local_search_base_budget",
            "local_search_max_budget",
            "local_search_disable_failure_streak",
            "local_search_failure_cooldown_updates",
        }
        assert {field.name for field in fields(CSACutoffSchedule)} == {
            "initial_distance_cutoff",
            "minimum_distance_cutoff",
            "initial_distance_divisor",
            "minimum_distance_divisor",
            "reduction_method",
            "reduction_factor",
            "stagnation_update_limit",
            "cycle_increment_requires_minimum_cutoff",
            "recover_steps",
            "recover_mode",
        }
        assert {field.name for field in fields(CSALocalRouteCutoffSchedule)} == {
            "initial_distance_cutoff",
            "minimum_distance_cutoff",
            "initial_distance_divisor",
            "minimum_distance_divisor",
            "reduction_method",
            "reduction_factor",
            "stagnation_update_limit",
            "cycle_increment_requires_minimum_cutoff",
            "recover_steps",
            "recover_mode",
            "target_local_route_fraction",
            "response",
        }
        assert {field.name for field in fields(CSAAcceptancePolicy)} == {
            "initial_temperature",
            "reduction_factor",
            "minimum_temperature",
            "boltzmann_constant",
            "recover",
        }
        assert {field.name for field in fields(CSAClusteringPolicy)} == {
            "enabled",
            "cluster_cutoff_ratio",
            "cluster_distance_ratio",
            "update_mode",
        }
        assert {field.name for field in fields(CSABankGrowthPolicy)} == {
            "enabled",
            "maximum_capacity",
            "initial_energy_gap_limit",
            "energy_gap_update_mode",
            "energy_gap_update_factor",
            "maximum_growth_per_generation",
            "require_distance_cutoff",
        }
        assert {field.name for field in fields(CSARefreshPolicy)} == {
            "mode",
            "preserve_fraction",
            "newcomer_first_round",
        }
        assert {field.name for field in fields(CSANicheQualityPolicy)} == {
            "mode",
            "ratio",
        }
        assert {field.name for field in fields(CSABankUpdatePolicy)} == {
            "minimum_significant_score_gap_ratio",
            "local_update_mode",
            "far_update_mode",
            "crowding_penalty_ratio",
            "niche_quality_policy",
        }
        assert {field.name for field in fields(CSABiasedPotential)} == {
            "maximum_bias",
            "sigma",
            "sigma_reference",
        }
        assert {field.name for field in fields(CSAAdaptivePotentialAxis)} == {
            "__orig_class__",
            "reference_candidate",
            "minimum_distance",
            "maximum_distance",
            "bin_count",
        }
        assert {field.name for field in fields(CSAAdaptivePotential)} == {
            "__orig_class__",
            "axes",
            "increment",
            "overflow_energy",
        }
        assert {field.name for field in fields(CSAScoreModel)} == {
            "__orig_class__",
            "biased_potential",
            "adaptive_potential",
        }
