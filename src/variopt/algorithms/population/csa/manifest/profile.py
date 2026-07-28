"""Projection of canonical resolved CSA profile values."""

from typing import TypeVar

from .....json_types import JSONDict, JSONValue
from .....spaces.serialization import space_candidate_to_dict
from .....spaces.structured import is_space_candidate_value
from ..banking.clustering import CSAClusteringPolicy
from ..banking.growth import CSABankGrowthPolicy
from ..banking.update import CSABankUpdatePolicy, CSANicheQualityPolicy
from ..generation.perturbation import CSAPerturbationSchedule, CSAPerturbationSpec
from ..generation.proposal import CSAProposalPolicy
from ..profile import CSAResolvedProfile
from ..progression.cutoff.policy import (
    CSACutoffSchedule,
    CSALocalRouteCutoffSchedule,
)
from ..progression.refresh import CSARefreshPolicy
from ..scoring.acceptance import CSAAcceptancePolicy
from ..scoring.model import (
    CSAAdaptivePotential,
    CSAAdaptivePotentialAxis,
    CSABiasedPotential,
    CSAScoreModel,
)
from .components import project_variation_operator
from .nodes import builtin_component_node
from .resolution import CSAComponentDescriptorResolver, CSAComponentPath

CandidateT = TypeVar("CandidateT")


def project_resolved_profile(
    profile: CSAResolvedProfile[CandidateT],
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONDict:
    """Project every canonical optimizer-facing CSA profile field."""
    return {
        "perturbation_schedule": project_perturbation_schedule(
            profile.perturbation_schedule,
            path=(*path, "perturbation_schedule"),
            resolver=resolver,
        ),
        "proposal_policy": project_proposal_policy(
            profile.proposal_policy,
            path=(*path, "proposal_policy"),
            resolver=resolver,
        ),
        "seed_count": profile.seed_count,
        "initial_new_bank_cut": profile.initial_new_bank_cut,
        "random_seed_mode": profile.random_seed_mode,
        "weighted_partner_selection": profile.weighted_partner_selection,
        "max_bank_capacity": profile.max_bank_capacity,
        "cutoff_schedule": project_cutoff_schedule(
            profile.cutoff_schedule,
            path=(*path, "cutoff_schedule"),
            resolver=resolver,
        ),
        "acceptance_policy": project_acceptance_policy(
            profile.acceptance_policy,
            path=(*path, "acceptance_policy"),
            resolver=resolver,
        ),
        "clustering_policy": project_clustering_policy(
            profile.clustering_policy,
            path=(*path, "clustering_policy"),
            resolver=resolver,
        ),
        "growth_policy": project_growth_policy(
            profile.growth_policy,
            path=(*path, "growth_policy"),
            resolver=resolver,
        ),
        "refresh_policy": project_refresh_policy(
            profile.refresh_policy,
            path=(*path, "refresh_policy"),
            resolver=resolver,
        ),
        "restart_lite": profile.restart_lite,
        "cycle_limit": profile.cycle_limit,
        "update_policy": project_update_policy(
            profile.update_policy,
            path=(*path, "update_policy"),
            resolver=resolver,
        ),
        "score_model": project_score_model(
            profile.score_model,
            path=(*path, "score_model"),
            resolver=resolver,
        ),
    }


def project_perturbation_schedule(
    schedule: CSAPerturbationSchedule[CandidateT],
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in perturbation schedule."""
    if type(schedule) is not CSAPerturbationSchedule:
        return resolver.resolve_custom_component(path)

    return builtin_component_node(
        identifier="variopt.csa.perturbation-schedule",
        configuration={
            "regular_family": project_perturbation_family(
                schedule.regular_family,
                path=(*path, "regular_family"),
                resolver=resolver,
            ),
            "initial_family": project_perturbation_family(
                schedule.initial_family,
                path=(*path, "initial_family"),
                resolver=resolver,
            ),
            "mutation_family": project_perturbation_family(
                schedule.mutation_family,
                path=(*path, "mutation_family"),
                resolver=resolver,
            ),
            "shuffle_children": schedule.shuffle_children,
        },
    )


def project_perturbation_family(
    family: tuple[CSAPerturbationSpec[CandidateT], ...],
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> list[JSONValue]:
    """Project an ordered CSA operator family."""
    return [
        project_perturbation_spec(
            spec,
            path=(*path, index),
            resolver=resolver,
        )
        for index, spec in enumerate(family)
    ]


def project_perturbation_spec(
    spec: CSAPerturbationSpec[CandidateT],
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in perturbation family member."""
    if type(spec) is not CSAPerturbationSpec:
        return resolver.resolve_custom_component(path)

    return builtin_component_node(
        identifier="variopt.csa.perturbation-spec",
        configuration={
            "operator": project_variation_operator(
                spec.operator,
                path=(*path, "operator"),
                resolver=resolver,
            ),
            "count": spec.count,
        },
    )


def project_proposal_policy(
    policy: CSAProposalPolicy,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in adaptive proposal policy."""
    if type(policy) is not CSAProposalPolicy:
        return resolver.resolve_custom_component(path)

    return builtin_component_node(
        identifier="variopt.csa.proposal-policy",
        configuration={
            "enabled": policy.enabled,
            "family_bias_strength": policy.family_bias_strength,
            "leaf_bias_strength": policy.leaf_bias_strength,
            "local_displacement_leaf_bias_strength": (
                policy.local_displacement_leaf_bias_strength
            ),
            "adaptation_decay": policy.adaptation_decay,
            "minimum_family_weight": policy.minimum_family_weight,
            "minimum_leaf_weight": policy.minimum_leaf_weight,
            "numeric_covariance_strength": policy.numeric_covariance_strength,
            "numeric_covariance_min_observations": (
                policy.numeric_covariance_min_observations
            ),
            "numeric_covariance_ridge": policy.numeric_covariance_ridge,
            "local_search_base_budget": policy.local_search_base_budget,
            "local_search_max_budget": policy.local_search_max_budget,
            "local_search_disable_failure_streak": (
                policy.local_search_disable_failure_streak
            ),
            "local_search_failure_cooldown_updates": (
                policy.local_search_failure_cooldown_updates
            ),
        },
    )


def project_cutoff_schedule(
    schedule: CSACutoffSchedule,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in cutoff schedule."""
    if isinstance(schedule, CSALocalRouteCutoffSchedule):
        if type(schedule) is not CSALocalRouteCutoffSchedule:
            return resolver.resolve_custom_component(path)
        configuration = cutoff_schedule_configuration(schedule)
        configuration["target_local_route_fraction"] = (
            schedule.target_local_route_fraction
        )
        configuration["response"] = schedule.response
        return builtin_component_node(
            identifier="variopt.csa.cutoff-schedule.local-route",
            configuration=configuration,
        )

    if type(schedule) is not CSACutoffSchedule:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.cutoff-schedule",
        configuration=cutoff_schedule_configuration(schedule),
    )


def cutoff_schedule_configuration(schedule: CSACutoffSchedule) -> JSONDict:
    """Return the explicitly represented base cutoff configuration."""
    return {
        "initial_distance_cutoff": schedule.initial_distance_cutoff,
        "minimum_distance_cutoff": schedule.minimum_distance_cutoff,
        "initial_distance_divisor": schedule.initial_distance_divisor,
        "minimum_distance_divisor": schedule.minimum_distance_divisor,
        "reduction_method": schedule.reduction_method,
        "reduction_factor": schedule.reduction_factor,
        "stagnation_update_limit": schedule.stagnation_update_limit,
        "cycle_increment_requires_minimum_cutoff": (
            schedule.cycle_increment_requires_minimum_cutoff
        ),
        "recover_steps": schedule.recover_steps,
        "recover_mode": schedule.recover_mode,
    }


def project_acceptance_policy(
    policy: CSAAcceptancePolicy,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in acceptance policy."""
    if type(policy) is not CSAAcceptancePolicy:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.acceptance-policy",
        configuration={
            "initial_temperature": policy.initial_temperature,
            "reduction_factor": policy.reduction_factor,
            "minimum_temperature": policy.minimum_temperature,
            "boltzmann_constant": policy.boltzmann_constant,
            "recover": policy.recover,
        },
    )


def project_clustering_policy(
    policy: CSAClusteringPolicy,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in clustering policy."""
    if type(policy) is not CSAClusteringPolicy:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.clustering-policy",
        configuration={
            "enabled": policy.enabled,
            "cluster_cutoff_ratio": policy.cluster_cutoff_ratio,
            "cluster_distance_ratio": policy.cluster_distance_ratio,
            "update_mode": policy.update_mode,
        },
    )


def project_growth_policy(
    policy: CSABankGrowthPolicy,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in bank-growth policy."""
    if type(policy) is not CSABankGrowthPolicy:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.bank-growth-policy",
        configuration={
            "enabled": policy.enabled,
            "maximum_capacity": policy.maximum_capacity,
            "initial_energy_gap_limit": policy.initial_energy_gap_limit,
            "energy_gap_update_mode": policy.energy_gap_update_mode,
            "energy_gap_update_factor": policy.energy_gap_update_factor,
            "maximum_growth_per_generation": policy.maximum_growth_per_generation,
            "require_distance_cutoff": policy.require_distance_cutoff,
        },
    )


def project_refresh_policy(
    policy: CSARefreshPolicy,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in refresh policy."""
    if type(policy) is not CSARefreshPolicy:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.refresh-policy",
        configuration={
            "mode": policy.mode,
            "preserve_fraction": policy.preserve_fraction,
            "newcomer_first_round": policy.newcomer_first_round,
        },
    )


def project_update_policy(
    policy: CSABankUpdatePolicy,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in bank-update policy."""
    if type(policy) is not CSABankUpdatePolicy:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.bank-update-policy",
        configuration={
            "minimum_significant_score_gap_ratio": (
                policy.minimum_significant_score_gap_ratio
            ),
            "local_update_mode": policy.local_update_mode,
            "far_update_mode": policy.far_update_mode,
            "crowding_penalty_ratio": policy.crowding_penalty_ratio,
            "niche_quality_policy": project_niche_quality_policy(
                policy.niche_quality_policy,
                path=(*path, "niche_quality_policy"),
                resolver=resolver,
            ),
        },
    )


def project_niche_quality_policy(
    policy: CSANicheQualityPolicy,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in niche-quality policy."""
    if type(policy) is not CSANicheQualityPolicy:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.niche-quality-policy",
        configuration={
            "mode": policy.mode,
            "ratio": policy.ratio,
        },
    )


def project_score_model(
    model: CSAScoreModel[CandidateT],
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in score model."""
    if type(model) is not CSAScoreModel:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.score-model",
        configuration={
            "biased_potential": (
                None
                if model.biased_potential is None
                else project_biased_potential(
                    model.biased_potential,
                    path=(*path, "biased_potential"),
                    resolver=resolver,
                )
            ),
            "adaptive_potential": (
                None
                if model.adaptive_potential is None
                else project_adaptive_potential(
                    model.adaptive_potential,
                    path=(*path, "adaptive_potential"),
                    resolver=resolver,
                )
            ),
        },
    )


def project_biased_potential(
    potential: CSABiasedPotential,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in biased potential."""
    if type(potential) is not CSABiasedPotential:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.biased-potential",
        configuration={
            "maximum_bias": potential.maximum_bias,
            "sigma": potential.sigma,
            "sigma_reference": potential.sigma_reference,
        },
    )


def project_adaptive_potential(
    potential: CSAAdaptivePotential[CandidateT],
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in adaptive potential."""
    if type(potential) is not CSAAdaptivePotential:
        return resolver.resolve_custom_component(path)
    return builtin_component_node(
        identifier="variopt.csa.adaptive-potential",
        configuration={
            "axes": [
                project_adaptive_potential_axis(
                    axis,
                    path=(*path, "axes", index),
                    resolver=resolver,
                )
                for index, axis in enumerate(potential.axes)
            ],
            "increment": potential.increment,
            "overflow_energy": potential.overflow_energy,
        },
    )


def project_adaptive_potential_axis(
    axis: CSAAdaptivePotentialAxis[CandidateT],
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in adaptive-potential axis."""
    if type(axis) is not CSAAdaptivePotentialAxis:
        return resolver.resolve_custom_component(path)
    reference_candidate = axis.reference_candidate
    if is_space_candidate_value(reference_candidate):
        projected_reference_candidate: JSONValue = {
            "kind": "structured-candidate",
            "value": space_candidate_to_dict(reference_candidate),
        }
    else:
        projected_reference_candidate = resolver.resolve_custom_component(
            (*path, "reference_candidate"),
        )

    return builtin_component_node(
        identifier="variopt.csa.adaptive-potential-axis",
        configuration={
            "reference_candidate": projected_reference_candidate,
            "minimum_distance": axis.minimum_distance,
            "maximum_distance": axis.maximum_distance,
            "bin_count": axis.bin_count,
        },
    )
