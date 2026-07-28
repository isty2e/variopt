"""Exact built-in component projection for CSA configuration manifests."""

from typing import TypeVar

from .....diversity import DiversityMetric, StructuredSpaceDiversityMetric
from .....json_types import JSONValue
from .....operators import VariationOperator
from .....sampling import CandidateSampler, SearchSpaceSampler
from .....spaces import SearchSpace
from ...permutation.operators import InversionMutation, OrderCrossover, SwapMutation
from ..operators import (
    BoundedMutation,
    DifferentialEvolutionVariation,
    MixtureVariation,
    RandomResetMutation,
    UniformCrossover,
)
from .nodes import builtin_component_node
from .resolution import CSAComponentDescriptorResolver, CSAComponentPath
from .spaces import project_space

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT")


def project_sampler(
    sampler: CandidateSampler[CandidateT] | None,
    *,
    optimizer_space: SearchSpace[BoundaryT, CandidateT],
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project the effective CSA boundary sampler."""
    if sampler is None:
        return builtin_component_node(
            identifier="variopt.sampler.search-space",
            configuration={
                "space": project_space(
                    optimizer_space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
            },
        )

    if isinstance(sampler, SearchSpaceSampler):
        if type(sampler) is not SearchSpaceSampler:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.sampler.search-space",
            configuration={
                "space": project_space(
                    sampler.space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
            },
        )

    return resolver.resolve_custom_component(path)


def project_diversity_metric(
    diversity_metric: DiversityMetric[CandidateT],
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in diversity metric."""
    if isinstance(diversity_metric, StructuredSpaceDiversityMetric):
        if type(diversity_metric) is not StructuredSpaceDiversityMetric:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.diversity.structured-space",
            configuration={
                "space": project_space(
                    diversity_metric.space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
            },
        )

    return resolver.resolve_custom_component(path)


def project_variation_operator(
    operator: VariationOperator[CandidateT],
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in variation operator recursively."""
    if isinstance(operator, UniformCrossover):
        if type(operator) is not UniformCrossover:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.operator.uniform-crossover",
            configuration={
                "space": project_space(
                    operator.space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
                "max_exchange_fraction": operator.max_exchange_fraction,
            },
        )

    if isinstance(operator, RandomResetMutation):
        if type(operator) is not RandomResetMutation:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.operator.random-reset-mutation",
            configuration={
                "space": project_space(
                    operator.space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
                "max_exchange_fraction": operator.max_exchange_fraction,
            },
        )

    if isinstance(operator, BoundedMutation):
        if type(operator) is not BoundedMutation:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.operator.bounded-mutation",
            configuration={
                "space": project_space(
                    operator.space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
                "max_perturbation_fraction": operator.max_perturbation_fraction,
            },
        )

    if isinstance(operator, DifferentialEvolutionVariation):
        if type(operator) is not DifferentialEvolutionVariation:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.operator.differential-evolution",
            configuration={
                "space": project_space(
                    operator.space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
                "mutation_range": list(operator.mutation_range),
                "recombination_probability": operator.recombination_probability,
                "n_cross": operator.n_cross,
            },
        )

    if isinstance(operator, MixtureVariation):
        if type(operator) is not MixtureVariation:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.operator.mixture",
            configuration={
                "operators": [
                    project_variation_operator(
                        child_operator,
                        path=(*path, "operators", index),
                        resolver=resolver,
                    )
                    for index, child_operator in enumerate(operator.operators)
                ],
                "weights": list(operator.weights),
            },
        )

    if isinstance(operator, OrderCrossover):
        if type(operator) is not OrderCrossover:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.operator.order-crossover",
            configuration={
                "space": project_space(
                    operator.space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
                "max_segment_fraction": operator.max_segment_fraction,
            },
        )

    if isinstance(operator, SwapMutation):
        if type(operator) is not SwapMutation:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.operator.swap-mutation",
            configuration={
                "space": project_space(
                    operator.space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
                "max_swap_fraction": operator.max_swap_fraction,
            },
        )

    if isinstance(operator, InversionMutation):
        if type(operator) is not InversionMutation:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.operator.inversion-mutation",
            configuration={
                "space": project_space(
                    operator.space,
                    path=(*path, "space"),
                    resolver=resolver,
                ),
                "max_inversion_fraction": operator.max_inversion_fraction,
            },
        )

    return resolver.resolve_custom_component(path)
