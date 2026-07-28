"""Recursive exact-type projection of CSA search spaces."""

from typing import TypeVar

from .....json_types import JSONDict, JSONValue
from .....spaces import (
    ArraySpace,
    CategoricalSpace,
    IntegerSpace,
    PermutationSpace,
    RealSpace,
    RecordSpace,
    SearchSpace,
    SpaceScalarValue,
    TupleSpace,
)
from .....spaces.composites.adapters import CompositeChildSpace
from .nodes import builtin_component_node
from .resolution import CSAComponentDescriptorResolver, CSAComponentPath

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT")


def project_space(
    space: SearchSpace[BoundaryT, CandidateT] | CompositeChildSpace,
    *,
    path: CSAComponentPath,
    resolver: CSAComponentDescriptorResolver,
) -> JSONValue:
    """Project one exact built-in space or consume its custom descriptor."""
    if isinstance(space, RealSpace):
        if type(space) is not RealSpace:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.space.real",
            configuration={
                "low": space.low,
                "high": space.high,
                "scale": space.scale,
            },
        )

    if isinstance(space, IntegerSpace):
        if type(space) is not IntegerSpace:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.space.integer",
            configuration={
                "low": space.low,
                "high": space.high,
                "scale": space.scale,
            },
        )

    if isinstance(space, CategoricalSpace):
        if type(space) is not CategoricalSpace:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.space.categorical",
            configuration={
                "choices": [project_space_scalar(choice) for choice in space.choices],
            },
        )

    if isinstance(space, PermutationSpace):
        if type(space) is not PermutationSpace:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.space.permutation",
            configuration={"size": space.size},
        )

    if isinstance(space, ArraySpace):
        if type(space) is not ArraySpace:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.space.array",
            configuration={
                "element_space": project_space(
                    space.element_space,
                    path=(*path, "element_space"),
                    resolver=resolver,
                ),
                "length": space.length,
            },
        )

    if isinstance(space, TupleSpace):
        if type(space) is not TupleSpace:
            return resolver.resolve_custom_component(path)
        return builtin_component_node(
            identifier="variopt.space.tuple",
            configuration={
                "child_spaces": [
                    project_space(
                        child_space,
                        path=(*path, "child_spaces", index),
                        resolver=resolver,
                    )
                    for index, child_space in enumerate(space.child_spaces)
                ],
            },
        )

    if isinstance(space, RecordSpace):
        if type(space) is not RecordSpace:
            return resolver.resolve_custom_component(path)
        fields: list[JSONValue] = []
        for index, (name, child_space) in enumerate(space.fields):
            fields.append(
                {
                    "name": name,
                    "space": project_space(
                        child_space,
                        path=(*path, "fields", index, "space"),
                        resolver=resolver,
                    ),
                },
            )
        return builtin_component_node(
            identifier="variopt.space.record",
            configuration={"fields": fields},
        )

    return resolver.resolve_custom_component(path)


def project_space_scalar(value: SpaceScalarValue) -> JSONDict:
    """Project one categorical scalar with exact runtime-type identity."""
    if type(value) is bool:
        return {"type": "boolean", "value": value}
    if type(value) is int:
        return {"type": "integer", "value": value}
    if type(value) is float:
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "string", "value": value}
    if type(value) is bytes:
        return {"type": "bytes", "hex": value.hex()}
    if type(value) is bytearray:
        return {"type": "bytearray", "hex": value.hex()}

    msg = "categorical choices must use canonical scalar types"
    raise TypeError(msg)
