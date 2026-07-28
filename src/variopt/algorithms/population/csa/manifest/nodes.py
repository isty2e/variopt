"""Structural built-in nodes for CSA configuration manifests."""

from collections.abc import Mapping

from .....json_types import JSONDict, JSONValue

CSA_BUILTIN_COMPONENT_VERSION = 1


def builtin_component_node(
    *,
    identifier: str,
    configuration: Mapping[str, JSONValue],
) -> JSONDict:
    """Return one structurally built-in manifest component node."""
    return {
        "kind": "builtin",
        "identifier": identifier,
        "version": CSA_BUILTIN_COMPONENT_VERSION,
        "configuration": dict(configuration),
    }
