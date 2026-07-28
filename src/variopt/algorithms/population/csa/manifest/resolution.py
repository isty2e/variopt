"""Semantic custom-component resolution for CSA configuration manifests."""

from collections.abc import Mapping
from typing import TypeAlias

from .....json_types import JSONDict
from .canonical import validate_utf8_string
from .model import CSAComponentDescriptor, CSAConfigurationResolutionError

CSAComponentPathSegment: TypeAlias = str | int
CSAComponentPath: TypeAlias = tuple[CSAComponentPathSegment, ...]


class CSAComponentDescriptorResolver:
    """Consume caller descriptors at validated semantic component locations."""

    __slots__ = ("_consumed_paths", "_descriptors", "_missing_paths")

    _consumed_paths: set[CSAComponentPath]
    _descriptors: dict[CSAComponentPath, CSAComponentDescriptor]
    _missing_paths: set[CSAComponentPath]

    def __init__(
        self,
        descriptors: Mapping[CSAComponentPath, CSAComponentDescriptor] | None,
    ) -> None:
        descriptor_items = () if descriptors is None else descriptors.items()
        normalized_descriptors: dict[CSAComponentPath, CSAComponentDescriptor] = {}
        for path, descriptor in descriptor_items:
            validate_component_path(path)
            if type(descriptor) is not CSAComponentDescriptor:
                msg = "custom component descriptors must use CSAComponentDescriptor"
                raise TypeError(msg)
            normalized_descriptors[path] = descriptor

        self._descriptors = normalized_descriptors
        self._consumed_paths = set()
        self._missing_paths = set()

    def resolve_custom_component(
        self,
        path: CSAComponentPath,
    ) -> JSONDict | None:
        """Return one custom descriptor payload or record its absence."""
        descriptor = self._descriptors.get(path)
        if descriptor is None:
            self._missing_paths.add(path)
            return None

        self._consumed_paths.add(path)
        return descriptor.to_dict()

    def require_complete(self) -> None:
        """Raise when any required or supplied semantic location is unresolved."""
        unused_paths = set(self._descriptors).difference(self._consumed_paths)
        if not self._missing_paths and not unused_paths:
            return

        raise CSAConfigurationResolutionError(
            missing_component_paths=tuple(self._missing_paths),
            unused_component_paths=tuple(unused_paths),
        )


def validate_component_path(path: CSAComponentPath) -> None:
    """Validate one caller-supplied semantic component path."""
    if type(path) is not tuple:
        msg = "component paths must be exact built-in tuples"
        raise TypeError(msg)
    if len(path) == 0:
        msg = "component paths must not be empty"
        raise ValueError(msg)

    for segment in path:
        if type(segment) is str:
            if segment == "":
                msg = "component path string segments must not be empty"
                raise ValueError(msg)
            validate_utf8_string(segment, field_name="component path string segment")
            continue
        if type(segment) is int:
            if segment < 0:
                msg = "component path integer segments must be non-negative"
                raise ValueError(msg)
            continue
        msg = "component path segments must be exact strings or integers"
        raise TypeError(msg)
