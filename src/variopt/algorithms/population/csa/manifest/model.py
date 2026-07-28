"""Canonical CSA configuration-manifest value objects."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256

from typing_extensions import Self

from .....json_types import JSONDict, JSONValue
from .canonical import (
    FrozenJSONObject,
    freeze_json_object,
    thaw_json_object,
    validate_utf8_string,
)

CSA_CONFIGURATION_MANIFEST_FORMAT = "variopt.csa.configuration-manifest"
CSA_CONFIGURATION_MANIFEST_SCHEMA_VERSION = 1
CSA_ALGORITHM_IDENTIFIER = "variopt.csa"
CSA_ALGORITHM_CONFIGURATION_VERSION = 1
CSA_CUSTOM_COMPONENT_KIND = "custom"
VARIOPT_COMPONENT_IDENTIFIER = "variopt"
VARIOPT_COMPONENT_IDENTIFIER_PREFIX = "variopt."


class CSAConfigurationResolutionError(ValueError):
    """Report unresolved semantic locations in one manifest projection.

    Parameters
    ----------
    missing_component_paths : tuple[tuple[str | int, ...], ...]
        Deterministically ordered locations that require caller descriptors.
    unused_component_paths : tuple[tuple[str | int, ...], ...]
        Deterministically ordered caller locations that were not consumed.
    """

    missing_component_paths: tuple[tuple[str | int, ...], ...]
    unused_component_paths: tuple[tuple[str | int, ...], ...]

    def __init__(
        self,
        *,
        missing_component_paths: tuple[tuple[str | int, ...], ...],
        unused_component_paths: tuple[tuple[str | int, ...], ...],
    ) -> None:
        canonical_missing_paths = _canonical_component_paths(
            missing_component_paths,
        )
        canonical_unused_paths = _canonical_component_paths(
            unused_component_paths,
        )
        self.missing_component_paths = canonical_missing_paths
        self.unused_component_paths = canonical_unused_paths

        details: list[str] = []
        if canonical_missing_paths:
            details.append(
                "missing component paths "
                + _component_paths_json(canonical_missing_paths),
            )
        if canonical_unused_paths:
            details.append(
                "unused component paths "
                + _component_paths_json(canonical_unused_paths),
            )
        super().__init__(
            "CSA configuration manifest resolution failed: " + "; ".join(details),
        )


@dataclass(frozen=True, slots=True, init=False)
class CSAComponentDescriptor:
    """Stable caller-owned description of one custom CSA component.

    A descriptor is a provenance assertion supplied by the caller. Variopt
    preserves and fingerprints the assertion but cannot verify that matching
    descriptors imply equivalent executable behavior.

    Parameters
    ----------
    identifier : str
        Stable component identifier outside the reserved ``variopt`` namespace.
    version : int
        Positive version for the identifier and configuration semantics.
    configuration : Mapping[str, JSONValue]
        Exact JSON configuration asserted by the caller.

    Raises
    ------
    TypeError
        If a field has an invalid type or the configuration is not JSON-safe.
    ValueError
        If the identifier or version is invalid, or the configuration contains
        non-finite floats, cycles, or excessive nesting.
    """

    identifier: str
    version: int
    _configuration: FrozenJSONObject = field(repr=False)

    def __init__(
        self,
        *,
        identifier: str,
        version: int,
        configuration: Mapping[str, JSONValue],
    ) -> None:
        _validate_custom_identifier(identifier)
        _validate_positive_version(version, field_name="version")
        frozen_configuration = freeze_json_object(
            configuration,
            field_name="configuration",
        )

        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "_configuration", frozen_configuration)

    @property
    def configuration(self) -> JSONDict:
        """Return a fresh JSON-safe copy of the asserted configuration."""
        return thaw_json_object(self._configuration)

    def to_dict(self) -> JSONDict:
        """Return the canonical JSON-safe descriptor representation.

        Returns
        -------
        JSONDict
            Fresh descriptor data with structural custom provenance.
        """
        return {
            "kind": CSA_CUSTOM_COMPONENT_KIND,
            "identifier": self.identifier,
            "version": self.version,
            "configuration": self.configuration,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, JSONValue]) -> Self:
        """Build a custom component descriptor from strict JSON data.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            Descriptor data produced by :meth:`to_dict`.

        Returns
        -------
        Self
            Immutable custom component descriptor.

        Raises
        ------
        TypeError
            If required fields are absent, extra fields are present, or field
            types are invalid.
        ValueError
            If provenance, identifier, version, or configuration values are
            invalid.
        """
        _require_exact_fields(
            data,
            expected=frozenset(
                {
                    "kind",
                    "identifier",
                    "version",
                    "configuration",
                },
            ),
            field_name="descriptor",
        )
        kind = _require_json_string(data["kind"], field_name="descriptor.kind")
        if kind != CSA_CUSTOM_COMPONENT_KIND:
            msg = "descriptor.kind must be 'custom'"
            raise ValueError(msg)

        identifier = _require_json_string(
            data["identifier"],
            field_name="descriptor.identifier",
        )
        version = _require_json_integer(
            data["version"],
            field_name="descriptor.version",
        )
        configuration = _require_json_object(
            data["configuration"],
            field_name="descriptor.configuration",
        )
        return cls(
            identifier=identifier,
            version=version,
            configuration=configuration,
        )


@dataclass(frozen=True, slots=True, init=False)
class CSAConfigurationManifest:
    """Immutable versioned description of resolved CSA configuration.

    The manifest identifies only the represented optimizer-side configuration.
    It is not an executable optimizer, runtime checkpoint, or complete experiment
    identity.

    Parameters
    ----------
    configuration : Mapping[str, JSONValue]
        Resolved optimizer configuration encoded as exact JSON data.

    Raises
    ------
    TypeError
        If a field has an invalid type or the configuration is not JSON-safe.
    ValueError
        If the configuration contains non-finite floats, cycles, or excessive
        nesting.
    """

    _configuration: FrozenJSONObject = field(repr=False)

    def __init__(
        self,
        *,
        configuration: Mapping[str, JSONValue],
    ) -> None:
        frozen_configuration = freeze_json_object(
            configuration,
            field_name="configuration",
        )

        object.__setattr__(self, "_configuration", frozen_configuration)

    @property
    def format_identifier(self) -> str:
        """Return the stable manifest format identifier."""
        return CSA_CONFIGURATION_MANIFEST_FORMAT

    @property
    def schema_version(self) -> int:
        """Return the supported manifest wire-schema version."""
        return CSA_CONFIGURATION_MANIFEST_SCHEMA_VERSION

    @property
    def algorithm_identifier(self) -> str:
        """Return the stable CSA algorithm identifier."""
        return CSA_ALGORITHM_IDENTIFIER

    @property
    def algorithm_configuration_version(self) -> int:
        """Return the supported CSA configuration-semantics version."""
        return CSA_ALGORITHM_CONFIGURATION_VERSION

    @property
    def configuration(self) -> JSONDict:
        """Return a fresh JSON-safe copy of the resolved configuration."""
        return thaw_json_object(self._configuration)

    @property
    def fingerprint(self) -> str:
        """Return the version-scoped SHA-256 content fingerprint."""
        digest = sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def canonical_json(self) -> str:
        """Return deterministic compact JSON text for the manifest.

        Returns
        -------
        str
            UTF-8-compatible JSON text with sorted object keys and compact
            separators.
        """
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_dict(self) -> JSONDict:
        """Return a fresh JSON-safe manifest representation.

        Returns
        -------
        JSONDict
            Manifest data including all format and semantic version axes.
        """
        return {
            "format": self.format_identifier,
            "schema_version": self.schema_version,
            "algorithm": {
                "identifier": self.algorithm_identifier,
                "configuration_version": self.algorithm_configuration_version,
            },
            "configuration": self.configuration,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, JSONValue]) -> Self:
        """Build a manifest from strict versioned JSON data.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            Manifest data produced by :meth:`to_dict`.

        Returns
        -------
        Self
            Immutable canonical manifest.

        Raises
        ------
        TypeError
            If required fields are absent, extra fields are present, or field
            types are invalid.
        ValueError
            If a format identifier, version, or configuration value is invalid.
        """
        _require_exact_fields(
            data,
            expected=frozenset(
                {
                    "format",
                    "schema_version",
                    "algorithm",
                    "configuration",
                },
            ),
            field_name="manifest",
        )
        format_identifier = _require_json_string(
            data["format"],
            field_name="manifest.format",
        )
        if format_identifier != CSA_CONFIGURATION_MANIFEST_FORMAT:
            msg = f"manifest.format must be '{CSA_CONFIGURATION_MANIFEST_FORMAT}'"
            raise ValueError(msg)

        schema_version = _require_json_integer(
            data["schema_version"],
            field_name="manifest.schema_version",
        )
        if schema_version != CSA_CONFIGURATION_MANIFEST_SCHEMA_VERSION:
            msg = (
                "manifest.schema_version must be "
                f"{CSA_CONFIGURATION_MANIFEST_SCHEMA_VERSION}"
            )
            raise ValueError(msg)

        algorithm = _require_json_object(
            data["algorithm"],
            field_name="manifest.algorithm",
        )
        _require_exact_fields(
            algorithm,
            expected=frozenset({"identifier", "configuration_version"}),
            field_name="manifest.algorithm",
        )
        algorithm_identifier = _require_json_string(
            algorithm["identifier"],
            field_name="manifest.algorithm.identifier",
        )
        if algorithm_identifier != CSA_ALGORITHM_IDENTIFIER:
            msg = f"manifest.algorithm.identifier must be '{CSA_ALGORITHM_IDENTIFIER}'"
            raise ValueError(msg)

        algorithm_configuration_version = _require_json_integer(
            algorithm["configuration_version"],
            field_name="manifest.algorithm.configuration_version",
        )
        if algorithm_configuration_version != CSA_ALGORITHM_CONFIGURATION_VERSION:
            msg = (
                "manifest.algorithm.configuration_version must be "
                f"{CSA_ALGORITHM_CONFIGURATION_VERSION}"
            )
            raise ValueError(msg)
        configuration = _require_json_object(
            data["configuration"],
            field_name="manifest.configuration",
        )
        return cls(configuration=configuration)


def _validate_custom_identifier(identifier: str) -> None:
    if type(identifier) is not str:
        msg = "identifier must be an exact built-in string"
        raise TypeError(msg)
    if identifier == "" or identifier.strip() != identifier:
        msg = "identifier must be non-empty without surrounding whitespace"
        raise ValueError(msg)
    validate_utf8_string(identifier, field_name="identifier")
    if identifier == VARIOPT_COMPONENT_IDENTIFIER or identifier.startswith(
        VARIOPT_COMPONENT_IDENTIFIER_PREFIX
    ):
        msg = "identifier must not use the reserved variopt namespace"
        raise ValueError(msg)


def _validate_positive_version(version: int, *, field_name: str) -> None:
    if type(version) is not int:
        msg = f"{field_name} must be an exact built-in integer"
        raise TypeError(msg)
    if version <= 0:
        msg = f"{field_name} must be positive"
        raise ValueError(msg)


def _component_paths_json(
    paths: tuple[tuple[str | int, ...], ...],
) -> str:
    return json.dumps(
        paths,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _canonical_component_paths(
    paths: tuple[tuple[str | int, ...], ...],
) -> tuple[tuple[str | int, ...], ...]:
    return tuple(sorted(set(paths), key=_component_path_sort_key))


def _component_path_sort_key(
    path: tuple[str | int, ...],
) -> tuple[tuple[int, str, int], ...]:
    key_segments: list[tuple[int, str, int]] = []
    for segment in path:
        if type(segment) is str:
            key_segments.append((0, segment, 0))
        elif type(segment) is int:
            key_segments.append((1, "", segment))
        else:
            msg = "component path segments must be exact strings or integers"
            raise TypeError(msg)
    return tuple(key_segments)


def _require_exact_fields(
    data: Mapping[str, JSONValue],
    *,
    expected: frozenset[str],
    field_name: str,
) -> None:
    for key in data:
        if type(key) is not str:
            msg = f"{field_name} keys must be exact built-in strings"
            raise TypeError(msg)

    actual = frozenset(data)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(
                "missing " + json.dumps(missing, ensure_ascii=False),
            )
        if unexpected:
            details.append(
                "unexpected " + json.dumps(unexpected, ensure_ascii=False),
            )
        msg = f"{field_name} fields are invalid: {'; '.join(details)}"
        raise TypeError(msg)


def _require_json_string(value: JSONValue, *, field_name: str) -> str:
    if type(value) is not str:
        msg = f"{field_name} must be an exact built-in JSON string"
        raise TypeError(msg)
    return value


def _require_json_integer(value: JSONValue, *, field_name: str) -> int:
    if type(value) is not int:
        msg = f"{field_name} must be an exact built-in JSON integer"
        raise TypeError(msg)
    return value


def _require_json_object(value: JSONValue, *, field_name: str) -> JSONDict:
    if type(value) is not dict:
        msg = f"{field_name} must be an exact built-in JSON object"
        raise TypeError(msg)
    return value
