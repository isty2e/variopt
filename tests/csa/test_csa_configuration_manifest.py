"""Tests for canonical CSA configuration-manifest values."""

import json
from collections.abc import Mapping
from hashlib import sha256

import pytest

import variopt.algorithms.population.csa.manifest as manifest_package
from variopt.algorithms.population.csa.manifest import (
    CSAComponentDescriptor,
    CSAConfigurationManifest,
)
from variopt.json_types import JSONDict, JSONValue, require_json_mapping


def make_manifest(
    configuration: Mapping[str, JSONValue] | None = None,
) -> CSAConfigurationManifest:
    """Return one manifest fixture with a representative JSON payload."""
    resolved_configuration: Mapping[str, JSONValue] = (
        {
            "bank_capacity": 24,
            "enabled": True,
            "weights": [0.25, -0.0, None],
            "label": "구성",
        }
        if configuration is None
        else configuration
    )
    return CSAConfigurationManifest(
        configuration=resolved_configuration,
    )


class UnsupportedInteger(int):
    """Non-JSON integer subclass used to probe exact boundary types."""


class CSAComponentDescriptorTests:
    """Exercise custom component descriptor invariants."""

    def test_round_trip_preserves_structural_custom_provenance(self) -> None:
        descriptor = CSAComponentDescriptor(
            identifier="org.example.sampler",
            version=2,
            configuration={"strategy": "balanced", "count": 3},
        )

        data = descriptor.to_dict()

        assert data == {
            "kind": "custom",
            "identifier": "org.example.sampler",
            "version": 2,
            "configuration": {
                "strategy": "balanced",
                "count": 3,
            },
        }
        assert CSAComponentDescriptor.from_dict(data) == descriptor

    def test_manifest_subpackage_exports_only_value_artifacts(self) -> None:
        assert frozenset(manifest_package.__all__) == {
            "CSAComponentDescriptor",
            "CSAConfigurationManifest",
        }

    def test_rejects_builtin_provenance_and_reserved_identifiers(self) -> None:
        with pytest.raises(ValueError, match="descriptor.kind must be 'custom'"):
            CSAComponentDescriptor.from_dict(
                {
                    "kind": "builtin",
                    "identifier": "org.example.sampler",
                    "version": 1,
                    "configuration": {},
                },
            )

        for identifier in ("variopt", "variopt.sampler"):
            with pytest.raises(ValueError, match="reserved variopt namespace"):
                CSAComponentDescriptor(
                    identifier=identifier,
                    version=1,
                    configuration={},
                )

    @pytest.mark.parametrize("identifier", ["", " org.example", "org.example "])
    def test_rejects_invalid_custom_identifiers(self, identifier: str) -> None:
        with pytest.raises(ValueError, match="identifier must be non-empty"):
            CSAComponentDescriptor(
                identifier=identifier,
                version=1,
                configuration={},
            )

    @pytest.mark.parametrize("version", [True, 1.0, "1"])
    def test_rejects_non_integer_versions(self, version: JSONValue) -> None:
        with pytest.raises(TypeError, match="descriptor.version must be an exact"):
            CSAComponentDescriptor.from_dict(
                {
                    "kind": "custom",
                    "identifier": "org.example.sampler",
                    "version": version,
                    "configuration": {},
                },
            )

    @pytest.mark.parametrize("version", [0, -1])
    def test_rejects_non_positive_versions(self, version: int) -> None:
        with pytest.raises(ValueError, match="version must be positive"):
            CSAComponentDescriptor(
                identifier="org.example.sampler",
                version=version,
                configuration={},
            )

    def test_rejects_extra_or_missing_fields(self) -> None:
        with pytest.raises(TypeError, match=r"missing \[\"configuration\"\]"):
            CSAComponentDescriptor.from_dict(
                {
                    "kind": "custom",
                    "identifier": "org.example.sampler",
                    "version": 1,
                },
            )

        with pytest.raises(TypeError, match=r"unexpected \[\"extra\"\]"):
            CSAComponentDescriptor.from_dict(
                {
                    "kind": "custom",
                    "identifier": "org.example.sampler",
                    "version": 1,
                    "configuration": {},
                    "extra": None,
                },
            )


class CSAConfigurationManifestCanonicalizationTests:
    """Exercise canonical JSON, equality, and fingerprint semantics."""

    def test_canonical_json_is_compact_sorted_and_utf8(self) -> None:
        manifest = make_manifest(
            {
                "z": 1,
                "a": "한글",
            },
        )

        assert manifest.canonical_json() == (
            '{"algorithm":{"configuration_version":1,'
            '"identifier":"variopt.csa"},'
            '"configuration":{"a":"한글","z":1},'
            '"format":"variopt.csa.configuration-manifest",'
            '"schema_version":1}'
        )

    def test_mapping_order_does_not_affect_identity_or_fingerprint(self) -> None:
        left = make_manifest(
            {
                "outer": {
                    "first": 1,
                    "second": 2,
                },
            },
        )
        right = make_manifest(
            {
                "outer": {
                    "second": 2,
                    "first": 1,
                },
            },
        )

        assert left == right
        assert hash(left) == hash(right)
        assert left.canonical_json() == right.canonical_json()
        assert left.fingerprint == right.fingerprint

    @pytest.mark.parametrize(
        ("left_value", "right_value"),
        [
            (True, 1),
            (1, 1.0),
            (-0.0, 0.0),
            ("é", "e\u0301"),
            ([1, 2], [2, 1]),
        ],
    )
    def test_exact_json_semantics_remain_distinguishable(
        self,
        left_value: JSONValue,
        right_value: JSONValue,
    ) -> None:
        left = make_manifest({"value": left_value})
        right = make_manifest({"value": right_value})

        assert left != right
        assert left.canonical_json() != right.canonical_json()
        assert left.fingerprint != right.fingerprint

    def test_fingerprint_hashes_the_complete_canonical_manifest(self) -> None:
        descriptor = CSAComponentDescriptor(
            identifier="org.example.metric",
            version=3,
            configuration={"scale": 0.5},
        )
        manifest = make_manifest(
            {
                "metric": descriptor.to_dict(),
            },
        )

        expected_digest = sha256(
            manifest.canonical_json().encode("utf-8"),
        ).hexdigest()

        assert manifest.fingerprint == f"sha256:{expected_digest}"
        assert manifest.format_identifier == "variopt.csa.configuration-manifest"
        assert manifest.schema_version == 1
        assert manifest.algorithm_identifier == "variopt.csa"
        assert manifest.to_dict()["algorithm"] == {
            "identifier": "variopt.csa",
            "configuration_version": 1,
        }

    def test_component_version_affects_manifest_identity(self) -> None:
        first_descriptor = CSAComponentDescriptor(
            identifier="org.example.metric",
            version=1,
            configuration={},
        )
        second_descriptor = CSAComponentDescriptor(
            identifier="org.example.metric",
            version=2,
            configuration={},
        )
        algorithm_v1 = make_manifest({"metric": first_descriptor.to_dict()})
        component_v2 = make_manifest({"metric": second_descriptor.to_dict()})

        assert algorithm_v1.fingerprint != component_v2.fingerprint


class CSAConfigurationManifestImmutabilityTests:
    """Exercise deep snapshot and defensive-copy behavior."""

    def test_source_mutation_cannot_change_descriptor_or_manifest(self) -> None:
        nested_values: list[JSONValue] = [1, {"name": "initial"}]
        descriptor_source: JSONDict = {"values": nested_values}
        descriptor = CSAComponentDescriptor(
            identifier="org.example.operator",
            version=1,
            configuration=descriptor_source,
        )
        manifest_source: JSONDict = {"operator": descriptor.to_dict()}
        manifest = make_manifest(manifest_source)
        original_fingerprint = manifest.fingerprint

        nested_values.append(2)
        require_json_mapping(
            nested_values[1],
            field_name="nested_values[1]",
        )["name"] = "changed"
        descriptor_source["extra"] = True
        manifest_source["extra"] = True

        assert descriptor.configuration == {
            "values": [1, {"name": "initial"}],
        }
        assert manifest.configuration == {
            "operator": {
                "kind": "custom",
                "identifier": "org.example.operator",
                "version": 1,
                "configuration": {
                    "values": [1, {"name": "initial"}],
                },
            },
        }
        assert manifest.fingerprint == original_fingerprint

    def test_returned_copies_cannot_change_descriptor_or_manifest(self) -> None:
        descriptor = CSAComponentDescriptor(
            identifier="org.example.operator",
            version=1,
            configuration={"values": [1, 2]},
        )
        manifest = make_manifest({"operator": descriptor.to_dict()})
        original_fingerprint = manifest.fingerprint

        descriptor_copy = descriptor.to_dict()
        descriptor_configuration = require_json_mapping(
            descriptor_copy["configuration"],
            field_name="descriptor.configuration",
        )
        descriptor_configuration["changed"] = True

        manifest_copy = manifest.to_dict()
        manifest_configuration = require_json_mapping(
            manifest_copy["configuration"],
            field_name="manifest.configuration",
        )
        manifest_configuration["changed"] = True

        assert "changed" not in descriptor.configuration
        assert "changed" not in manifest.configuration
        assert manifest.fingerprint == original_fingerprint

    def test_shared_containers_are_snapshotted_by_value_not_identity(self) -> None:
        shared_values: list[JSONValue] = [1, 2]

        aliased = make_manifest(
            {
                "first": shared_values,
                "second": shared_values,
            },
        )
        independent = make_manifest(
            {
                "first": [1, 2],
                "second": [1, 2],
            },
        )

        assert aliased == independent
        assert aliased.fingerprint == independent.fingerprint


class CSAConfigurationManifestValidationTests:
    """Exercise strict parsing and adversarial JSON boundaries."""

    def test_round_trip_rebuilds_equal_immutable_manifest(self) -> None:
        manifest = make_manifest()

        restored = CSAConfigurationManifest.from_dict(manifest.to_dict())

        assert restored == manifest
        assert restored.canonical_json() == manifest.canonical_json()
        assert restored.fingerprint == manifest.fingerprint

    def test_canonical_json_round_trip_preserves_exact_manifest_identity(self) -> None:
        manifest = make_manifest(
            {
                "negative_zero": -0.0,
                "boolean": True,
                "integer": 1,
                "float": 1.0,
                "unicode": "구성",
            },
        )

        restored = CSAConfigurationManifest.from_dict(
            json.loads(manifest.canonical_json()),
        )

        assert restored == manifest
        assert restored.fingerprint == manifest.fingerprint

    @pytest.mark.parametrize("version", [True, 1.0, "1"])
    def test_rejects_non_integer_algorithm_versions(self, version: JSONValue) -> None:
        data = make_manifest().to_dict()
        algorithm = require_json_mapping(
            data["algorithm"],
            field_name="manifest.algorithm",
        )
        algorithm["configuration_version"] = version

        with pytest.raises(
            TypeError,
            match="manifest.algorithm.configuration_version must be an exact",
        ):
            CSAConfigurationManifest.from_dict(data)

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_rejects_non_finite_values_with_field_path(self, value: float) -> None:
        with pytest.raises(
            ValueError,
            match=r'configuration\["nested"\]\[0\] must be finite',
        ):
            make_manifest({"nested": [value]})

    def test_rejects_non_json_scalar_subclasses(self) -> None:
        unsupported_value: JSONValue = UnsupportedInteger(1)

        with pytest.raises(TypeError, match=r'configuration\["value"\]'):
            make_manifest({"value": unsupported_value})

    @pytest.mark.parametrize(
        "configuration",
        [
            {"value": "\ud800"},
            {"\ud800": "value"},
        ],
    )
    def test_rejects_strings_that_are_not_valid_utf8(
        self,
        configuration: Mapping[str, JSONValue],
    ) -> None:
        with pytest.raises(ValueError, match="must be valid UTF-8 text"):
            make_manifest(configuration)

    @pytest.mark.parametrize("configuration", [{1: "invalid"}])
    def test_rejects_non_string_mapping_keys(
        self,
        configuration: Mapping[str, JSONValue],
    ) -> None:
        with pytest.raises(TypeError, match="keys must be exact built-in strings"):
            make_manifest(configuration)

    def test_rejects_container_cycles_with_field_path(self) -> None:
        cyclic: JSONDict = {}
        cyclic["self"] = cyclic

        with pytest.raises(
            ValueError,
            match=r'configuration\["self"\] must not contain a container cycle',
        ):
            make_manifest(cyclic)

    def test_rejects_excessive_nesting(self) -> None:
        nested: JSONValue = 0
        for _ in range(258):
            nested = [nested]

        with pytest.raises(ValueError, match="supported manifest JSON depth"):
            make_manifest({"nested": nested})

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("format", "other.format", "manifest.format"),
            ("schema_version", 2, "manifest.schema_version"),
            (
                "algorithm",
                {"identifier": "other", "configuration_version": 1},
                "algorithm.identifier",
            ),
            (
                "algorithm",
                {"identifier": "variopt.csa", "configuration_version": 2},
                "algorithm.configuration_version",
            ),
        ],
    )
    def test_rejects_unsupported_format_and_versions(
        self,
        field: str,
        value: JSONValue,
        message: str,
    ) -> None:
        data = make_manifest().to_dict()
        data[field] = value

        with pytest.raises(ValueError, match=message):
            CSAConfigurationManifest.from_dict(data)

    @pytest.mark.parametrize("version", [True, 1.0, "1"])
    def test_rejects_non_integer_schema_versions(self, version: JSONValue) -> None:
        data = make_manifest().to_dict()
        data["schema_version"] = version

        with pytest.raises(
            TypeError,
            match="manifest.schema_version must be an exact",
        ):
            CSAConfigurationManifest.from_dict(data)

    def test_rejects_unknown_and_missing_algorithm_fields(self) -> None:
        data = make_manifest().to_dict()
        algorithm = require_json_mapping(
            data["algorithm"],
            field_name="manifest.algorithm",
        )
        algorithm["extra"] = None

        with pytest.raises(TypeError, match=r"unexpected \[\"extra\"\]"):
            CSAConfigurationManifest.from_dict(data)

        missing_data = make_manifest().to_dict()
        missing_algorithm = require_json_mapping(
            missing_data["algorithm"],
            field_name="manifest.algorithm",
        )
        del missing_algorithm["identifier"]

        with pytest.raises(TypeError, match=r"missing \[\"identifier\"\]"):
            CSAConfigurationManifest.from_dict(missing_data)

    def test_rejects_unknown_and_missing_manifest_fields(self) -> None:
        data = make_manifest().to_dict()
        data["extra"] = None

        with pytest.raises(TypeError, match=r"unexpected \[\"extra\"\]"):
            CSAConfigurationManifest.from_dict(data)

        missing = make_manifest().to_dict()
        del missing["configuration"]

        with pytest.raises(TypeError, match=r"missing \[\"configuration\"\]"):
            CSAConfigurationManifest.from_dict(missing)
