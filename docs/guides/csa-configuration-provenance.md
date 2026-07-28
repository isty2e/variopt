# Record CSA Configuration Provenance

A CSA checkpoint records optimizer state. It does not record the configuration
that gives that state meaning. Use
[`CSAOptimizer.configuration_manifest()`][variopt.algorithms.population.CSAOptimizer.configuration_manifest]
to capture the fully resolved optimizer-side configuration separately.

The resulting
[`CSAConfigurationManifest`][variopt.algorithms.population.CSAConfigurationManifest]
is useful for:

- recording the configuration used for a run
- contributing a version-scoped optimizer-configuration component to a cache key
- rejecting a checkpoint restore under a different CSA configuration
- comparing resolved presets and overrides rather than raw constructor inputs

It is not an optimizer snapshot, executable reconstruction recipe, or complete
experiment identity.

## Create a Built-In Manifest

Exact built-in spaces, samplers, metrics, policies, and operators need no extra
metadata:

```python
from variopt import IntegerSpace
from variopt.algorithms.population import CSAOptimizer

space = IntegerSpace(-20, 20)
optimizer = CSAOptimizer.from_space_defaults(
    space=space,
    bank_capacity=8,
    random_state=11,
)

manifest = optimizer.configuration_manifest()

print(manifest.fingerprint)
print(manifest.canonical_json())
```

The manifest represents the effective `resolved_profile`, not merely the
boundary-level `CSAProfile` passed to the constructor. Two constructor forms
that resolve to the same exact built-in configuration produce the same
canonical JSON and fingerprint.

`random_state=11` records the initialization seed. `random_state=None` records
an explicit nondeterministic initialization mode. Manifest generation never
materializes or advances an RNG and never includes the current RNG state.

## Persist and Parse the Manifest

`canonical_json()` returns deterministic compact JSON under the Variopt
manifest contract:

```python
import json
from pathlib import Path

from variopt.algorithms.population import CSAConfigurationManifest

manifest_path = Path("csa-configuration.json")
manifest_path.write_text(
    manifest.canonical_json() + "\n",
    encoding="utf-8",
)

stored_manifest = CSAConfigurationManifest.from_dict(
    json.loads(manifest_path.read_text(encoding="utf-8")),
)

assert stored_manifest.fingerprint == manifest.fingerprint
```

`to_dict()` and `from_dict()` provide the structured equivalent.
`from_dict()` validates the manifest format, schema version, algorithm identity,
algorithm-configuration version, and JSON value contract. It does not
reconstruct an optimizer or executable component.

The canonical JSON representation is Variopt's versioned contract. It uses
sorted object keys, compact separators, finite JSON numbers, and UTF-8 text, but
it is not an implementation of RFC 8785 or a promise that arbitrary JSON
libraries in other languages will produce identical bytes.

## Describe Custom Components

Variopt can project exact built-ins because it owns their semantic contracts.
For a custom component or a subclass of a built-in component, the caller must
provide a
[`CSAComponentDescriptor`][variopt.algorithms.population.CSAComponentDescriptor]
at every semantic occurrence.

The following optimizer has a custom sampler and diversity metric:

```python
import numpy as np
from typing_extensions import override

from variopt import IntegerSpace
from variopt.algorithms.population import (
    CSAComponentDescriptor,
    CSAConfigurationResolutionError,
    CSAOptimizer,
)
from variopt.diversity import DiversityMetric
from variopt.sampling import CandidateSampler


class CenterBiasedSampler(CandidateSampler[int]):
    @override
    def sample(self, random_state: np.random.RandomState) -> int:
        return int(random_state.randint(-5, 6))


class AbsoluteDistance(DiversityMetric[int]):
    @override
    def distance(self, left: int, right: int) -> float:
        return float(abs(left - right))


space = IntegerSpace(-20, 20)
optimizer = CSAOptimizer.from_space_defaults(
    space=space,
    bank_capacity=8,
    sampler=CenterBiasedSampler(),
    diversity_metric=AbsoluteDistance(),
    random_state=11,
)

try:
    optimizer.configuration_manifest(
        custom_component_descriptors={
            ("obsolete_component",): CSAComponentDescriptor(
                identifier="org.example.obsolete",
                version=1,
                configuration={},
            ),
        },
    )
except CSAConfigurationResolutionError as error:
    print(error.missing_component_paths)
    print(error.unused_component_paths)
```

The exception reports all missing and unused locations together. No partial
manifest is returned. Supply descriptors for the two actual custom occurrences:

```python
component_descriptors = {
    ("sampler",): CSAComponentDescriptor(
        identifier="org.example.center-biased-sampler",
        version=1,
        configuration={
            "minimum_sample": -5,
            "maximum_sample": 5,
        },
    ),
    ("diversity_metric",): CSAComponentDescriptor(
        identifier="org.example.absolute-distance",
        version=1,
        configuration={},
    ),
}

custom_manifest = optimizer.configuration_manifest(
    custom_component_descriptors=component_descriptors,
)
```

A descriptor is a caller assertion. Variopt validates and fingerprints its
identifier, version, and JSON configuration, but it cannot verify that two
implementations with the same descriptor behave identically. Include every
execution-relevant custom setting and bump the descriptor version when its
semantics change.

Custom identifiers must be stable, non-empty UTF-8 strings outside the reserved
`variopt` namespace. Descriptor configuration must be finite, acyclic,
JSON-safe data.

## Understand Semantic Locations

Descriptor keys are tuples of exact `str` and non-negative `int` segments. They
identify locations in resolved CSA configuration, not Python attribute paths,
module paths, object identities, or incidental serialized-dictionary keys.

Common locations include:

| Component occurrence | Semantic location |
| --- | --- |
| Optimizer search space | `("space",)` |
| Sampler | `("sampler",)` |
| Sampler-owned space | `("sampler", "space")` |
| Diversity metric | `("diversity_metric",)` |
| Metric-owned space | `("diversity_metric", "space")` |
| Resolved cutoff schedule | `("resolved_profile", "cutoff_schedule")` |
| Resolved update policy | `("resolved_profile", "update_policy")` |
| Niche policy inside the update policy | `("resolved_profile", "update_policy", "niche_quality_policy")` |
| First regular-family operator | `("resolved_profile", "perturbation_schedule", "regular_family", 0, "operator")` |
| Second operator in a mixture | `(..., "operator", "operators", 1)` |
| First adaptive-potential axis | `("resolved_profile", "score_model", "adaptive_potential", "axes", 0)` |
| Opaque reference candidate on that axis | `(..., "axes", 0, "reference_candidate")` |

Nested exact built-in spaces extend their owning location as follows:

| Space structure | Child suffix |
| --- | --- |
| Array element space | `("element_space",)` |
| Tuple child | `("child_spaces", child_index)` |
| Record field space | `("fields", field_index, "space")` |
| Space-bound operator | `("space",)` |

Record fields use their ordered index rather than the field name in semantic
locations. The field name remains part of the represented record-space value.

An exact built-in parent is traversed recursively. A custom parent consumes one
descriptor at its own location and its internals are not inspected. Supplying a
descriptor for one of that custom parent's hypothetical descendants is therefore
an unused-descriptor error. Descriptors supplied for exact built-ins or unknown
locations are also reported as unused.

Component-local spaces are represented by value at each occurrence. Reusing one
space object and constructing equivalent independent space objects therefore
produce the same manifest; Python aliasing is not semantic identity.

Semantic locations are deterministic within an algorithm-configuration version.
If a future version changes the represented configuration ontology, it must
change the corresponding version axis rather than silently reinterpreting old
locations.

## Interpret the Fingerprint

The fingerprint is SHA-256 over the complete canonical manifest, including:

- manifest format and schema version
- algorithm identifier and algorithm-configuration version
- exact built-in component identifiers, versions, and configuration
- custom descriptor identifiers, versions, and asserted configuration
- resolved optimizer configuration, including the initialization seed mode

Matching fingerprints mean that the represented manifest data are identical
under those version axes. They do not prove:

- behavioral equivalence of custom executable code
- equality of objectives, problems, datasets, or data splits
- equality of evaluators, kernels, execution models, or worker topology
- equality of dependency versions, platform, environment, or hardware
- equality of runtime optimizer state or in-flight work
- full reproducibility of an optimization run

For full experiment provenance, store the manifest alongside caller-owned
problem/data/objective identity, evaluator and execution settings, environment
and dependency metadata, and the checkpoint or terminal result as applicable.

## Guard a Checkpoint Restore

Persist the configuration manifest beside the JSON-safe CSA checkpoint. Given
the `optimizer`, `manifest_path`, and a separately loaded `checkpoint_data`:

```python
import json

from variopt.algorithms.population import CSAConfigurationManifest

stored_manifest = CSAConfigurationManifest.from_dict(
    json.loads(manifest_path.read_text(encoding="utf-8")),
)
current_manifest = optimizer.configuration_manifest()

if current_manifest.fingerprint != stored_manifest.fingerprint:
    raise RuntimeError("CSA configuration does not match the checkpoint")

restored_state = optimizer.state_from_dict(checkpoint_data)
```

For custom components, pass the same descriptor mapping when creating
`current_manifest`. This guard prevents restoring under a different represented
CSA configuration. It does not compare the objective, data, evaluator,
environment, or dependency provenance; validate those dimensions separately
before claiming exact continuation.

See [Checkpointing](../reference/checkpointing.md) for safe-boundary state
serialization and the remaining runtime-state exclusions.

## Version Axes

The manifest separates several reasons for incompatibility:

- `schema_version` changes when the manifest wire shape or parsing contract
  changes.
- `algorithm.configuration_version` changes when represented CSA configuration
  semantics or semantic locations change.
- built-in component versions change when a represented built-in component's
  configuration semantics change independently.
- custom component versions are caller-owned and must change when the caller's
  asserted semantics change.

All version axes participate in the fingerprint. An unsupported manifest or
algorithm-configuration version is rejected by `from_dict()` rather than being
silently interpreted under current semantics.
