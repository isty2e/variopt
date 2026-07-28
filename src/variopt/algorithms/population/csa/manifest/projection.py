"""Root projection from resolved CSA optimizer facts into a manifest."""

from collections.abc import Mapping
from typing import TypeVar

from .....diversity import DiversityMetric
from .....json_types import JSONDict
from .....randomness import RandomSeed
from .....sampling import CandidateSampler
from .....spaces import SearchSpace
from ..profile import CSAResolvedProfile
from .components import project_diversity_metric, project_sampler
from .model import CSAComponentDescriptor, CSAConfigurationManifest
from .profile import project_resolved_profile
from .resolution import CSAComponentDescriptorResolver, CSAComponentPath
from .spaces import project_space

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT")


def project_csa_configuration(
    *,
    space: SearchSpace[BoundaryT, CandidateT],
    diversity_metric: DiversityMetric[CandidateT],
    bank_capacity: int,
    resolved_profile: CSAResolvedProfile[CandidateT],
    sampler: CandidateSampler[CandidateT] | None,
    random_state: RandomSeed,
    custom_component_descriptors: (
        Mapping[CSAComponentPath, CSAComponentDescriptor] | None
    ),
) -> CSAConfigurationManifest:
    """Project canonical optimizer-side CSA configuration into a manifest."""
    resolver = CSAComponentDescriptorResolver(custom_component_descriptors)
    configuration: JSONDict = {
        "bank_capacity": bank_capacity,
        "space": project_space(
            space,
            path=("space",),
            resolver=resolver,
        ),
        "sampler": project_sampler(
            sampler,
            optimizer_space=space,
            path=("sampler",),
            resolver=resolver,
        ),
        "diversity_metric": project_diversity_metric(
            diversity_metric,
            path=("diversity_metric",),
            resolver=resolver,
        ),
        "random_initialization": project_random_initialization(random_state),
        "resolved_profile": project_resolved_profile(
            resolved_profile,
            path=("resolved_profile",),
            resolver=resolver,
        ),
    }
    resolver.require_complete()
    return CSAConfigurationManifest(configuration=configuration)


def project_random_initialization(random_state: RandomSeed) -> JSONDict:
    """Project a public random seed without materializing runtime RNG state."""
    if random_state is None:
        return {"mode": "nondeterministic"}
    if type(random_state) is not int:
        msg = "random_state must be an int or None"
        raise TypeError(msg)
    if random_state < 0 or random_state >= 2**32:
        msg = "random_state must lie in the NumPy uint32 seed range"
        raise ValueError(msg)
    return {
        "mode": "seeded",
        "seed": random_state,
    }
