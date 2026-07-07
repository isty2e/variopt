"""Regression tests for CSA proposal-state export boundaries."""

from variopt.algorithms.population.csa.generation.proposal.policy import (
    CSAProposalPolicy,
)
from variopt.algorithms.population.csa.generation.proposal.state import (
    CSAProposalState,
    PlannedProposalAttribution,
    ProposalFamilyStat,
)
from variopt.algorithms.population.csa.generation.proposal.state.aggregate import (
    CSAProposalState as CSAProposalStateDirect,
)
from variopt.algorithms.population.csa.generation.proposal.state.attribution import (
    PlannedProposalAttribution as PlannedProposalAttributionDirect,
)
from variopt.algorithms.population.csa.generation.proposal.state.stats import (
    ProposalFamilyStat as ProposalFamilyStatDirect,
)


class CSAProposalStateExportTests:
    """Lock the canonical and semantic CSA proposal-state import paths."""

    def test_facade_re_exports_canonical_state_symbols(self) -> None:
        assert CSAProposalState is CSAProposalStateDirect
        assert PlannedProposalAttribution is PlannedProposalAttributionDirect
        assert ProposalFamilyStat is ProposalFamilyStatDirect

    def test_semantic_submodules_are_importable(self) -> None:
        state = CSAProposalStateDirect.from_policy(CSAProposalPolicy())
        attribution = PlannedProposalAttributionDirect(source_score=1.0)
        family_stat = ProposalFamilyStatDirect(family_key="mutation:0")

        assert state.update_index == 0
        assert attribution.proposal_family_key is None
        assert family_stat.family_key == "mutation:0"
