---
id: pn-0n2b
status: closed
deps: [pn-cn4y]
links: []
created: 2026-06-30T22:47:53Z
type: task
priority: 1
assignee: isty2e
parent: pn-qiwp
tags: [candidate-refinement, var-0tjq]
---
# Emit refinement metadata from local-search kernels

Logical ID: var-0tjq. Local-search kernels that refine proposals must emit the typed refinement payload from var-qlp8 when a candidate is modified before evaluation.

## Acceptance Criteria

Structured local-search refinement emits source/refined candidates and changed paths; no-op refinement emits no payload and avoids avoidable allocation; evaluation count semantics remain unchanged; focused kernel tests cover changed and unchanged paths.


## Notes

**2026-06-30T23:04:26Z**

Hardened ticket self-review for logical var-0tjq:

1. Engineering style / ontology: local-search kernels own request-local refinement emission; EvaluationProtocol and Study must not infer it. Emit CandidateRefinement only at kernel outcome construction points.
2. Ontology hardening / adversarial cases: disabled local search, no improving move, max-step termination, stochastic sampled neighborhoods, scheduled pair moves, variable-neighborhood stages, iterated kicks, and SciPy continuous optimization must not collapse into one untyped diagnostics field.
3. Lean ontology / prune pressure: do not add a new public changed-path helper or hierarchy. Use existing topology owners: PreparedStructuredLocalSearchRuntime for structured discrete kernels and ContinuousStructuredSpaceCodec for SciPy continuous kernels.
4. Model responsibility: CandidateRefinement owns provenance validation; runtime/codec owns changed leaf-path calculation because it owns leaf topology and leaf value access; kernels own the decision of whether a proposal was actually refined.
5. Package taxonomy: keep path inference local to local-search runtime/adapter packages. Do not move CSA adaptation, Study assimilation, terminal accounting, or docs tutorial material into this ticket.
6. Performance re-attack: no-op paths must not compute changed paths or allocate CandidateRefinement. Structured kernels can gate on completed_steps; SciPy disabled paths already bypass optimization and should keep refinement=None.

**2026-06-30T23:09:00Z**

Edge-hunt campaign for logical var-0tjq:

Round 1
- Exploit cases: structured changed-path emission, no-improvement no payload, disabled local search no payload, SciPy continuous refinement attribution.
- Explore cases: iterated local search accepted kick without completed leafwise steps; stochastic sampled improvement.
- Command: uv run pytest tests/local_search/test_structured_local_optimization.py tests/local_search/test_scipy_local_optimization.py
- Result: failure found. Iterated accepted-kick refinement was missing because emission was gated only on completed_steps.
- Fix: added explicit accepted_refinement tracking in StructuredIteratedLocalSearchKernel.
- Green streak reset to 0.

Round 1 after fix
- Exploit cases: changed/no-op/disabled local-search metadata plus static type and lint surface.
- Commands: uv run pytest tests/local_search/test_structured_local_optimization.py tests/local_search/test_scipy_local_optimization.py && uv run ruff check src tests; uv run basedpyright
- Result: all-green.
- Green streak: 1.

Round 2 after fix
- Exploit cases: local-search package export/import blast radius; CSA proposal/covariance/orchestration adjacency before adaptation consumes refinement metadata.
- Command: uv run pytest tests/local_search tests/csa/test_csa_proposal.py tests/csa/test_csa_covariance.py tests/csa/test_csa_orchestration.py
- Result: all-green.
- Green streak: 2.

Round 3 after fix
- Exploit cases: full repo test interactions, docs rendering, static typing, lint, whitespace.
- Commands: uv run pytest; uv run basedpyright; uv run ruff check src tests && uv run mkdocs build --strict && git diff --check
- Result: all-green by exit status. Full pytest reported 578 passed, 16 skipped, with 3 pre-existing joblib worker-cancellation warnings. MkDocs Material emitted its upstream MkDocs 2.0 warning.
- Green streak: 3.

Residual risk: Study assimilation still drops/ignores refinement metadata until pn-fo46 lands; CSA adaptation still uses inference/fallback until pn-h0tc lands.
