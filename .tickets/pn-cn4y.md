---
id: pn-cn4y
status: closed
deps: []
links: []
created: 2026-06-30T22:47:53Z
type: task
priority: 1
assignee: isty2e
parent: pn-qiwp
tags: [candidate-refinement, var-qlp8]
---
# Model refinement outcome metadata

Logical ID: var-qlp8. Add a typed candidate refinement artifact to runtime outcomes. It must represent source candidate, refined candidate, and explicit changed structured paths without using untyped metadata bags. Keep EvaluationProtocol semantics separate from refinement execution metadata.

## Design

Ontology constraints: EvaluationProtocol evaluates candidates; refinement is execution-side provenance attached to an outcome. The artifact must be canonical enough for Study and CSA to consume later, but not a speculative hierarchy.

## Acceptance Criteria

A typed refinement payload exists; EvaluationOutcome can carry it without changing evaluation counts; invalid/inconsistent payloads are rejected or made unrepresentable; root/public exports and API docs are synchronized if the payload is public; focused tests cover changed-path normalization, no-refinement outcomes, and inconsistent refinement payloads.


## Notes

**2026-06-30T22:50:09Z**

Hardened ticket self-review for logical var-qlp8:

1. Engineering style / ontology: refinement is execution-side provenance on EvaluationOutcome, not EvaluationProtocol semantics. Keep the flow as Proposal -> EvaluationRequest -> EvaluationRecord -> EvaluationOutcome(+refinement), not protocol-owned refinement.
2. Ontology hardening / adversarial cases: source candidate may differ from evaluated candidate, changed paths may be empty, non-scalar EvaluationRecord payloads must still work, and later CSA adaptation needs explicit paths without re-inferring on no-op paths.
3. Lean ontology / prune pressure: do not add a hierarchy or metadata dict. Add the smallest typed value object only if it enforces candidate/path invariants and serves later Study/CSA tickets.
4. Model responsibility: CandidateRefinement owns only source/refined candidate plus changed-path normalization/validation. EvaluationOutcome owns cross-object consistency with its carried record and evaluation accounting. It must refuse study assimilation, CSA policy, serialization, and terminal aggregation responsibilities.
5. Package taxonomy: keep the new artifact in variopt.outcomes because it is outcome-local execution provenance, not request-plane, record-plane, or terminal/report state. Export from the root only if it is a user-facing/type-hint surface attached to public EvaluationOutcome.
6. Re-attack after partition: rejected alternatives are EvaluationProtocol fields (mixes semantic evaluation with execution provenance), KernelDiagnostics metadata (untyped diagnostic side channel), and variopt.artifacts placement (wrong artifact tier). The surviving basis is EvaluationRecord owns semantic result; CandidateRefinement owns pre/post candidate provenance; EvaluationOutcome owns execution accounting and consistency.

**2026-06-30T23:01:16Z**

Edge-hunt campaign for logical var-qlp8:

Round 1
- Exploit cases: duplicate changed paths; mismatched refined candidate; no-refinement outcome default; scalar observation compatibility.
- Explore cases: non-scalar LabelRecord outcome; list-to-tuple changed-path normalization.
- Command: uv run pytest tests/core/test_runtime_artifacts.py -k "refinement or outcome" && uv run pytest tests/surface/test_root_exports.py && uv run ruff check src tests
- Result: all-green.
- Green streak: 1.

Round 2
- Exploit cases: public reference drift; glossary link target; architecture ontology row; optimization-contract invariant.
- Explore cases: mkdocstrings root facade rendering; strict navigation/reference validation.
- Command: uv run mkdocs build --strict
- Result: all-green. MkDocs Material emitted its upstream MkDocs 2.0 warning; no strict build failure.
- Green streak: 2.

Round 3
- Exploit cases: generic protocol blast radius; evaluator/study type bounds; full runtime artifact suite; CSA/local-search import blast radius.
- Explore cases: optional benchmark skips; whole-suite interaction with joblib async tests.
- Commands: uv run pytest; uv run basedpyright
- Result: all-green by exit status. Full pytest reported 578 passed, 16 skipped, with 3 pre-existing joblib worker-cancellation warnings.
- Green streak: 3.

Post-round expression cleanup changed only the changed-path segment predicate. Fresh focused verification after that cleanup: uv run pytest tests/core/test_runtime_artifacts.py -k "refinement or outcome"; uv run basedpyright; uv run ruff check src tests.

Residual risk: CandidateRefinement records changed paths but local-search kernels do not emit it yet; that is tracked by pn-0n2b. Study/CSA terminal propagation remains in dependent tickets.
