---
id: pn-9euu
status: closed
deps: [pn-h0tc, pn-4u63]
links: []
created: 2026-06-30T22:47:53Z
type: task
priority: 2
assignee: isty2e
parent: pn-qiwp
tags: [candidate-refinement, var-4ped]
---
# Document refinement execution boundaries

Logical ID: var-4ped. Document where refinement lives in the execution ontology and how users should interpret proposed, refined, evaluated, and accepted candidates.

## Acceptance Criteria

User-facing docs explain refinement boundary, accounting, local-search behavior, CSA adaptation behavior, and limitations; API docs include public refinement artifacts; docs avoid promising evaluation-protocol-owned refinement.


## Notes

**2026-07-01T00:18:09Z**

Documented refinement execution boundaries and public artifact surface.

Ontology partition: proposal/source/refined/evaluated/accepted candidates are now named separately; EvaluationProtocol owns only evaluation semantics for the actual evaluated candidate; CandidateRefinement remains execution-side provenance carried by EvaluationOutcome and terminal surfaces; RunMethod.tell remains record-based while tell_outcomes is the narrow metadata-aware hook.

Lean/ontology decisions: added one concept page rather than scattering the vocabulary across unrelated docs; did not add runtime models or serialization promises; corrected CandidateRefinement.changed_leaf_paths docstrings instead of introducing a new unknown-path state in this ticket; promoted only kernel implementation contract types already needed by public Kernel examples/signatures to the root facade.

Self-review passes: checked local-search ownership, Study transport/accounting ownership, CSA adaptation-only behavior, terminal result alignment/sentinel behavior, checkpoint/serialization limitations, and public import-path consistency. Edge findings fixed: stale glossary wording claimed EvaluationOutcome was consumed by RunMethod.tell; changed_leaf_paths=() wording conflicted with explicit empty-path CSA behavior; README example used deep non-facade imports while demonstrating a public kernel contract.

Verification: uv run pytest -> 614 passed, 16 skipped, 3 known joblib warnings; uv run pytest tests/surface/test_root_exports.py -> 2 passed; uv run ruff check src tests -> clean; uv run basedpyright -> 0 errors, 0 warnings; uv run mkdocs build --strict -> built successfully with upstream Material warning; README root-facade import smoke -> ok; git diff --check -> clean.
