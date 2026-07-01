---
id: pn-fo46
status: closed
deps: [pn-cn4y, pn-0n2b]
links: []
created: 2026-06-30T22:47:53Z
type: task
priority: 1
assignee: isty2e
parent: pn-qiwp
tags: [candidate-refinement, var-tz0p]
---
# Thread refinement metadata through Study assimilation

Logical ID: var-tz0p. Study assimilation must preserve typed refinement metadata from evaluation outcomes through accepted observations and trace/report artifacts.

## Acceptance Criteria

Assimilation keeps refinement metadata attached to the relevant accepted candidate record; exact async and synchronous paths preserve it consistently; non-refined outcomes remain allocation-light; focused tests cover sync, exact async resume/session surfaces, and batch ordering.


## Notes

**2026-06-30T23:15:23Z**

Procedure hardening and self-review before implementation:
1. Ontology split: refinement is execution provenance attached to an EvaluationOutcome, while Study assimilation consumes records for RunMethod.tell. Keep RunMethod state assimilation record-only and preserve provenance in Study-owned artifacts, not optimizer semantics.
2. Boundary basis: RunReport is the generic terminal report over ordered record history, so a record-aligned refinement channel belongs there. TraceEvent is diagnostics-only scalar metadata and should not become a typed provenance carrier.
3. Lean pressure: do not add a new public wrapper/session API or change Study.step/finish signatures. Add the smallest invariant-bearing field to RunReport and lazy threading in run().
4. Allocation invariant: no-refinement runs must not allocate a tuple of None values per record; empty refinements means no refinement metadata was recorded.
5. Responsibility audit: CandidateRefinement owns refinement facts and path validation; RunReport owns report-level alignment validation; Study execution owns workflow threading; exact-async session owns lifecycle storage, not terminal accounting.
6. Package taxonomy: terminal provenance stays in artifacts/terminal.py; orchestration threading stays in study/execution.py; exact-async session tests stay with study tests; no new package or broad helper module is justified.
7. Rejected alternatives: attaching refinement to Observation/EvaluationRecord collapses semantic evaluation with execution provenance; changing RunMethod.tell widens optimizer API unnecessarily; stuffing typed refinement into TraceEvent mixes diagnostics with domain provenance; adding per-surface terminal summaries belongs to the dependent accounting ticket.
8. Edge-hunt targets: mismatched RunReport refinement length, mixed refined/unrefined batch alignment, exact-async out-of-order ordering, suspend/resume ordered_outcomes preservation, zero/no-refinement compact sentinel, and evaluation-cost budgeting alignment.

**2026-06-30T23:34:33Z**

Implementation and edge-hunt completion log:
- Implemented record-aligned refinement provenance on RunReport with compact no-metadata sentinel `refinements == ()`.
- Threaded refinement metadata through sync, exact-async, and stale-async Study run assimilation without changing RunMethod.tell, Study.step, or exact-async finish return signatures.
- Moved CandidateRefinement to the artifacts ontology tier because both EvaluationOutcome and RunReport need the same provenance value object; kept variopt.outcomes.CandidateRefinement as an explicit compatibility export.
- Docs synced in glossary, optimization contract, and optimization ontology notes.

Edge-hunt failures found and fixed:
1. Aligned all-None refinement metadata could survive instead of canonicalizing to the empty sentinel; fixed in RunReport.__post_init__.
2. Direct no-metadata RunReport instances entered strict zip validation and failed normal reports; fixed by returning immediately for the empty sentinel.
3. Stale-async Study.run discarded refinement metadata; fixed with the same lazy record-aligned threading used by sync/exact run.

Adversarial cases added:
- RunReport rejects refinement length mismatch, zero-record metadata, mismatched refined candidates, and ambiguous candidate equality.
- RunReport canonicalizes all-None refinement metadata from from_records and direct constructor.
- Sync Study.run preserves mixed refined/unrefined alignment, late first refinement backfill, no-refinement compact sentinel, and evaluation-cost overshoot alignment.
- Exact-async Study.run preserves both kernel-origin and evaluator-origin refinement order under out-of-order completion.
- Exact-async suspend/resume preserves already-completed refinement payloads in ordered_outcomes.
- Stale-async Study.run preserves refinement completion order and keeps no-refinement reports compact.

Three consecutive all-green rounds after the last fix:
1. `uv run pytest tests/core/test_runtime_artifacts.py tests/study/test_study.py tests/study/test_study_exact_async.py tests/study/test_study_stale_async.py tests/surface/test_root_exports.py` -> 75 passed; `uv run ruff check src tests` clean; `uv run basedpyright` 0 errors/0 warnings/0 notes.
2. `uv run pytest tests/local_search tests/study tests/core/test_runtime_artifacts.py` -> 110 passed; `uv run pytest tests/evaluators tests/core/test_problem_execution.py tests/core/test_problem_contracts.py` -> 39 passed with 3 known joblib cancellation warnings; `uv run mkdocs build --strict` succeeded with the upstream MkDocs Material warning.
3. `uv run pytest` -> 593 passed, 16 skipped, 3 known joblib cancellation warnings; `uv run ruff check src tests` clean; `uv run basedpyright` 0 errors/0 warnings/0 notes; `git diff --check` clean.
