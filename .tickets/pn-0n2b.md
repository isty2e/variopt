---
id: pn-0n2b
status: open
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

