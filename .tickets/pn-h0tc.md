---
id: pn-h0tc
status: open
deps: [pn-fo46, pn-0n2b]
links: []
created: 2026-06-30T22:47:53Z
type: task
priority: 1
assignee: isty2e
parent: pn-qiwp
tags: [candidate-refinement, var-g60y]
---
# Feed explicit refinement paths into CSA adaptation

Logical ID: var-g60y. CSA adaptation should use explicit refinement changed paths instead of inferring path changes from source/refined candidate comparison whenever metadata is available.

## Acceptance Criteria

CSA adaptation consumes explicit changed paths when present; fallback inference remains correct only where metadata is absent; no eager path inference happens for no-op refinement; tests cover categorical/numeric/mixed structured paths and disabled adaptation paths.

