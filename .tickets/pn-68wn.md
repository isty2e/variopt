---
id: pn-68wn
status: closed
deps: []
links: []
created: 2026-07-01T13:19:54Z
type: task
priority: 1
assignee: isty2e
---
# Make refinement candidate equality a space contract

GitHub #13: make refinement record-to-candidate validation use explicit search-space candidate equality semantics instead of raw candidate == candidate scalar bool assumptions. Preserve hard validation errors and cover ordinary scalar candidates plus ambiguous array-like equality.

