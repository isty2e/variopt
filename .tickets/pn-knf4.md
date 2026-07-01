---
id: pn-knf4
status: closed
deps: []
links: []
created: 2026-07-01T20:41:20Z
type: bug
priority: 2
assignee: isty2e
---
# Carry terminal refinement equality through replace

Terminal run artifacts accept space-owned candidate equality during construction, but dataclasses.replace can drop the InitVar and fall back to scalar equality when refinements change. Carry the comparison contract like EvaluationOutcome, while stripping non-picklable predicates from pickle state.
