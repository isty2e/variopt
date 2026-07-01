---
id: pn-phl2
status: closed
deps: []
links: []
created: 2026-07-01T20:28:19Z
type: task
priority: 1
assignee: isty2e
---
# Fix dataclass pickle state override for refinement outcomes

Follow-up for issue #13 review: Python 3.11 frozen slotted dataclasses override in-class __getstate__/__setstate__, so attach pickle hooks after class creation and require explicit candidate_equal when revalidating changed unpickled refinements.

