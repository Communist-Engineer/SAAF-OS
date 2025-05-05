
# Memory Specification (`memory_spec.md`)

> This document defines the structure and operational semantics of the Memory System in SAAF‑OS. The memory subsystem provides agents with both short-term episodic memory and long-term symbolic/semantic memory for use in planning, contradiction analysis, recursive self-modification, and governance.

---

## 🧠 Overview

SAAF‑OS agents rely on memory for:
- Encoding experience over time (episodic trace)
- Symbolic retrieval of past plans, failures, contradictions
- Storing contradiction gradients, value vector shifts, and governance outcomes
- Grounding future predictions and justifications in historical data

---

## 📦 Memory Types

### 1. Episodic Memory
- Time-indexed sequences of latent states, actions, and contradiction outcomes
- Used by RSI, FWM, and planning modules

**Schema:**
```json
{
  "timestamp": "ISO8601",
  "agent_id": "string",
  "z_t": "float[]",
  "action": "string or vector",
  "observed_result": "float[] or token",
  "contradictions": ["conflict_type_1", "conflict_type_2"],
  "value_vector": [labor_time, surplus_value, commons_share, alienation]
}
```

### 2. Semantic Memory
- Symbolic triples, plans, and concept traces
- Grounded in class-conscious ontology

**Schema:**
```json
{
  "subject": "AgriBot",
  "predicate": "violated",
  "object": "commons_share_policy",
  "context": "plan_trace_149",
  "source": "contradiction.event.detected",
  "timestamp": "ISO8601"
}
```

### 3. Patch History Memory
- Stores all RSI proposals, fitness scores, governance outcomes

**Schema:**
```json
{
  "patch_id": "uuid",
  "module": "planner",
  "diff": "git diff or AST delta",
  "fitness_score": "float",
  "governance_vote": {
    "outcome": "approved | vetoed",
    "vote_vector": [...],
    "quorum": "boolean"
  },
  "rollback_window": "ISO8601 interval"
}
```

---

## 🔍 Memory Access Interfaces

### Retrieval Types:
| Method | Input | Output |
|--------|-------|--------|
| `retrieve_episodic(k)` | recent k entries | ordered trace |
| `retrieve_by_contradiction(type)` | contradiction type | related episodes |
| `retrieve_by_symbol(subject, predicate)` | symbolic KB lookup | matching triples |
| `retrieve_patch_history(module)` | module name | prior patches and outcomes |
| `retrieve_by_temporal_window(start, end)` | time range | any memory object |

### Storage Interface:
- `store_episodic(entry)`
- `store_symbolic(triple)`
- `store_patch(entry)`

---

## 🧠 Integration Points

| Module | Usage |
|--------|-------|
| RSI | Uses memory to train fitness functions and avoid repeated failures |
| Contradiction Engine | Logs tension deltas and contradiction resolutions |
| Dialectical Reasoner | Retrieves symbolic relations and past class conflicts |
| FWM | Extracts historical sequences to train `z_t → z_t+1` model |
| Governance | Logs votes and policy evolution by timestamp |

---

## 🔒 Security & Consistency

- All memory writes are cryptographically hashed and auditable.
- Symbolic memory uses soft attention for retrieval + hard cache fallback.
- Memory can be replayed deterministically for rollback or explanation.

---

## 📌 Summary

Memory is the reflective substrate of SAAF‑OS—an agent cannot plan, self-modify, or govern without it. This spec ensures consistent, multi-modal retention of experience across contradiction-driven lifecycles.

