
# Message Bus Specification (`message_bus_spec.md`)

> This document defines the communication substrate for all SAAF‑OS modules. It provides a publish/subscribe messaging protocol to enable coordination between distributed components such as the Contradiction Engine, Forward World Model, RSI, Governance Planner, and Agents.

---

## 🎯 Purpose

- Ensure modularity and decoupling of components.
- Support asynchronous, distributed communication.
- Enable traceability and auditability of all interactions.

---

## 📡 Transport Layer

| Component     | Protocol     | Notes                           |
|---------------|--------------|----------------------------------|
| Message Bus   | ZeroMQ or NATS | Lightweight, pub/sub, low-latency |
| Serialization | JSON or Protobuf | Use JSON for debugging, Protobuf for production |

---

## 🧭 Core Concepts

- **Topic**: A named channel for communication (e.g. `contradiction.events`, `governance.vote.cast`)
- **Publisher**: Any module emitting a message.
- **Subscriber**: Any module reacting to a message.

---

## 📬 Message Structure (Generic)

```json
{
  "header": {
    "msg_id": "uuid",
    "timestamp": "ISO8601",
    "sender": "module_name",
    "topic": "string",
    "priority": "low | normal | high"
  },
  "payload": {
    "type": "schema_version",
    "data": { /* topic-specific content */ }
  }
}
```

---

## 🧵 Standard Topics

### 🔹 Contradiction Engine

| Topic | Payload Schema |
|-------|----------------|
| `contradiction.event.detected` | `{ edge_id, tension_score, contradiction_type, context_z_t }` |
| `contradiction.plan.suggested` | `{ plan_id, actions, tension_diff, synthesis_path }` |

---

### 🔹 Forward World Model

| Topic | Payload Schema |
|-------|----------------|
| `fwm.simulation.request` | `{ state_z_t, action_seq, horizon }` |
| `fwm.simulation.result`  | `{ plan_id, trajectory, metrics }` |

---

### 🔹 Recursive Self-Improvement (RSI)

| Topic | Payload Schema |
|-------|----------------|
| `rsi.patch.proposed` | `{ patch_id, module, diff, rationale, fitness_score }` |
| `rsi.patch.vetoed`   | `{ patch_id, reason, governance_vector }` |

---

### 🔹 Polycentric Governance

| Topic | Payload Schema |
|-------|----------------|
| `governance.vote.cast`   | `{ voter_id, vote_id, vote_weight, vector }` |
| `governance.policy.updated` | `{ policy_id, scope, change_log }` |

---

### 🔹 Memory & Observation

| Topic | Payload Schema |
|-------|----------------|
| `memory.episodic.store` | `{ z_t, action, result, timestamp }` |
| `memory.retrieve.request` | `{ query_type, target, time_window }` |

---

### 🔹 Agent Interface

| Topic | Payload Schema |
|-------|----------------|
| `agent.tick.request` | `{ agent_id, current_time }` |
| `agent.state.update` | `{ agent_id, z_t, status, contradiction_level }` |
| `agent.plan.complete` | `{ agent_id, plan_id, result }` |

---

## 🔐 Security & Access Control

- All messages must be signed with module keys.
- Governance layer may filter or intercept high-impact messages.
- Audit trail enabled via message logging middleware.

---

## 🔧 Developer Notes

- Use topic namespaces (`contradiction.*`, `rsi.*`) for clarity.
- Default message priority is `normal`.
- Every message should include `z_t` or a `context_id` for ULS traceability.

---

## ✅ Summary

The Message Bus is the nervous system of SAAF‑OS: it connects perception, planning, contradiction synthesis, memory, and governance into a unified flow. All modules must speak through this protocol to ensure modularity, reflexivity, and class-conscious supervision.
