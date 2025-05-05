
# Build Plan for SAAF‑OS Prototype (`build_plan.md`)

> This plan outlines the staged development path to bring up a functioning SAAF‑OS prototype, including recommended priorities, module order, interface testing, and scenario-based validation.

---

## 🧭 Phase 0: Infrastructure & Foundations (Week 0–1)

### ✅ Deliverables
- Set up message bus (`message_bus_spec.md`)
- Implement base agent class (`base_agent.py`)
- Add simulation backend scaffolds (`simulation_backends.md`)
- Define class ontology and data schemas (`class_ontology.jsonld`, `value_vector_schema.json`)
- Build in-memory message queue, logging, and replay support

---

## 🧠 Phase 1: Cognition Core Loop (Week 1–2)

### ✅ Deliverables
- Implement ULS encoder (`latent_encoding_spec.md`)
- Basic `encode_state(u_t) → z_t` with real and synthetic data
- Implement Contradiction Engine:
  - Tension graph
  - EWMA edge scoring
  - A* synthesis search
- First metric logs: contradiction emergence, plan delta

---

## 🔁 Phase 2: Planning & Simulation (Week 2–3)

### ✅ Deliverables
- Implement Forward World Model (FWM):
  - `simulate()`, `evaluate_plan()`
- Implement RL loop (`rl_loop_spec.md`):
  - Policy π_θ and latent MCTS
- Test end-to-end: contradiction → synthesis plan → z_{t+1}

---

## 🗳️ Phase 3: Governance & Safety Layer (Week 3–4)

### ✅ Deliverables
- Implement Polycentric Governance Planner:
  - Vote casting, quorum check, policy gating
- Add Governance Gate and Value-Lock check
- Load `governance_rules.toml`
- Simulate scenarios where policy blocks high-impact RSI patches

---

## ♻️ Phase 4: Recursive Self-Improvement (Week 4–5)

### ✅ Deliverables
- Patch generator (LLM or DSL-based)
- RSI Deployer + rollback safety
- Fitness tracker and governance veto pathway
- Memory logging of patches and outcomes

---

## 🧪 Phase 5: Scenario Testing & Expansion (Week 5+)

### ✅ Deliverables
- Implement `testing_harness.md`
- Run scenarios from `scenario_deck.md`
- Measure:
  - Contradiction resolution %
  - Governance override frequency
  - Value drift
  - RSI effectiveness

---

## 📌 Tooling Notes

- Each module must:
  - Register to the Message Bus
  - Declare its topic interface
  - Include unit + scenario test hooks

---


## 🔍 Testing Milestones (added)

Each phase below includes associated testing goals:

---

## 🧭 Phase 0: Infrastructure & Foundations (Week 0–1)

**Tests:**
- Validate MessageBusAdapter routing across mock topics
- Unit test base_agent tick timing and message logging

---

## 🧠 Phase 1: Cognition Core Loop (Week 1–2)

**Tests:**
- Unit test `encode_state()` for known inputs
- ContradictionEngine:
  - Tension score boundary values
  - A* fails on unsolvable contradiction sets
- Validate that `z_t` variance responds to input changes

---

## 🔁 Phase 2: Planning & Simulation (Week 2–3)

**Tests:**
- FWM simulate rollouts vs. known ground truth
- RLPlanner:
  - Unit test hybrid policy fallback logic
  - MCTS feasibility pruning
- Integration test: contradiction → simulation → action trace

---

## 🗳️ Phase 3: Governance & Safety Layer (Week 3–4)

**Tests:**
- GovernancePlanner:
  - Unit test vote weight scaling, quorum logic
  - Value-lock enforcement
- Governance gate override test with veto trace logging
- Memory logs vote outcomes with full context

---

## ♻️ Phase 4: Recursive Self-Improvement (Week 4–5)

**Tests:**
- RSI:
  - Unit test patch generation + rollback failover
  - Fitness scoring and memory comparison
- Integration: RSI → governance gate → deploy or veto trace

---

## 🧪 Phase 5: Scenario Testing & Expansion (Week 5+)

**Tests:**
- Full system test: simulate all `scenario_deck.md` cases
- Track contradiction resolution rate, RSI patch success
- Regression tests: ensure resolution patterns match golden set


## ✅ Final Goal: Reflexive, Planning Agent Loop

1. Input: `u_t` → encode → `z_t`
2. Contradiction Engine logs tension
3. FWM simulates possible futures
4. RL loop generates candidate plans
5. Governance votes on patch or vetoes
6. Memory logs events, RSI learns from feedback

---

## 🧠 Bonus Goals (Phase 6+)

- Add class-based agent differentiation
- Add neurosymbolic plan abstraction for generalization
- Enable distributed agent swarms and latent consensus

