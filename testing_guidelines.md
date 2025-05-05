
# Testing Guidelines for SAAF‑OS (`testing_guidelines.md`)

> This document defines the expectations and structure for unit, integration, and scenario testing in the SAAF‑OS codebase. All modules—especially those involving contradiction reasoning, planning, self-improvement, and governance—must include appropriate test coverage.

---

## ✅ General Testing Requirements

- Use `pytest` for all tests.
- All modules must include:
  - Unit tests for core functions
  - Integration tests with message flow and memory interaction
  - Scenario-based validation using `scenario_deck.md`
- Tests must pass in CI or pre-commit before deployment of new patches.

---

## 📌 Unit Testing Guidelines

| Component            | Required Unit Tests                                             |
|---------------------|------------------------------------------------------------------|
| ContradictionEngine | `score_tension()`, `detect_conflict()`, A* planner edge cases   |
| ForwardWorldModel   | `simulate()`, `predict_next()`, latent alignment loss validation|
| RLPlanner           | MCTS feasibility pruning, plan comparison scoring               |
| RSIEngine           | Patch generation logic, fitness regression checks               |
| GovernancePlanner   | Quorum logic, veto triggers, vote weight calculation            |
| MemoryManager       | Episodic write/read integrity, symbolic triple resolution       |
| ULS Encoder         | `encode_state()`, latent variance under perturbed inputs        |
| DialecticalReasoner | Class interest shift, symbolic rule match, RDF resolution       |

---

## 🔁 Integration Testing

- Simulate full message-passing cycles between 2–3 modules:
  - ContradictionEngine → FWM → RLPlanner
  - RSIEngine → Governance → Memory
- Validate message schemas match `message_bus_spec.md`
- Use mock memory and bus adapters

---

## 🧪 Scenario-Driven Testing

- Run at least 3 scenarios from `scenario_deck.md` per integration milestone
- Required Metrics:
  - Contradiction resolved %
  - Patch acceptance rate
  - Value vector drift
  - Governance decision frequency

---

## 🛠 Developer Notes

- Place all tests in `/tests` with mirrors of module structure.
- Use test doubles for governance, memory, and sensors where needed.
- Include assertion coverage for:
  - Boundary conditions
  - Regression against golden outputs
  - Safety guardrails (e.g., patch rollback, veto enforcement)

---

## 🧬 Test Extensions (Optional)

- Use `hypothesis` for property-based testing on symbolic reasoners
- Generate test traces for RL rollouts and patch effects across generations
- Validate consistency of latent states under minor input perturbations

---

## ✅ Summary

Tests are critical for contradiction-rich, evolving systems like SAAF‑OS. Every contradiction resolved or patch applied must be measurable, auditable, and reproducible. These guidelines ensure trust, alignment, and reflexive growth are tested—not just assumed.

