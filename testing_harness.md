
# Testing Harness Specification (`testing_harness.md`)

> This document outlines the testing framework for validating the performance, stability, and correctness of SAAF‑OS modules across synthetic contradiction scenarios, recursive self-modification, and governance integration.

---

## 🎯 Goals

- Validate contradiction emergence and resolution behavior
- Ensure recursive self-improvement patches improve fitness
- Verify policy alignment with collective goals via governance logs
- Test latent encoding quality and forward model accuracy
- Measure end-to-end performance under synthetic scenario stress

---

## 🧪 Core Components

### 1. Scenario Loader

- Loads predefined environments from `scenario_deck.md`
- Seeds agents, world states, and initial contradictions
- Supports batch simulation and checkpointing

### 2. Metrics Engine

Tracks the following:

| Category | Metric |
|----------|--------|
| Contradiction | Tension score (avg/max), resolution rate, unresolved count |
| RSI | Patch success rate, average fitness gain, veto rate |
| Planning | Plan divergence (Δ intended vs. actual path), success %
| Governance | Vote logs, quorum % met, override counts |
| FWM | Prediction error (MSE), feasibility rate, rollout length |
| Value Drift | Δ per value vector dimension across episodes |
| Latent Encoding | Latent variance, InfoNCE loss, alignment consistency |

### 3. Test Modes

| Mode | Description |
|------|-------------|
| `unit` | Module-level test: encoder, contradiction score, patch logic |
| `integration` | Test full loops: contradiction → plan → governance → RSI |
| `scenario` | End-to-end test using predefined deck |
| `regression` | Repeat test against known golden outputs (e.g. historical contradiction behavior) |

---

## 🧰 CLI Test Runner

Example:
```bash
saaf-test --scenario agri_conflict           --mode integration           --metrics contradiction,rsi,planning           --output ./results/run_01.json
```

---

## 📤 Output Format

Results should be stored in structured JSON:

```json
{
  "scenario": "agri_conflict",
  "mode": "integration",
  "timestamp": "ISO8601",
  "metrics": {
    "contradiction": {
      "avg_tension": 0.41,
      "resolved": 7,
      "unresolved": 2
    },
    "rsi": {
      "patches_proposed": 12,
      "fitness_gain_avg": 0.17,
      "vetoes": 2
    }
  },
  "trace": {
    "events": [ /* ordered event log */ ]
  }
}
```

---

## 🧠 Integration Notes

- Hooks into message bus and memory system for logging
- Supports headless simulation and deterministic replay
- Can be extended to monitor GPU usage, step latency, and agent throughput

---

## ✅ Summary

The testing harness ensures SAAF‑OS remains dialectically sound, recursively improvable, and class-conscious under pressure. It validates that contradiction drives growth—and that growth remains accountable to collective governance.

