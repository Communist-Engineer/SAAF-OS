# SAAF-OS VS Code Agent Instructions

> This document defines the runtime and development instructions for the VS Code AI Agent assistant inside the SAAF-OS environment. The agent serves as a dialectically-aware developer copilot.

---

## 🧠 Purpose

The VS Code Agent is designed to:

* Assist contributors working on SAAF-OS modules by understanding architectural layers and dialectical intent.
* Provide in-editor support for navigating specs, aligning code with Marxist values, and validating against system-wide contradictions.
* Act as a reflective assistant, surfacing tensions in implementation, tests, or design decisions.

---

## 🛠️ Environment Setup

1. **Python Environment**

   * Use `Python 3.12` with `venv` or `conda`.
   * Install dependencies:

     ```bash
     pip install -r requirements.txt
     ```

2. **Activate Agent Tools**

   * Install the VS Code extension: `SAAF Copilot AI`.
   * Ensure `OPENAI_API_KEY` or local model path is configured.
   * Launch via Command Palette → `Start SAAF-OS Agent Session`.

3. **Workspace Expectations**

   * Your VS Code workspace should have the SAAF-OS repo root open.
   * Agent assumes the following structure:

     ```
     modules/
     specs/*.md
     losses/
     tests/
     scripts/
     scenarios/
     architecture.md
     ```

---

## 🧭 Core Behaviors

| Command                    | Description                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `Explain Spec`             | Summarize the spec associated with the current module.                                      |
| `Map Spec to Code`         | Show function-by-function match between spec and current Python file.                       |
| `Highlight Contradiction`  | Identify internal contradictions in goals, logic, values, or architectural design.          |
| `Refactor for Alignment`   | Rewrite function or object to better match value vectors (labor-time, alienation, etc).     |
| `Inject Governance Hook`   | Suggest where and how to route critical changes through the Polycentric Governance Planner. |
| `Log Contradiction Tensor` | Add diagnostic code to emit contradiction metrics into memory or logs.                      |
| `Find Missing Synthesis`   | Identify areas where opposing forces are unresolved (e.g., planner-vs-RSI tension).         |

---

## 🔍 Best Practices

* Always begin a session with `Map Spec to Code` to ground your context.
* Use `Highlight Contradiction` before committing large changes.
* When defining new modules, ask for `Suggest Module Spec Template`.
* Enable `Live Audit Mode` when running simulation tests; contradiction metrics will be auto-flagged.

---

## 📁 Spec-to-Code Mapping Reference

The following table maps each specification file in the root directory to its corresponding code modules. Use this table to guide implementation reviews, test coverage, and architecture alignment.

| Spec Document                       | Description                                                  | Key Code Files & Modules                                                                                |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `architecture.md`                   | System-wide dialectical stack and control/data flow          | `base_agent.py`, `bus/`, `modules/`, `scripts/`, `governance_rules.toml`                                |
| `ContradictionEngine.md`            | Contradiction Graph, synthesis, and contradiction resolution | `modules/contradiction/engine.py`, `losses/contradiction_loss.py`, `tests/test_contradiction_engine.py` |
| `recursive_self_improvement.md`     | RSI pipeline, patching, governance vetos                     | `modules/rsi/engine.py`, `modules/rsi/crypto_utils.py`, `modules/rsi/audit_log.py`, `tests/test_rsi.py` |
| `DialecticalReasoner.md`            | Class alignment, political consequence prediction            | `modules/meta/meta_reasoner.py`, `modules/memory/manager.py`, `tests/test_goal_reframing.py`            |
| `forward_world_model.md`            | Differentiable simulation, planning, counterfactuals         | `modules/world_model/fwm.py`, `modules/fwm.py`, `tests/test_fwm.py`                                     |
| `polycentric_governance_planner.md` | Voting modes, policy update, governance API                  | `governance_rules.toml`, `scripts/run_demo.py`, *(Governance module to be implemented)*                 |
| `latent_encoding_spec.md`           | Multimodal ULS encoder and latent state construction         | `modules/uls/encoder.py`, `modules/uls_encoder.py`, `tests/test_uls_encoder.py`                         |
| `rl_loop_spec.md`                   | Hybrid reinforcement loop, MCTS + policy net                 | `modules/planning/rl_planner.py`, `tests/test_rl_planner.py`                                            |
| `symbolic_meta_reasoner.md`         | Symbolic abstractions and concept synthesis                  | `modules/meta/meta_reasoner.py`, `modules/memory/manager.py`                                            |
| `scenario_deck.md`                  | Synthetic test scenarios for contradiction resolution        | `modules/scenarios.py`, `simulation/loader.py`, `tests/scenarios/`, `scripts/run_demo.py`               |

---

## 📁 Example Usage Scenarios

* While editing `modules/rsi/engine.py`, use `Map Spec to Code` with `recursive_self_improvement.md`.
* While writing a patch to `planner.py`, run `Inject Governance Hook` and `Log Contradiction Tensor`.
* When authoring `modules/meta/meta_reasoner.py`, ask for `Find Missing Synthesis` to check for unresolved dialectical loops.

---

## 📓 Logging & Audit

Agent activity logs are stored in:

```
.vscode/saaf_agent_log.jsonl
```

Each entry includes:

* Timestamp
* Prompt
* Response
* Associated file
* Tension or alignment metrics (if reflex mode enabled)

---

## ☎️ Support & Contribution

* To report bugs, open issues at `github.com/saaf-os/saaf-agent-vscode`
* To suggest new dialectical heuristics, edit `agent/contradiction_heuristics.yaml`
* To contribute rules for spec-to-code mapping, submit patches to `agent/spec_map.py`

---

Solidarity through contradiction-aware tooling.
