
# File Structure Specification (`file_structure.md`)

> This document defines the recommended directory and file layout for the SAAF‑OS codebase. It helps maintain modularity, traceability, and developer clarity as AI agents and human contributors expand the system.

---

## 📁 Root Layout

```bash
saaf-os/
├── agents/                    # Agent classes and logic (e.g. base_agent, worker_agent)
├── modules/                   # Core cognition and control modules
│   ├── contradiction/         # Contradiction Engine + graph
│   ├── governance/            # Polycentric Governance Planner
│   ├── memory/                # Episodic and symbolic memory layers
│   ├── planning/              # RL loop, MCTS, planners
│   ├── reasoning/             # Dialectical Reasoner + ontology hooks
│   ├── rsi/                   # Recursive Self-Improvement Engine
│   └── uls/                   # Unified Latent Space encoder
├── simulation/                # World model, scenario runner, sim interfaces
├── bus/                       # Message bus adapters, pub/sub handlers
├── data/                      # Ontologies, config files, static test data
├── scripts/                   # CLI tools for testing, evaluation, monitoring
├── tests/                     # Unit and integration tests
├── docs/                      # Documentation and specs
│   ├── architecture.md
│   ├── message_bus_spec.md
│   ├── memory_spec.md
│   ├── testing_harness.md
│   └── ...
├── configs/                   # Agent config templates, governance rules, schema files
│   ├── governance_rules.toml
│   ├── example_agent_config.json
│   ├── class_ontology.jsonld
│   ├── value_vector_schema.json
├── requirements.txt
├── build_plan.md
└── README.md
```

---

## 🧠 Key Principles

- Modules are logically separated by *function*, not model type
- All inter-module communication goes through `bus/`
- No module should call another directly without an interface or message
- Tests are colocated for fast iteration (`tests/` mirrors top-level structure)

---

## 🛠 Suggested Naming Conventions

- Modules: `snake_case` for files, `CamelCase` for classes
- Topics: `module.action.verb` (e.g., `rsi.patch.proposed`)
- Configs: Use `.json` or `.toml` with strong schema validation

---

## 📦 Install & Execution

- Use `make`, `bash scripts`, or `cli.py` in `scripts/` to run:
  - `make test`
  - `python scripts/run_scenario.py --scenario energy_conflict`
  - `python scripts/launch_agent.py --config configs/example_agent_config.json`

---

## ✅ Summary

This structure supports AI-led code generation, human-auditable oversight, and layered cognitive growth—just like the system it contains.

