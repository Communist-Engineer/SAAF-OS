
# System Requirements for SAAF‑OS (`system_requirements.md`)

> This document outlines the hardware, software, and runtime dependencies required to build, test, and deploy the SAAF‑OS prototype. It is intended for AI agents, developers, and integrators setting up the system.

---

## 🖥️ Minimum Hardware Requirements

| Component        | Minimum                          | Recommended                        |
|------------------|----------------------------------|------------------------------------|
| CPU              | 4-core x86_64                    | 8-core+ x86_64 or ARM64            |
| RAM              | 8 GB                             | 16 GB or more                      |
| Storage          | 10 GB SSD                        | 50 GB NVMe                         |
| GPU (optional)   | CUDA 11.7+ compatible (NVIDIA)   | RTX 30XX+ w/ 12GB VRAM             |

---

## 🧰 Software Stack

### 🧠 Language & Runtime

- Python 3.10+
- Optional: Conda or venv for environment management

### 📦 Core Libraries

| Category         | Library                          |
|------------------|----------------------------------|
| ML / RL          | `torch`, `transformers`, `stable-baselines3` |
| GNN / Graphs     | `torch-geometric`, `networkx`    |
| Simulation       | `mesa`, `numpy`, `gym`, `scipy`  |
| Messaging        | `pyzmq`, `nats-py`                |
| Serialization    | `protobuf`, `pydantic`, `orjson` |
| Ontologies       | `rdflib`, `jsonld`, `owlready2`  |
| CLI / Logging    | `rich`, `click`, `loguru`         |
| Governance / Math| `cvxpy`, `sympy`                  |

---

## ⚙️ Services & Tools

- **Redis** *(optional)* — used for memory cache or message brokering
- **Docker** *(optional)* — for containerized deployment
- **Git** — for version control and patch diff tracking
- **Make / shell scripts** — for CLI test harness

---

## 🔄 Message Bus Layer

| Protocol | Description |
|----------|-------------|
| ZeroMQ   | Default pub/sub transport |
| NATS     | Optional high-performance alternative |
| REST/gRPC | (Optional) for external governance dashboard APIs |

---

## 🚀 GPU Acceleration

- `torch` will auto-detect CUDA
- GPU acceleration is used for:
  - ULS encoder models
  - Forward World Model rollouts
  - Latent-space MCTS evaluations (optional)

---

## 🧪 Testing & Validation

- Test runner: `pytest`
- Linting: `black`, `flake8`
- Scenario harness: `testing_harness.md` driven CLI runner
- Use `scenario_deck.md` to validate contradiction, governance, and planning modules

---

## ✅ Summary

This environment balances symbolic reasoning, neural planning, governance simulation, and recursive self-improvement. It is modular and can run without a GPU, but benefits significantly from acceleration.

