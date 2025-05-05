
# Module Registry (`module_registry.md`)

> This registry provides a unified overview of all major modules in the SAAF‑OS architecture. Each module includes its role, message interfaces, dependencies, and integration notes. This helps coordinate development, testing, and system orchestration.

---

## 📦 Core Modules

### 🧠 `ULSEncoder`
- **Role**: Encodes raw system/environment state `u_t` into unified latent state `z_t`
- **Publishes**: `agent.state.update`
- **Subscribes**: None (invoked locally)
- **Dependencies**: Encoder models, latent manifold spec

---

### 🔥 `ContradictionEngine`
- **Role**: Maintains the Contradiction Graph, scores tensions, proposes synthesis paths
- **Publishes**: `contradiction.event.detected`, `contradiction.plan.suggested`
- **Subscribes**: `agent.state.update`, `fwm.simulation.result`
- **Dependencies**: Memory, message bus, scenario inputs

---

### 🧮 `ForwardWorldModel`
- **Role**: Simulates forward latent transitions `F(z_t, a_t) → z_{t+1}`
- **Publishes**: `fwm.simulation.result`
- **Subscribes**: `fwm.simulation.request`
- **Dependencies**: ULS, simulation backend, time embeddings

---

### 🎯 `RLPlanner`
- **Role**: Hybrid RL controller using direct policy and MCTS for planning
- **Publishes**: `agent.plan.complete`, `fwm.simulation.request`
- **Subscribes**: `agent.state.update`, `contradiction.plan.suggested`, `fwm.simulation.result`
- **Dependencies**: ForwardWorldModel, Memory, RL library

---

### 🗳️ `PolycentricGovernancePlanner`
- **Role**: Applies collective value-aligned voting to policy changes and RSI patches
- **Publishes**: `governance.vote.cast`, `governance.policy.updated`
- **Subscribes**: `rsi.patch.proposed`, `memory.episodic.store`
- **Dependencies**: Governance ruleset, agent class positions, vote weight logic

---

### ♻️ `RSIEngine`
- **Role**: Generates, tests, and deploys architectural patches
- **Publishes**: `rsi.patch.proposed`, `rsi.patch.vetoed`
- **Subscribes**: `agent.plan.complete`, `governance.policy.updated`
- **Dependencies**: Memory, governance, patch fitness tracker, LLM rewrite hook

---

### 🧠 `DialecticalReasoner`
- **Role**: Performs class conflict modeling and historical-materialist inference
- **Publishes**: `memory.episodic.store`
- **Subscribes**: `contradiction.event.detected`, `agent.state.update`
- **Dependencies**: Symbolic knowledge base, temporal graph embeddings

---

### 💾 `MemoryManager`
- **Role**: Stores and retrieves episodic, symbolic, and patch memory
- **Publishes**: `memory.episodic.store`
- **Subscribes**: `agent.plan.complete`, `fwm.simulation.result`, `rsi.patch.proposed`
- **Dependencies**: Storage backend, schema validation, vector store

---

### 🧪 `ScenarioRunner`
- **Role**: Loads and runs predefined test environments
- **Publishes**: `agent.tick.request`, `memory.retrieve.request`
- **Subscribes**: `agent.plan.complete`, `agent.state.update`
- **Dependencies**: Scenario deck, testing harness, simulation backend

---

### 🧵 `MessageBusAdapter`
- **Role**: Wraps underlying pub/sub implementation (e.g. ZeroMQ or NATS)
- **Publishes**: Any
- **Subscribes**: Any
- **Dependencies**: `message_bus_spec.md`

---

## ✅ Usage

Each module registers its name, type, and topic schema with the system coordinator on boot. Governance-critical modules must expose configuration hashes for validation.

