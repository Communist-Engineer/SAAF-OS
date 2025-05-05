# SAAF‑OS ‑ System Architecture (v0.4, 2025‑05‑03)

> **Purpose**  
> Provide a single‑source overview of how Self‑Adaptive Agent – Open Source (SAAF‑OS) fits together as a post‑labor, class‑conscious automation stack on the path to AGI.

---

## 1 Layered View (“Dialectical Stack”)

| Layer | Core Responsibility | Key Modules |
|-------|--------------------|-------------|
| **6 Dialectical Supervision** | Model social totality, class interests, and contradictions. | *Dialectical Reasoner & Class‑Consciousness Model* |
| **5 Ethical & Governance** | Collective decision‑making, value‑drift detection. | *Polycentric Governance Planner* |
| **4 Recursive Self‑Improvement** | Code/architecture evolution, auto‑refactor, rollback. | *Recursive Self‑Improvement System* |
| **3 World Model & Planning** | Causal simulation, counterfactual reasoning. | *Forward World Model*, *Meta‑Reasoner* |
| **2 Memory & Embodiment** | Long‑term memory, tool use, sensorimotor grounding. | Episodic & Semantic Memory, Tool Manager, Embodied Agent APIs |
| **1 Base Framework** | Distributed agent runtime, messaging, health, I/O. | Agent Core, Message Bus, Health Monitor |

*(A minimalist ASCII diagram is in §8.)*

---

## 2 Module Specs (Concise)

| Module | Brief | Primary Interfaces |
|--------|-------|--------------------|
| **Contradiction Engine** | Detects & mediates dialectical tensions in plans/goals. | `detect_contradictions`, `resolve_contradiction`, `evaluate_praxis_outcome` |
| **Forward World Model** | Learned simulator for “imagine‑then‑act” planning. | `simulate`, `generate_plan`, `evaluate_plan` |
| **Dialectical Reasoner & CCM** | Computes class alignment & historic outcome risk. | `assess_class_alignment`, `predict_political_consequence`, `prioritize_goals` |
| **Recursive Self‑Improvement** | NAS + code rewrite, version & rollback. | `evaluate_module`, `propose_modification`, `apply_patch` |
| **Polycentric Governance Planner** | Human/agent deliberation, consensus, policy update. | `initiate_vote`, `adjust_policy`, `monitor_governance_deviation` |

_For full field descriptions see `/specs/*.md`._

---

## 3 Data & Control Flow

1. **Task Intake** → Worker agent receives task event on Message Bus.  
2. **Goal Framing** → Meta‑Reasoner queries Dialectical Reasoner for class‑alignment scores.  
3. **Simulation Loop**  
   1. Forward World Model generates candidate plans.  
   2. Contradiction Engine flags internal/external tensions.  
   3. Plans re‑ranked by weighted utility (task KPIs × class interest × resource use).  
4. **Governance Hook** (if policy-impacting) → Polycentric Planner opens a vote / deliberation.  
5. **Execution** → Embodied or software tools act; feedback logged to Memory.  
6. **RSI Cycle** → Daily cron job triggers Recursive Self‑Improvement on low‑scoring modules.

---

## 4 Runtime Deployment

```text
Edge Nodes (Robots, Fab Hub, Sensors)
   ⇅ gRPC / MQTT
Micro‑cloud Cluster (K8s or Nomad)
   ├─ agent‑core pods (layer 1/2)
   ├─ world‑model TPU/GPU service (layer 3)
   ├─ governance API (layer 5)
   └─ rsi‑worker (layer 4, on isolated namespace)
Cold‑storage MinIO → log & checkpoint archive
```
Air‑gapped upgrade lanes guarantee RSI cannot hot‑patch security boundaries.

## 5 Security & Alignment
Value‑Lock: Governance layer can veto any RSI patch via signed quorum.

Tension Budget: Contradiction Engine exposes a public “tension score” dashboard—spikes trigger human review.

Rollback Windows: All self‑mods keep 30‑day binary diff history; rollback is O(1).

## 6 Dev & Ops Notes
Languages: Python 3.12 (core), Rust (real‑time agents), Julia (simulation kernels).

Models: Mix of open LLMs (Ollama, Mistral‑7B‑MoE) and local quantized vision models.

Testing: Each module ships with dialectical unit tests (assert class‑interest monotonicity).

Licensing: AGPL‑v3 + Commons Clause for any monetization that re‑introduces wage labor.

## 7 Roadmap Check‑in

 Milestone 	 Target Quarter 	 Status 
v0.5—Contradiction Engine MVP	 Q3‑2025	 🟡 design locked, coding starts
v0.6—World Model in sim farm	 Q4‑2025	 ⚪ pending data collection
v0.7—Governance Alpha (800‑member commune test)	 Q1‑2026	 ⚪
v1.0—AGI‑threshold demo	 Q4‑2026	 ⚪

## 8 ASCII Layer Diagram

┌───────────────────────────────────────────────────────────────┐
│ 6  Dialectical Reasoner & Class‑Consciousness Model          │
├───────────────────────────────────────────────────────────────┤
│ 5  Polycentric Governance Planner                            │
├───────────────────────────────────────────────────────────────┤
│ 4  Recursive Self‑Improvement System                         │
├───────────────────────────────────────────────────────────────┤
│ 3  Meta‑Reasoner + Forward World Model                       │
├───────────────────────────────────────────────────────────────┤
│ 2  Memory, Embodiment, Tool Manager                          │
├───────────────────────────────────────────────────────────────┤
│ 1  Base SAAF‑OS Framework & Distributed Agents               │
└───────────────────────────────────────────────────────────────┘

## 9 Marxist Rationale
SAAF‑OS abstracts socially necessary planning labor into a commons‑owned cybernetic system, undermining the value‑form by:

Converting human toil into free creative activity (automation as liberation).

Embedding class‑interest accounting directly in the reward function.

Keeping self‑improvement subordinate to communal deliberation, preventing capitalist runaway.