
# Latent Encoding Specification for ULS (`latent_encoding_spec.md`)

> This document defines the unified input state representation `u_t` and its encoding into the Unified Latent Space `z_t ∈ 𝓛` for the SAAF‑OS architecture. The encoded state supports contradiction gradients, recursive self-modification, and class-conscious planning.

---

## 🧠 Overview

The Unified Latent Space (ULS) relies on an encoder function:

```
Encoder: u_t → z_t ∈ 𝓛 ⊂ ℝⁿ
```

The goal is to construct `z_t` from multimodal signals spanning embodiment, socio-economic states, contradiction awareness, memory, and system introspection.

---

## 🔹 1. Physical Embodiment State

| Feature              | Type              | Notes                                    |
|---------------------|-------------------|------------------------------------------|
| Joint positions      | ℝⁿ vector         | Normalized per robot type                |
| Force/torque sensors | ℝⁿ vector         | Captures tool-environment interaction    |
| Tool status flags    | Binary/Categorical| `tool_engaged`, `tool_id`, etc.          |
| Energy usage         | Float             | Per device snapshot                      |
| Environment map      | Image/Voxel grid  | Encoded via CNN or VQ-VAE                |

---

## 🔹 2. Economic / Social State

| Feature                | Type              | Notes                                    |
|------------------------|-------------------|------------------------------------------|
| Class position         | One-hot/embedding | From Dialectical Reasoner ontology       |
| Labor-time logs        | Matrix            | Time-series: agent × task                |
| Surplus extraction     | Float             | Output vs. retention by class            |
| Commons access index   | Float             | % communal vs. private consumption       |
| Alienation index       | Float             | From intention vs. actual labor deviation|
| Governance status      | Categorical       | e.g., active vote, quorum, etc.          |

---

## 🔹 3. Contradiction Signals

| Feature                    | Type             | Notes                                     |
|----------------------------|------------------|-------------------------------------------|
| Top-K contradiction tensors| Vector set       | ∇₍g₎ L_contradiction(z) per contradiction |
| Tension graph deltas       | Float matrix     | Δ tension from Contradiction Engine       |
| Open synthesis paths       | Sequence/graph   | Representation of unresolved contradictions|
| Conflict signal deltas     | Matrix           | Δ(resource, class vector, goals)          |

---

## 🔹 4. Memory Embeddings

| Feature                  | Type             | Notes                                   |
|--------------------------|------------------|-----------------------------------------|
| Retrieved episodes       | List of `z_i`    | K-nearest by geodesic in 𝓛              |
| Token path tracebacks    | List of tokens   | Memory-salient symbolic traces          |
| Latent plan prototypes   | Latent vectors   | Prior planning solutions for reuse      |

---

## 🔹 5. Meta-Cognitive State (RSI)

| Feature                  | Type             | Notes                                      |
|--------------------------|------------------|--------------------------------------------|
| Module fitness scores    | Real vector      | RSI evaluator outputs                      |
| Patch approval vector    | Binary vector    | Passed or vetoed modules                   |
| Value-drift indicators   | Float            | Δ(policy vector) over time                 |
| Rollback status          | Categorical      | Is system recovering from vetoed patch     |

---

## 🧩 Unified Encoder Pipeline

```python
def encode_state(u_t) -> z_t:
    z_phys = CNN_RNN(u_t['robotics'], u_t['environment_map'])
    z_soc = MLP(u_t['census'], u_t['value_vector'], u_t['governance'])
    z_contra = Transformer(u_t['contradictions'], u_t['tension_graph'])
    z_mem = MemoryRetriever(u_t['episodic_trace'], u_t['symbol_path'])
    z_meta = MLP(u_t['rsi_signals'])

    # Hyperdimensional Fusion Layer
    z_t = HyperdimensionalFusion([z_phys, z_soc, z_contra, z_mem, z_meta])
    return z_t
```

This encoding respects manifold regularization and supports downstream computation of:
- Riemannian gradients ∇₍g₎ L_contradiction
- Feasibility heuristics for synthesis planning
- Shared projections P(z_t) for consensus protocols

---

## ✅ ULS Integration Summary

- **Input Space**: `u_t` spans physical, symbolic, economic, memory, and introspective domains
- **Output Space**: `z_t ∈ 𝓛 ⊂ ℝⁿ`, ULS-compatible
- **Supports**: Planning, contradiction scoring, recursive self-modification, and value-aligned governance
