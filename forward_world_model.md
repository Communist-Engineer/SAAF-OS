
---

### **`/specs/forward_world_model.md`**
```markdown
# Forward World Model – Spec v0.1

## 1 Purpose
Provide differentiable simulations for planning, counterfactual reasoning, and risk analysis across physical and socio‑technical domains.

## 2 Architecture
* **Encoder** (multimodal V‑Transformer) → latent `z`  
* **Dynamics Core** (GNN‑Transformer hybrid) : `(z, action) ⟶ z`  
* **Decoder** (domain‑specific heads) → observations, metrics.

## 3 Training Pipeline
1. Collect trajectories from robots, IoT, economic ledgers.  
2. Self‑supervised learning (masked‑step prediction).  
3. Fine‑tune with RL‑HF on task performance.

## 4 API
```python
simulate(actions: ActionSeq,
         state: State,
         horizon: int = 32,
         samples: int = 8) -> List[Trajectory]

generate_plan(goal: Goal,
              constraints: ConstraintSet) -> ActionGraph

evaluate_plan(plan: ActionGraph,
              crit: MetricSet) -> UtilityMap
```
## 5 Uncertainty
Ensemble of N=5 lightweight cores; output entropy + epistemic variance.

## 6 Persistence
Latest weights in model‑registry; old checkpoints in cold storage (keep 90 days).

## 7 Hardware
v0.1 targets 1×A100 or 2×RTX 6000 Ada; TPU v5‑lite optional.

## 8 Testing
One‑step MSE < 0.02 on validation sim.

End‑to‑end plan success ≥ 85 % in “farm‑bot” scenario.

## 9 Open Issues
Cross‑domain transfer degradation.

Real‑time inference on edge GPUs.