# Recursive Self‑Improvement (RSI) System – Spec v0.1

## 1 Goal
Continuously boost SAAF‑OS performance while respecting governance vetoes and rollback guarantees.

## 2 Pipeline
1. **Metrics Collector** → module KPIs.  
2. **Fitness Assessor** → rank modules < threshold.  
3. **Patch Generator**  
   * NAS for NN modules (Evo‑NAS).  
   * LLM‑driven code refactor for Python/Rust.  
4. **Patch Tester** → sandbox run (unit + integration).  
5. **Governance Gate** → multi‑sig approval if scope ≥ policy level.  
6. **Deployer** → blue‑green rollout; keep previous version 30 days.

## 3 API
```python
evaluate_module(name: str) -> ScoreMap
propose_modification(name: str,
                     target_score: float) -> PatchPlan
apply_patch(patch: PatchPlan,
            force: bool = False) -> bool
```
## 4 Storage
Git‑like object store with content‑addressed blobs; metadata in Postgres.

## 5 Security
Patches signed by RSI keypair.

Deployer verifies sig + governance JWT.

## 6 Testing
Synthetic “bug farm” ensures RSI can auto‑fix ≥ 70 % issues.

## 7 Open Issues
Trust calibration of LLM code‑rewrites.