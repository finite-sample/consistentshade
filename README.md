## Stabilising Neural Networks via a Micro‑Bootstrap Variance Penalty

## 1  Problem

Refitting a multilayer perceptron (MLP) on a fresh bootstrap draw often changes its predictions more than sampling noise alone would justify. Such **re‑fit variance** undermines confidence intervals, decision thresholds, and fair comparisons between deployments.

Goal → *learn a model whose predictions barely move if we retrain on another i.i.d. sample*, while giving up as little out‑of‑sample (OOS) accuracy as possible.

---

## 2  Related work

| Family | Key idea | Gap for stability |
|--------|----------|-------------------|
| **Sharpness‑Aware** (SAM, Entropy‑SGD) | Penalise weight‑space curvature | Targets *weight* sensitivity, not data resample variance |
| **Stochastic‑network consistency** (R‑Drop) | Two dropout masks, KL(logits) | Controls network noise only |
| **EMA Teacher–student** (Mean Teacher) | Student matches EMA weights | Anchors SGD noise, agnostic to bootstrap |
| **Variance‑regularised ERM / χ²‑DRO** | Penalise √Var(loss) | Losses may stay equal while predictions differ |

None directly minimizes *bootstrap prediction variance*.

---

## 3  Our contribution

* **Loss augmentation**

\[
\mathcal{L}
  = \frac{1}{K}\sum_{k=1}^{K}\!\!
        \frac{1}{|\text{idx}^{(k)}|}\sum_{i\in\text{idx}^{(k)}}
               \ell\!\bigl(f_{\theta^{(k)}}(x_i),\,y_i\bigr)
  + \lambda\;
        \frac{1}{B}\sum_{j=1}^{B}\!
            \operatorname{Var}_{k}\!\bigl[f_{\theta^{(k)}}(x_j)\bigr]
\]

  * `idx^(k)`: with‑replacement resample of the mini‑batch  
  * \(K=3\) “shadow” copies share gradients; λ=0.05 in all experiments.

* **Directly targets** the quantity of interest: variance of predictions across bootstrap resamples.

* **Cheap**: 2–3× forward/backward time; zero extra cost at inference (we average copies only for evaluation).

---

## 4  Experimental setup

| Component | Choice |
|-----------|--------|
| Datasets | Synthetic 20‑dim regression · California Housing · Adult Income · German Credit |
| Splits | 75 % train / 25 % test (stratified for classification) |
| Architecture | 2‑layer Dropout MLP (64–128 units) |
| Baseline | Standard ERM with dropout |
| Proposed | + variance penalty, \(K=3\), λ = 0.05 |
| Metrics | Test RMSE / Accuracy · **StabilityRMSE** = √(mean Var\(_{\text{fit}}\)[pred]) |

30 independent fits per condition.

---

## 5  Results

| Dataset | Metric | Baseline | k = 3 (λ=.05) | Δ Error | Δ Stability |
|---------|--------|----------|---------------|---------|-------------|
| Synthetic | RMSE | **23.88** ± 0.37 | 29.46 ± 0.64 | ↑ 23 % | **–38 %** |
| California | RMSE | **0.591** ± 0.005 | 0.598 ± 0.004 | ↑ 1 % | **–26 %** |
| Adult | Acc | **0.826** ± .001 | 0.825 ± .001 | –0.1 pp | **–81 %** |
| Credit | Acc | **0.697** ± .009 | 0.688 ± .006 | –0.9 pp | **–48 %** |

> *StabilityRMSE* is expressed in the same units as the prediction  
> (log‑odds for classification), so lower is better.

---

## 6  Interpretation

* Real‑world datasets show **25–80 % reduction** in bootstrap drift for ≤ 1 pp accuracy loss.  
* Synthetic toy shows the worst‑case cost (+23 % RMSE) when the model has high capacity relative to data size.  
* Stability share of total error drops from 18 % → 14 % (Synthetic) and 29 % → 10 % (Adult).

---

## 7  Positioning vs. baselines

| Criterion | SAM | R‑Drop | χ²‑DRO | **Ours** |
|-----------|-----|--------|--------|----------|
| Direct data‑resample target | ✗ | ✗ | ✗ | ✓ |
| Regression & classification | ✓ | ✗ | ✓ | ✓ |
| Extra train memory | ✓ | ✓ | ✓ | ✗ (*K* copies) |
| No extra inference cost | ✓ | ✓ | ✓ | ✓ |

---

## 8  Future work

* Influence‑function variance estimates → single‑model penalty (no memory hit).  
* Hybrid with EMA Teacher: control both data and optimiser noise.  
* Downstream effect on policy regret and fairness audits.

---

**TL;DR** A micro‑bootstrap variance penalty is a drop‑in, architecture‑agnostic technique that halves prediction instability with minimal accuracy loss on common tabular tasks.

