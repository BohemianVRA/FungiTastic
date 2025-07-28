# ⏳ Chronological / Domain Shift

## Overview
Biological data are not i.i.d.—they change over time due to **seasonality, climate, and location**.  
FungiTastic’s temporal splits let you test **robustness to domain shifts**.

---

## Use Cases
- Domain adaptation, continual learning, and test-time adaptation research
- Models deployed in changing or dynamic environments

---

## Data & Splits
- Train: Observations up to 2021
- Validation: Observed in 2022
- Test: Observed in 2023
- Splits follow real-world chronology, including natural class and distribution shifts

---

## Evaluation Protocol
- **Metrics:** Standard classification metrics (Top-1, F1) or task-specific as appropriate
- Can be applied to closed/open/few-shot tasks

---

## Baselines & Results
- Baselines demonstrate substantial accuracy drops across years
- Encourages research into robust, adaptive models

---

## Quick Start
- See [usage/evaluation.md](../usage/evaluation.md) for scripts and analysis
- Download temporal splits from [Kaggle](https://www.kaggle.com/datasets/picekl/fungitastic)

---

## Related
- [Closed-set Classification](closed.md)
- [Open-set Classification](open.md)
