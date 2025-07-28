# ⚖️ Cost-sensitive Classification

## Overview
Not all mistakes are equal!  
In FungiTastic, confusing a poisonous species for an edible one can have severe consequences, whereas the reverse is less dangerous.  
**This benchmark evaluates models under custom loss/cost functions that reflect real-world priorities.**

---

## Use Cases
- Safety-critical systems (e.g., mobile ID for foragers)
- Research into cost-sensitive, risk-aware, or calibrated classifiers

---

## Data & Splits
- Uses standard FungiTastic splits (temporal)
- Poisonous/edible labels and additional risk metadata provided

---

## Evaluation Protocol
- **Metrics:** Weighted error rates, custom cost matrices (see [Appendix in the paper](https://arxiv.org/pdf/2408.13632))
- You may define your own costs for specific errors

---

## Baselines & Results
- Example: False positive edible has high penalty; false positive poisonous has lower penalty
- See example code and detailed description in [Baselines & Models](../models/closed.md) and [evaluation protocols](../usage/evaluation.md)

---

## Quick Start
- Use provided cost matrix and baseline scripts
- Extend to new tasks with your own risk functions

---

## Related
- [Closed-set Classification](closed.md)
