# 🌍 Open-set Classification

## Overview
**Real-world fungal datasets are never closed:** New and previously unseen species appear in the wild each year.  
The open-set benchmark asks: **Can your model detect when a specimen belongs to an unknown class?**

---

## Use Cases
- Discovering new or rare species
- Deployed models that must avoid confident mistakes on out-of-distribution data

---

## Data & Splits
- **Training:** Species up to end of 2021.
- **Validation:** Contains species first observed in 2022.
- **Test:** Contains species first observed in 2023.
- "Unknown" label is used for new classes.

---

## Evaluation Protocol
- **Primary Metric:** Area Under ROC Curve (AUC)
- **Secondary Metric:** True Negative Rate @ 95% True Positive Rate (TNR95)

---

## Baselines & Results
Includes Max Softmax Probability, Max Logit Score, and Nearest Mean approaches, evaluated on both supervised and pre-trained (DINOv2, BEiT) backbones.

**Detailed results and code: [Baselines & Models](../models/open.md)**

---

## Quick Start
- Data splits and scripts available in the repo and [Kaggle](https://www.kaggle.com/datasets/picekl/fungitastic).
- Tutorial: [usage/evaluation.md](../usage/evaluation.md)

---

## Related
- [Closed-set Classification](closed.md)
- [Few-shot Learning](few_shot.md)
