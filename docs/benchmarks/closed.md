# 🔒 Closed-set Classification

## Overview
The classic supervised task: **Given an image, assign it to one of the species present in the training set**.  
This is especially challenging in FungiTastic due to its fine-grained labels, long-tailed class distribution, and domain shifts.

---

## Use Cases
- Automated fungal species identification for field guides, citizen science apps, etc.
- Baseline for comparing closed-world vision models.

---

## Data & Splits
- Uses the main FungiTastic split: training (all data up to end of 2021), validation (2022), test (2023).
- Only species present in the training set are evaluated.

---

## Evaluation Protocol
- **Primary Metric:** Top-1 accuracy
- **Secondary Metrics:** Macro F1-score, Top-3 accuracy

---

## Baselines & Results
State-of-the-art models (e.g., ResNet, EfficientNet, ViT, BEiT) are provided as benchmarks.
- Transformer architectures outperform classic CNNs, but all models struggle due to dataset complexity.

**See [Baselines & Models](../models/closed.md) for scripts and results.**

---

## Quick Start
- Download split files from [Kaggle](https://www.kaggle.com/datasets/picekl/fungitastic).
- Use [usage/training.md](../usage/training.md) for a quickstart on model training.

---

## Related
- [Open-set Classification](open.md)
- [Few-shot Learning](few_shot.md)
