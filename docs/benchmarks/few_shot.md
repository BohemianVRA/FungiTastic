# 🎯 Few-shot Learning

## Overview
Rare species are the norm, not the exception!  
The few-shot benchmark isolates species with **fewer than five training samples**. The task: **Can your model generalize with minimal supervision?**

---

## Use Cases
- Identification of rare, endangered, or newly observed fungi
- General few-shot learning research (prototypical networks, embedding-based methods, etc.)

---

## Data & Splits
- Only species with <5 training images are included.
- Standard temporal splits: train (up to 2021), validation (2022), test (2023).

---

## Evaluation Protocol
- **Primary Metric:** Top-1 accuracy
- **Secondary Metrics:** Macro F1-score, Top-3 accuracy

---

## Baselines & Results
Includes nearest-neighbor, centroid-prototype, and standard classifiers on strong feature extractors (CLIP, BioCLIP, DINOv2, BEiT).

**See [Baselines & Models](../models/few_shot.md) for results and example code.**

---

## Quick Start
- Few-shot split provided in the dataset package and [Kaggle](https://www.kaggle.com/datasets/picekl/fungitastic).

---

## Related
- [Open-set Classification](open.md)
- [Closed-set Classification](closed.md)
