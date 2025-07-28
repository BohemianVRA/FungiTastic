# ✂️ Segmentation Task

## Overview
Accurate segmentation of fungal body parts (caps, gills, stems, etc.) is crucial for **fine-grained recognition and interpretability**.  
FungiTastic–Mini includes **part segmentation masks** for a subset of species.

---

## Use Cases
- Automated species identification pipelines
- Explainable AI and morphological analysis
- Training and evaluating segmentation models on natural data

---

## Data & Splits
- FungiTastic–Mini: Images with expert-checked segmentation masks
- Classes: cap, gills, pores, ring, stem (and background)
- Split: train/val/test following Mini subset

---

## Evaluation Protocol
- **Semantic Segmentation:** mean Intersection over Union (mIoU)
- **Instance Segmentation:** mean Average Precision (mAP)

---

## Baselines & Results
- Baseline: Zero-shot pipeline (GroundingDINO + SAM)
- Baseline IoU: ~89% (but misses small/atypical parts)
- Further improvements possible with domain-specific models

**See [Baselines & Models](../models/segmentation.md) for details.**

---

## Quick Start
- Download Mini subset and annotation masks
- Use sample notebooks/scripts in [usage/training.md](../usage/training.md)

---

## Related
- [Photographs & Captions](../data/photographs.md)
- [Closed-set Classification](closed.md)
