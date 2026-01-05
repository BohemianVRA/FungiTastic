# 🧪 Baselines & Models Overview

FungiTastic includes **strong, diverse baselines** for all benchmark tasks.  
You’ll find:

- Ready-to-use models with pre-trained weights: classic CNNs, modern transformers, and multimodal models.
- Results on all main benchmarks.
- Evaluation scripts to reproduce the results.
- Demo notebooks with example usage.

---

## 📋 Model Categories

| Task                    | Model Page                        | Baseline Types           |
|-------------------------|-----------------------------------|--------------------------|
| Closed-set Classification | [Closed-set Models](closed)   | CNN, ViT, BEiT, etc.     |
| Few-shot Learning         | [Few-shot Models](few_shot)   | CE, prototypes, NN       |
| Open-set Classification   | [Open-set Models](open)       | Softmax, logit, DINOv2   |
| Vision-Language Fusion    | [VLM Fusion](vlm)             | DistilBERT, Fusion       |

Segmentation models are covered in the [Segmentation Baselines](segmentation.md).

---

## 🚀 Getting Started

- All model scripts/notebooks are in the [GitHub repo](https://github.com/bohemianvra/FungiTastic/)
- Download checkpoints and splits from [Kaggle](https://www.kaggle.com/datasets/picekl/fungitastic)
- Benchmark protocols: see [usage/training.md](../usage/training.md) and [usage/evaluation.md](../usage/evaluation.md)

---

**Choose a model page for more details on architectures, training, and results.**
