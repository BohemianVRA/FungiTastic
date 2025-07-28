# 🧪 Evaluation Protocols

Accurate, fair evaluation is key to FungiTastic!  
This page explains how to evaluate your models for all main benchmarks, including **metrics, submission format, and leaderboard suggestions**.

---

## 1. Standard Metrics

| Task                     | Main Metric(s)                             |
|--------------------------|--------------------------------------------|
| Closed-set Classification| Top-1 Accuracy, Macro F1, Top-3 Accuracy   |
| Open-set Classification  | AUC, TNR@95%TPR                            |
| Few-shot Learning        | Top-1 Accuracy, Macro F1                   |
| Segmentation             | mIoU, mAP                                  |
| Cost-sensitive           | Custom weighted loss, see below            |

---

## 2. How to Evaluate

- Use the provided `*_test.csv` files for each benchmark.
- After inference, output a CSV:
    - For classification: `image_id,predicted_label`
    - For open-set: add a column for `is_unknown` or use "unknown" as a label
    - For segmentation: follow COCO or Pascal VOC format

- Python example (classification):
    ```python
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score

    y_true = [...]  # ground-truth labels
    y_pred = [...]  # your predictions

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    ```

- Open-set: Use ROC/AUC and TNR metrics, see [Open-set Models](../models/open.md)

---

## 3. Cost-sensitive Evaluation

- Download or define a **cost matrix**: higher penalties for dangerous mistakes (e.g., poisonous predicted edible).
- Use sample code from [Baselines & Models](../models/closed.md) or Appendix A in the paper.

---

## 4. Submitting to Leaderboards

- For official challenges or leaderboards, see instructions in [README](../README.md) or [GitHub Issues](https://github.com/bohemianvra/FungiTastic/issues).

---

> Need more? Check [Benchmarks](../benchmarks/index.md) for protocol details or open an issue!
