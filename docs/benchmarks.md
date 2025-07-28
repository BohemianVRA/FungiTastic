# 🏆 FungiTastic Benchmarks

FungiTastic is designed to **push the limits of machine learning in real-world biological classification** through a comprehensive suite of benchmarks. These challenges reflect the unique, dynamic, and fine-grained nature of fungal data in the wild.

---

## Why Multiple Benchmarks?

- **Realistic evaluation:** Biological data are messy—seasonal, imbalanced, shifting, and full of new discoveries.
- **Diverse ML challenges:** Tackle closed-set, open-set, few-shot, domain adaptation, segmentation, and more.
- **Multi-modal & cost-sensitive:** Go beyond images, leverage metadata and context, and reason about real-world consequences (e.g., edible vs. poisonous).

---

## 🧩 Benchmark Tasks

Below is an overview of all the supported benchmarks.  
**Click the task name to jump to the dedicated section and learn more about the data splits, evaluation metrics, and baseline results.**

---

| Task                       | Description                                                                 | Link                        |
|----------------------------|-----------------------------------------------------------------------------|-----------------------------|
| Closed-set Classification  | Standard fine-grained classification—species from a fixed set.              | [Closed-set](closed.md)     |
| Open-set Classification    | Identify when an observation belongs to a new, unseen species.              | [Open-set](open.md)         |
| Few-shot Learning          | Recognize rare species with only a handful of training samples.              | [Few-shot](few_shot.md)     |
| Chronological (Domain Shift) | Handle distribution shifts—train/test splits follow real observation dates. | [Chronological](chronological.md) |
| Cost-sensitive Classification | Penalize mistakes by their real-world impact (e.g., toxic/edible errors). | [Cost-sensitive](cost_sensitive.md) |
| Segmentation               | Detect and segment key body parts of fungi for fine-grained recognition.     | [Segmentation](segmentation.md)   |

---

## 🔍 Benchmark Descriptions

### 🔒 Closed-set Classification
* **Goal:** Assign the correct species label to each image from a fixed set of classes.
* **Highlights:** Long-tailed, visually similar classes; strong baseline models provided.
* 👉 [Read more…](closed.md)

---

### 🌍 Open-set Classification
* **Goal:** Detect when a test observation is from a species **not seen during training**.
* **Use-case:** New/rare fungi are continually discovered in nature—your model should know when it’s unsure.
* 👉 [Read more…](open.md)

---

### 🎯 Few-shot Learning
* **Goal:** Correctly classify species with fewer than five training samples.
* **Highlights:** Rewards ability to generalize from very little data—crucial for rare/under-observed species.
* 👉 [Read more…](few_shot.md)

---

### ⏳ Chronological / Domain Shift
* **Goal:** Evaluate robustness to **distribution shift over time** (e.g., due to seasonality, climate, or new locations).
* **Highlights:** Train on older data, validate/test on newer years; reflects real-world deployment scenarios.
* 👉 [Read more…](chronological.md)

---

### ⚖️ Cost-sensitive Classification
* **Goal:** Minimize "real-world" error costs. E.g., confusing a poisonous with an edible mushroom is much worse than the reverse.
* **Highlights:** Supports research into non-standard losses and safety-aware AI.
* 👉 [Read more…](cost_sensitive.md)

---

### ✂️ Segmentation
* **Goal:** Identify and segment morphological parts (caps, stems, gills, etc.) for selected species/groups.
* **Highlights:** Supports interpretable and part-aware models, with hand-checked masks in the Mini subset.
* 👉 [Read more…](segmentation.md)

---

## 📈 Baselines & Results

For each benchmark, you’ll find:
- Data splits & preparation instructions
- Recommended metrics
- Baseline architectures and results
- Download links for splits & scripts

**See the [Baselines & Models](../models/index.md) section for implementation details and ready-to-use checkpoints.**

---

## 📎 Quick Start

- See [How to Use](../usage/training.md) for step-by-step tutorials.
- Review [Evaluation Protocols](../usage/evaluation.md) for metric details and leaderboard submission formats.

---

> 💡 Want to add a new benchmark or task? [Open an issue](https://github.com/bohemianvra/FungiTastic/issues) or submit a pull request!