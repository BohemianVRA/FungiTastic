<p align="center">
  <a href="https://github.com/bohemianvra/FungiTastic"><img src="https://img.shields.io/github/stars/bohemianvra/FungiTastic" alt="Stars"></a>
  <a href="https://github.com/bohemianvra/FungiTastic/issues"><img src="https://img.shields.io/github/issues/bohemianvra/FungiTastic" alt="Issues"></a>
  <a href="https://github.com/bohemianvra/FungiTastic/pulls"><img src="https://img.shields.io/github/issues-pr/bohemianvra/FungiTastic" alt="Pull Requests"></a>
  <a href="https://github.com/bohemianvra/FungiTastic/blob/main/LICENSE"><img src="https://img.shields.io/github/license/bohemianvra/FungiTastic" alt="License"></a>
</p>
# 🍄 Welcome to FungiTastic!

**FungiTastic** is a large-scale, expert-verified, **multi-modal benchmark** for **machine learning**.  
It’s designed to stress-test models under realistic conditions—fine-grained classes, long tails, temporal and geographic shift, and multi-modal fusion (images, metadata, climate, etc.).

The example below illustrates why this benchmark is exceptionally challenging for ML: cross-species differences can be subtle, while within a species the appearance varies widely with age, lighting, and habitat.

<p align="center">
  <img src="./assets/intra_inter.svg" alt="Intra- and inter-class visual similarity across selected fungi species" width="95%">
</p>

**Figure 1. Intra- and inter-class visual similarity.** Examples across **nine species** from **three families** show subtle cross-species differences and large within-species variation, capturing the fundamental challenge that FungiTastic offers to benchmark ML methods.

---

##  Dataset Overview

**FungiTastic** is a large-scale, multi-modal dataset with more than 600k images** across **~350k observations** of around **5,000+ species** collected over **20+ years**. It is purpose-built for **fine-grained recognition**, **domain adaptation**, **multi-modal learning**, **open-set / few-shot** settings, and **interpretability**.

To make the dataset immediately usable for different research goals, we provide **predefined subsets** that trade off scale, difficulty, and annotation depth:

- **Full (FungiTastic):** ~600k images with all modalities — best when you need **maximum coverage** for *closed-/open-set classification* and **novel class discovery** at scale.
- **Mini (FungiTastic–M):** ~70k images from 6 common genera with **part segmentation masks** — ideal for **fast prototyping**, **segmentation**, and **explainable, part-aware models**.
- **Few-shot (FungiTastic–FS):** ~12k images where species have **<5 training samples** — targeted for **few-shot** and **rare-class** learning under extreme data scarcity.

> Need details, stats, or file layout? See the **[Dataset page](./dataset.md)**.

![FungiTastic Example](assets/observation.png)
**Figure2:** A Fungi observation represents a real-world record of a fungus and includes one or more photos of the fungi specimen [🟩] with expert-verified labels (sometimes including images of spores) and rich contextual data: captions [🟦], metadata [🟧], geospatial [🟫], and climatic time-series [🟦]. For a subset (~70k images), ground-truth body part masks [🟥] are included.

---

## Use Cases?

FungiTastic is built to **mirror real field conditions**: rare species, shifting seasons, noisy observations, and evolving taxonomies. With **ready-to-use splits, baselines, and multimodal inputs**, you can prototype quickly and evaluate fairly across tasks that matter in practice.

- **Fine-grained classification** (closed-set, open-set)
- **Few-shot learning** of rare species 
- **Multi-modal** (mix visual, tabular, geospatial, text)
- **Domain adaptation & temporal shift** (yearly, seasonal, geographic)
- **Vision-language modeling** (rich captions)
- **Semantic/instance segmentation**
- **Cost-sensitive classification** (e.g., edible vs. poisonous)
- **Multi-task learning** 

and many more, with predefined benchmarks reflecting real-world challenges and use cases.


> See [Benchmarks](./benchmarks.md) for benchmark challenges and usage.

---

## Resources

| Resource             | Description                             | Link                                                                                                  |
|----------------------| --------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 📄 Publication       | CVPR 2025 (FGVC Workshop) | [arXiv Paper (PDF)](https://arxiv.org/pdf/2408.13632)                                                 |
| 🧠 GitHub Repository | Code, loaders, scripts, and baselines   | [FungiTastic Repo](https://github.com/bohemianvra/FungiTastic)                                        |
| 🚀 Starter Notebooks | Baseline pipelines and scripts          | [Kaggle Code Notebooks](https://www.kaggle.com/datasets/picekl/fungitastic/code)                      |
| 📦 Download          | How to access subsets and modalities    | [Download & Usage Guide](https://bohemianvra.github.io/FungiTastic/usage/download/)                   |

---

## Citation 
If you use FungiTastic in your research, please cite the following:
  ```
  @InProceedings{Picek_2025_CVPR,
      author    = {Picek, Lukas and Janouskova, Klara and Cermak, Vojtech and Matas, Jiri},
      title     = {FungiTastic: A Multi-Modal Dataset and Benchmark for Image Categorization},
      booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
      month     = {June},
      year      = {2025},
      pages     = {2046-2056}
  }
  ```

---

_Enjoy!_ 🍄