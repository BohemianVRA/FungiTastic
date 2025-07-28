# 🗃️ FungiTastic Dataset

FungiTastic is a large-scale, **multi-modal benchmark dataset** for computer vision, machine learning, and biodiversity research, centered around wild fungi observations. It offers expert-labeled data across more than 5,000 species, collected over 20+ years, and is designed to power research in fine-grained recognition, domain adaptation, multi-modal learning, open-set recognition, and more.

---

## 🚀 What’s Inside?

- **Size:** ~350,000 observations, >600,000 images, ~5,000 species.
- **Modalities:** Photographs, satellite data, climate time series, segmentation masks, expert taxon labels, rich metadata, and image captions.
- **Expert Curation:** Species labels are expert-verified; a subset includes DNA-sequenced ground truth.
- **Region:** Predominantly Denmark and Northern Europe.
- **Time span:** 2003–2023.

---

## 🧑‍🔬 What Can You Use It For?

FungiTastic is specifically designed to benchmark and develop advanced ML models for:

- **Fine-grained image classification** (closed-set, open-set)
- **Few-shot learning** and rare species recognition
- **Multi-modal and multi-task learning** (combine visual, tabular, geospatial, and textual data)
- **Domain adaptation and temporal shift** (study seasonal, habitat, and year-to-year distribution changes)
- **Vision-language modeling** (with detailed image captions)
- **Semantic and instance segmentation**
- **Cost-sensitive classification** (e.g., recognizing poisonous vs. edible species)

> **See the [Benchmarks](../benchmarks/closed.md) section for a deep dive on supported challenges.**

---

## 📚 Dataset Subsets

FungiTastic is split into several subsets, each tailored for different research tasks:

### 1. **FungiTastic (Full)**
- The main benchmark set: >346k observations, >4,500 species, with all modalities.
- **Use:** Closed-set & open-set classification, multi-modal modeling.

### 2. **FungiTastic-Mini (FungiTastic–M)**
- A compact subset from 6 challenging genera (e.g., Russula, Amanita, Boletus).
- Includes **body part segmentation masks** for ~70k images.
- **Use:** Fast prototyping, segmentation, few-shot, open-set.

### 3. **FungiTastic–FS (Few-shot)**
- Observations from species with <5 training samples.
- **Use:** Few-shot learning & rare species recognition.

> Details and statistics for each subset are provided in the [Subsets](#fungitastic-dataset-subsets) and [Benchmarks](../benchmarks/closed.md) pages.

---

## 🔗 Available Data Modalities

Each observation can include a combination of:

- **[Photographs & Captions](./photographs.md):** High-quality images (incl. some spore micrographs) and automatic detailed text descriptions.
- **[Taxonomic Labels](./metadata.md):** Full biological hierarchy and toxicity (edible/poisonous).
- **[Body Part Segmentation Masks](./masks.md):** For toadstool-type fungi in the Mini subset.
- **[Tabular Metadata](./metadata.md):** Date, location, habitat, substrate, elevation, land cover, etc.
- **[Remote Sensing Data](./satellite.md):** 64x64 multi-band satellite images at 10m spatial resolution.
- **[Climate Time Series](./climate.md):** 20 years of temperature/precipitation data and bioclimatic variables.

---

## 🗂️ Quick Links to Data Sections

| Data Type              | Description                               | Link                                  |
|------------------------|-------------------------------------------|---------------------------------------|
| Photographs & Captions | Images and generated text descriptions    | [Photographs & Captions](./photographs.md) |
| Taxonomic Metadata     | Labels, toxicity, environment info        | [Metadata](./metadata.md)             |
| Segmentation Masks     | Body part annotations (Mini subset)       | [Segmentation Masks](./masks.md)      |
| Satellite Data         | Local satellite/environmental context     | [Satellite Data](./satellite.md)      |
| Climate Series         | Long-term climate for each location       | [Climate Time Series](./climate.md)   |

---

## 📥 How to Download

Instructions for access and download are provided in the [Dataset Download Guide](../usage/download.md).

---

## 📝 Citations & Further Reading

For detailed methodology, statistics, and baselines, see our [paper](https://arxiv.org/pdf/2408.13632) and the [Citation](../citation.md) page.

---

> Still have questions? Explore the [FAQ](../index.md) or reach out via [GitHub Issues](https://github.com/bohemianvra/FungiTastic/issues).

