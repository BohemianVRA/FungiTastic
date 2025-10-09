FungiTastic is a **realistic ML benchmark**: beyond images, each observation aligns **visual and contextual predictors** (metadata, captions, satellite, climate etc.). This design addresses gaps in image-only FGVC/few-shot benchmarks by enabling **fusion models**, **temporal robustness** checks, and **part-aware** evaluation on the *same specimens*.

### Modalities at a glance and why they matter
Unlike classic FGVC datasets that assume an image is enough, FungiTastic provides context that often *decides* the label:

- 🖼️ **Photographs** (+ 📝 **captions**): standard image-based data accompanied by captions (MOLMO-7b-generated) to support **vision-language** classification and retrieval.
- 🧾 **Metadata & hierarchical labels**: date, location, habitat, substrate, toxicity, hierarchy — essential for **separation look-alikes** and studying **cost-sensitive** errors.
- 🛰️ **Remote sensing context** (4 band, 64×64 @10m resolution): local environment around the site; enables **geospatial priors** unavailable in image-only benchmarks.
- 🌦️ **Climate time series** (20y temps/precip + 19 bioclims): long-term signals for **temporal shift** and **distribution modeling** beyond static splits.
- ✂️ **Segmentation masks** (Mini only): Annotation masks of 5 body parts, i.e., caps/gills/stems/pores/rings; allowing **part-aware models** and **explainability**, rarely available in FGVC suites.

> For bigger detail, see  [Metadata](data/metadata.md) • [Captions](data/captions.md) • [Satellite](data/satellite.md) • [Climate](data/climate.md) • [Masks](data/masks.md)

### FungiTastic subsets; built for different research goals
FungiTastic ships predefined subsets so you can balance **scale, rarity, and annotation depth** (a limitation in many prior benchmarks that offer only one).

| Subset                  | Observ. | Images | Species | Why it exists / Primary use |
|-------------------------|--------:|-------:|---:|---|
| **FungiTastic**         |   ≈350k |  ≈630k | 4 507 | **Comprehensive** closed/open-set classification, **multimodal fusion**, and **chronological shift** on one dataset (incl. a DNA-verified test slice). |
| **FungiTastic Mini**    |  36 287 | 67 848 | 253  | **Fast iteration** with **part masks** (~70k) for segmentation and **part-aware** fine-grained recognition; smaller but **richer labels** than typical FGVC sets. |
| **FungiTastic FewShot** |   6 391 | 12 015 | 2 427  | **True long-tail few-shot**: species selected by **<5 train images**, preserving natural imbalance—unlike episodic, class-balanced few-shot suites. |

> Coverage note: metadata, captions, climate, and satellite are available for virtually all observations; **masks** are provided in **Mini**.

### Split protocol — why chronological (vs. static random splits)
To measure **generalization over time and space** (seasonality, community drift, sensors, regions), we split by observation year rather than random sampling (typical of many FGVC/few-shot sets):

- **Train:** ≤ 2021  
- **Validation:** 2022  
- **Test:** 2023  

Task specifics, aligned to ML practice:
- **Few-shot (FS):** species go to FS if **training images < 5**; val/test remain chronological → tests **rare-class learning** without synthetic balancing.  
- **Open-set:** train on seen species; val/test include **held-out species** labeled *unknown* → evaluates **novelty/unknown detection**.  
- **Cost-sensitive:** errors weighted by risk (e.g., edible vs. poisonous) → reflects **application stakes**, not just accuracy.

### How to choose — quick, practical guidance
These recommendations reflect FungiTastic’s design goals and where prior datasets fall short:

- **Strong image baseline quickly** → start with **Mini** (fast, part masks), then scale to **Full** for robustness under shift.  
- **Rare-class / few-shot methods** → use **FS** (naturally long-tailed) and report few-shot metrics.  
- **Multimodal fusion** (image + context) → **Full** with metadata + satellite + climate (image-only sets cannot test this).  
- **Explainability / parts** → train segmentation on **Mini** and connect parts to downstream classifiers.

> Next: see **[Benchmarks](../benchmarks.md)** for task definitions and metrics, and **[Download Guide](../usage/download.md)** to grab subsets and scripts.