# 🌍 Open-set Models

## Approaches

- **Maximum Softmax Probability (MSP):** Detects OOD via classifier confidence
- **Maximum Logit Score (MLS):** Uses the largest logit for OOD detection
- **Nearest Mean (NM):** Distance to closest class mean in embedding space
- **Backbones:** BEiT-Base/p16 (trained), DINOv2 (pretrained features)

---

## Results Summary

| Backbone        | Method | TNR@95 | AUC    |
|-----------------|--------|--------|--------|
| BEiT-Base/p16   | MLS    | 27.7   | 83.9   |
| DINOv2          | MLS    | 36.9   | 74.5   |
| DINOv2          | MSP    | 32.5   | 82.4   |

(Full results in the paper and repo.)

---

## Code & Usage

- Open-set detection scripts: [`examples/openset_classification.py`](https://github.com/bohemianvra/FungiTastic/)
- Data splits: [Kaggle](https://www.kaggle.com/datasets/picekl/fungitastic)

---

## Related

- [Closed-set Models](closed.md)
- [Few-shot Models](few_shot.md)
