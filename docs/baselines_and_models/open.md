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


Results can be reproduced using the code in `baselines/open_set/`:

1. `extract_features.ipynb` : extracts features for downstream tasks. The notebook loads both DINOv2 and the Hugging Face BEiT checkpoint fine-tuned on FungiTastic (`hf-hub:BVRA/beit_base_patch16_384.in1k_ft_fungitastic_384` via `timm`) and saves train/val/test features and logits.
2. `dino_linear.ipynb`: trains a simple linear classifier on frozen DINOv2 features so that MSP and MLS can be computed from logits comparable to the closed-set setup.
3. `ood.ipynb`: builds known-vs.-novel score distributions and computes evaluation metrics. Uses helper functions from `ood_utils.py`.


---

## Related

- [Closed-set Models](closed.md)
- [Few-shot Models](few_shot.md)
