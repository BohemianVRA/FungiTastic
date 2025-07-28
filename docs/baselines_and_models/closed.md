# 🔒 Closed-set Classification Models

## Approaches

- **CNNs:** ResNet-50, ResNeXt-50, EfficientNet-B3, EfficientNetV2-B3, ConvNeXt-Base
- **Transformers:** ViT-Base/p16, Swin-Base/p4w12, BEiT-Base/p16

All models are trained with recommended augmentations and hyperparameters from the paper.

---

## Results Summary

| Model          | Top-1 (%) | Top-3 (%) | Macro F1 (%) |
|----------------|-----------|-----------|--------------|
| ResNet-50      | 61.7      | 79.3      | 35.2         |
| EfficientNet-B3| 61.9      | 79.2      | 36.0         |
| ConvNeXt-Base  | 66.9      | 84.0      | 41.0         |
| ViT-Base/p16   | 68.0      | 84.9      | 39.9         |
| BEiT-Base/p16  | **69.1**  | **84.6**  | **42.3**     |

(See paper and repo for full table.)

---

## Code & Weights

- [GitHub – Training scripts](https://github.com/bohemianvra/FungiTastic/)
- [Pre-trained models on HuggingFace](https://huggingface.co/collections/BVRA/fungitastic-66a227ce0520be533dc6403b)

---

## Related

- [Open-set Models](open.md)
- [Few-shot Models](few_shot.md)
