# 🔤 Vision-Language Fusion Models

## Approaches

- **Text Encoder:** DistilBERT fine-tuned for species classification using generated captions
- **Fusion:** Simple ensemble – average logits from image and text models

---

## Results Summary

| Model       | Top-1 (%) | Top-3 (%) | Macro F1 (%) |
|-------------|-----------|-----------|--------------|
| DistilBERT  | 31.2      | 50.2      | 11.5         |
| BEiT-Base   | 67.3      | 83.3      | 40.5         |
| Fusion      | **67.7**  | **83.8**  | 39.8         |

- Fusion boosts accuracy on common classes, but language model alone is weaker than vision.

---

## Code & Usage

- Caption generation and fusion code: [`examples/vlm_fusion.py`](https://github.com/bohemianvra/FungiTastic/)
- Pre-computed captions included in the dataset

---

## Related

- [Closed-set Models](closed.md)
- [Data Modalities: Captions](../data/captions.md)
