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

Results can be reproduced using the code in `baselines/vlm_fusion/`:

1. `distilbert.ipynb`: fine-tune DistilBERT on captions and export test logits to `results/bert_logits_{mini,full}_test.pth`.
2. `extract_logits.ipynb`: extract BEiT test logits with `hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224` (via `timm`); saves under `features/beit-224-{mini,full}/`.
3. `fusion.ipynb`: mean-logit fusion and Top-1 / Top-3 / macro-F1 on the test split. Use `variant = "mini"` for FungiTastic-M and `variant = "full"` for the full benchmark.
---

## Related

- [Closed-set Models](closed.md)
- [Data Modalities: Captions](../data/captions.md)
