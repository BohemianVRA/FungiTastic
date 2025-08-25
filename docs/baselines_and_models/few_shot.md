# 🎯 Few-shot Models

## Approaches

- **Classic:** Supervised cross-entropy classifiers (CNN/ViT/BEiT)

With frozen feature embeddings of foundation models (CLIP, BioCLIP, DINOv2)
- **Nearest-neighbor (1-NN):** Standard nearest-neighbour classification with 1 NN.
- **Centroid-prototype:** Nearest-prototype classification with mean class embeddings.

---

## Results Summary

| Method             | Top-1 (%) | Top-3 (%) |
|--------------------|-----------|-----------|
| CLIP 1-NN          | 6.1       | –         |
| CLIP centroid      | 7.2       | 13.0      |
| DINOv2 1-NN        | 17.4      | –         |
| DINOv2 centroid    | 17.9      | 27.8      |
| BioCLIP centroid   | 21.8      | 32.6      |
| BEiT/ViT (CE loss) | 11–19     | 17–29     |

(See [paper](https://arxiv.org/pdf/2408.13632) for full table.)

---

## Code & Usage

- Example few-shot scripts: [`examples/fewshot_classification.py`](https://github.com/bohemianvra/FungiTastic/)
- Notebook tutorial: [usage/training.md](../usage/training.md)

---

## Related

- [Closed-set Models](closed.md)
- [Open-set Models](open.md)
