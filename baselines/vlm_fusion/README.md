# Vision–language fusion baselines

Reproduces **Table 7** from the FungiTastic paper: DistilBERT on Molmo captions, BEiT-Base/p16 image logits, and **mean-logit fusion**.

## Pipeline

1. **Captions** — Generate per-image JSON captions with [`metadata_extraction/captions/`](../../metadata_extraction/captions/) (Molmo-7B). Point `caption_dir` in `distilbert.ipynb` to a folder of `<filename>.json` files.
2. **`distilbert.ipynb`** — Fine-tune DistilBERT on captions (10 epochs, cross-entropy); export **test** logits to `results/`.
3. **`extract_logits.ipynb`** — Extract **test** logits from the fine-tuned BEiT checkpoint (`hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224`).
4. **`fusion.ipynb`** — Top-1, Top-3, and macro-F1 for BEiT, DistilBERT, and fusion (`logits_beit + logits_bert`) on FungiTastic-M and full.

## Installation

Python 3.10+.

```bash
pip install -r requirements.txt
```

## Paths

Set `dataset_dir` to your FungiTastic root (default `/FungiTastic`). Metadata CSVs live under `dataset_dir/metadata/…`; test images under `dataset_dir/FungiTastic/…` or `FungiTastic-Mini/…`.

DistilBERT checkpoints: `checkpoints/distilbert-ft-{mini,full}/`. BEiT logits: `features/beit-224-{mini,full}/`.
