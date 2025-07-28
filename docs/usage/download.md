# 📥 Downloading the FungiTastic Dataset

FungiTastic is hosted for easy access on [Kaggle](https://www.kaggle.com/datasets/picekl/fungitastic) and [HuggingFace](https://huggingface.co/collections/BVRA/fungitastic-66a227ce0520be533dc6403b).  
This page explains how to download the data, subset splits, and any associated files.

---

## 1. Download from Kaggle

- **Go to:** [Kaggle FungiTastic page](https://www.kaggle.com/datasets/picekl/fungitastic)
- Use the “Download All” button, or use the Kaggle API:

    ```bash
    kaggle datasets download -d picekl/fungitastic
    unzip fungitastic.zip
    ```

- Each subset (full, Mini, Few-shot) is in its own folder.
- Metadata, splits, and README included.

---

## 2. Download Pre-trained Models

- **HuggingFace Models:**  
  Visit the [FungiTastic Collection on HuggingFace](https://huggingface.co/collections/BVRA/fungitastic-66a227ce0520be533dc6403b)  
  You can download weights for all baselines and ready-to-use models.

---

## 3. Access via Scripts

- See [usage/training.md](training.md) for code samples to load images, metadata, and splits.
- For custom splits or subsets, refer to [dataset.md](../dataset.md).

---

## FAQ

- **License:** See included `LICENSE` file and dataset README.
- **Issues?** Open a GitHub Issue: [FungiTastic Issues](https://github.com/bohemianvra/FungiTastic/issues)

---

> Jump to:  
> - [Dataset Overview](../dataset.md)  
> - [How to Train a Model](training.md)
