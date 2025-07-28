# 🏋️ Training Your Model on FungiTastic

This page gives practical steps to get started training a model for any FungiTastic benchmark.

---

## 1. Setup

- Clone the repo and install requirements:
    ```bash
    git clone https://github.com/bohemianvra/FungiTastic.git
    cd FungiTastic
    pip install -r requirements.txt
    ```

- Download the dataset (see [Downloading the Dataset](download.md)).

---

## 2. Loading the Data

- Each benchmark split is a CSV/TSV file listing image paths and metadata.
- Example (Python/pandas):
    ```python
    import pandas as pd

    df = pd.read_csv("fungitastic/closed_set_train.csv")
    print(df.columns)  # See available metadata fields
    ```

- Images are typically stored as paths in the DataFrame. Load with your favorite library (PIL, OpenCV, etc).

---

## 3. Training Scripts & Examples

- Example scripts for **closed-set**, **few-shot**, **open-set**, and **segmentation** are in `examples/` in the repo.
    - E.g., `examples/train_closed_set.py`
- All baselines use standard PyTorch; configs and hyperparameters are in the paper and README.

---

## 4. Tips

- Always start with the [Mini subset](../dataset.md#fungitastic-mini) for fast prototyping.
- For multi-modal training (images + metadata), see the relevant examples.
- Use pre-trained weights as strong starting points—see [Baselines & Models](../models/index.md).

---

## 5. Extending to New Benchmarks

- FungiTastic is benchmark-agnostic: write your own splits, add modalities, or propose new tasks.

---

> Next steps:  
> - [Evaluation Protocols](evaluation.md)  
> - [Baselines & Models](../models/index.md)
