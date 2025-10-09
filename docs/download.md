# 📥 How to download the FungiTastic dataset?

FungiTastic is available for easy access via **Kaggle** or using our **flexible Python download script** (recommended for customized or partial downloads).

| Method                              | Best For                                                         | Features                      |
| ----------------------------------- | ---------------------------------------------------------------- | ----------------------------- |
| **Download Script** *(recommended)* | Researchers needing subsets, specific resolutions, or modalities | Download *just* what you need |
| **Kaggle**                          | Anyone wanting the full 500px dataset fast                       | 1-click, \~50GB, all-in-one   |
---

## 1. Download via Script (Recommended)

The download script gives **full control**:

* Choose any subset (Full, Mini, Few-shot)
* Select image resolution (300, 500, 720, fullsize)
* Download only the modalities you want (images, masks, satellite, climate...)

**Prerequisites:**

* Python 3.7+
* `wget` utility

  * *macOS*: `brew install wget`
  * *Ubuntu/Debian*: `sudo apt-get install wget`
  * *Windows*: [wget for Windows](https://gnuwin32.sourceforge.net/packages/wget.htm) or use WSL

**Example:** Download “Mini” subset at 300px:

```bash
cd dataset
python download.py --metadata --images --subset "m" --size "300" --save_path "./"
```

This downloads metadata and the FungiTastic-Mini images at 300px to your chosen directory.

**Example:** Download segmentation masks, climate, and satellite data:

```bash
cd dataset
python download.py --masks --climatic --satellite --save_path "./"
```

**Common Options:**

| Option            | Description                       | Values                          |
| ----------------- | --------------------------------- | ------------------------------- |
| `--subset`        | Which dataset split               | `full`, `m`, `fs`               |
| `--size`          | Image resolution                  | `300`, `500`, `720`, `fullsize` |
| `--metadata`      | Download metadata CSV             | *(flag)*                        |
| `--images`        | Download images                   | *(flag)*                        |
| `--masks`         | Download segmentation masks       | *(flag)*                        |
| `--satellite`     | Download satellite image data     | *(flag)*                        |
| `--climatic`      | Download climate/bioclimatic data | *(flag)*                        |
| `--keep_zip`      | Keep zip files after extraction   | *(flag)*                        |
| `--no_extraction` | Download but do not extract       | *(flag)*                        |
| `--rewrite`       | Overwrite existing files          | *(flag)*                        |
| `--save_path`     | Where to save files (required)    | folder path                     |

For full usage and all options, run:

```bash
python download.py --help
```

---

## 2. Download from Kaggle (Full Dataset, 500px)

* Go to: [Kaggle FungiTastic page](https://www.kaggle.com/datasets/picekl/fungitastic)

* Download manually, or via API:

  ```bash
  pip install kaggle
  kaggle datasets download -d picekl/fungitastic
  unzip fungitastic.zip
  ```

* All subsets (Full, Mini, Few-shot) included.

* Metadata, splits, and README are inside.

> **Note:** Kaggle always downloads the entire dataset at 500px resolution. To get other resolutions or only part of the data, use the script above.

---

## 📝 FAQ & Troubleshooting

* **Issues or questions?** [Open a GitHub Issue](https://github.com/bohemianvra/FungiTastic/issues)
* **How do I load data in Python?** See [How to Train a Model](training.md) for code examples.

---

> 🔗 **Related links:**
> - [Dataset Overview](../dataset.md)
> - [How to Train a Model](training.md)
> - [Download script on GitHub](https://github.com/bohemianvra/FungiTastic/tree/main/dataset)
