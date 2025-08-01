# About FungiTastic data

## Download instructions

There are two options to download the dataset:
1. Use the download script (preferred)
2. Download the whole dataset from kaggle [~50GB]

### 1. Use the download script
The preferred approach is to use our script to get the data that allows downloading 
different subsets separately and in the desired image resolution.

**Prerequisites:** The download script requires `wget` to be installed on your system.
- **macOS**: `brew install wget`
- **Ubuntu/Debian**: `sudo apt-get install wget`
- **Windows**: Download from [wget for Windows](https://gnuwin32.sourceforge.net/packages/wget.htm) or use WSL

The following example downloads the metadata (common for all subsets) and the 'FungiTastic-Mini'
dataset subset with the image resolution of 300px
and saves it in the current directory. The argument 'save_path' is required.
   
```
cd dataset
python download.py --metadata --images --subset "m" --size "300" --save_path "./"  
```

By default, the data is extracted in the save_path folder and copies the file structure from Kaggle.
After extraction, the zip files are deleted.

**Related options:**
- **size**: [_300_, _500_, _720_, _fullsize_]
- **subset**: [full, m, fs]
- **keep_zip**: Do not delete the downloaded zip files (default: False)
- **no_extraction**: Do not extract the downloaded zip files (default: False)
- **rewrite**: Rewrite existing files (default: False)
- **satellite**: Download satellite data.
- **climatic**: Download climatic data.

Downloading segmentation masks, climate data and satellite images:

```
cd datasets
python download.py --masks --climatic --satellite --save_path "./"  
```


### 2. Download the dataset from kaggle [~50GB]:
While the structure of the data is kept the same, the dataset available on Kaggle only contains the 500p images.

[info] Using the Kaggle API, you have to always download all the data and subsets.

To download the data, you have to:
1. Register and login to Kaggle.
2. Install Kaggle API `pip install kaggle`
4. Store Kaggle login settings and locally.
   ```
   !mkdir ~/.kaggle
   !touch ~/.kaggle/kaggle.json
   api_token = {"username":"FILL YOUR USERNAME","key":"FILL YOUR APIKEY"}
   ```
5. Use CLI `kaggle datasets download -d picekl/fungitastic`


## Getting started: Data Loading Classes and Demo Notebooks

We provide data loading classes and demo notebooks to help you get started with different tasks.

### Core Data Loading Classes

#### `FungiTastic` Class (`fungi.py`)

The main dataset class for image classification tasks. It inherits from `fgvc.datasets.ImageDataset` and provides:

- **Flexible data loading**: Support for different subsets (Mini, FewShot, all), splits (train/val/test/dna), image sizes (300p, 500p, 720p, fullsize), and tasks (closed-set/open-set)
- **Metadata management**: Automatic handling of species labels, category IDs, and file paths
- **Parameter validation**: Built-in checks for valid dataset configurations
- **Easy visualization**: `show_sample()` method for displaying images with class information

**Usage example:**
```python
from dataset.fungi import FungiTastic

dataset = FungiTastic(
    root='path_to_data',
    data_subset='Mini',
    split='val',
    size='300',
    task='closed'
)

# Get a sample
image, category_id, file_path = dataset[0]
dataset.show_sample(0)
```

#### `MaskFungiTastic` Class (`mask_fungi.py`)

Extended dataset class for segmentation tasks, inheriting from `FungiTastic`. Supports three segmentation task types:

1. **Binary Segmentation** (`seg_task='binary'`): Foreground-background segmentation
2. **Semantic Segmentation** (`seg_task='semantic'`): Multi-class segmentation with masks merged by label
3. **Instance-aware Part Segmentation** (`seg_task='part'`): Individual mask parts for fine-grained segmentation

**Usage example:**
```python
from dataset.mask_fungi import MaskFungiTastic

# Binary segmentation
dataset_binary = MaskFungiTastic(
    root='path_to_data',
    split='val',
    seg_task='binary'
)

# Semantic segmentation
dataset_semantic = MaskFungiTastic(
    root='path_to_data',
    split='val',
    seg_task='semantic'
)

# Instance segmentation
dataset_instance = MaskFungiTastic(
    root='path_to_data',
    split='val',
    seg_task='instance'
)

# Get samples with masks
image, masks, category_id, file_path, labels = dataset_semantic[0]
dataset_semantic.show_sample(0)
```

### Demo Notebooks

#### `demo.ipynb`

Demonstration notebook covering:
- Basic dataset loading and exploration
- Image classification examples
- Metadata analysis and visualization
- Dataset statistics and class distribution

#### `mask_demo.ipynb`

Specialized notebook for segmentation tasks demonstrating:
- Loading and visualizing different segmentation task types
- Comparison of binary, semantic, and instance segmentation

### Dataset Configuration

The classes support various configuration options:

**Available subsets:**
- `'Mini'`: Smaller dataset with all image sizes available
- `'FewShot'`: Few-shot learning subset (300p, 500p only)
- `'all'`: Complete dataset (300p, 500p only)

**Available splits:**
- `'train'`: Training data
- `'val'`: Validation data  
- `'test'`: Test data
- `'dna'`: DNA-based test set (closed-set only)

**Available image sizes:**
- `'300'`: 300px resolution
- `'500'`: 500px resolution
- `'720'`: 720px resolution (Mini subset only)
- `'fullsize'`: Original resolution (Mini subset only)

**Available tasks:**
- `'closed'`: Closed-set classification/segmentation
- `'open'`: Open-set classification/segmentation (not available for DNA split)

