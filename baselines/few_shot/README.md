# 🍄 Few-Shot Segmentation Baselines

This directory contains scripts for implementing and evaluating few-shot segmentation baselines using various pretrained models.

## 📋 Overview

The few-shot segmentation pipeline consists of two main steps:
1. 🔍 Feature generation using pretrained models
2. 📊 Evaluation using different classifiers

## 🧠 Technical Details

### Feature Extraction
The pipeline uses state-of-the-art vision models to extract rich feature representations:
- 🎨 CLIP: Contrastive Language-Image Pre-training model that learns visual concepts from natural language supervision
- 🦖 DINOv2: Self-supervised vision transformer that learns powerful visual features without labels
- 🧬 BioCLIP: Specialized version of CLIP fine-tuned on biological images

### Classification Methods

#### Prototype Classifier
- Computes class prototypes by averaging feature vectors of training samples for each class
- Uses cosine similarity to measure distance between test samples and class prototypes
- Predicts class based on highest similarity score
- Advantages:
  - Simple and interpretable
  - Works well with limited training data
  - Computationally efficient

#### Nearest Neighbor Classifier
- Uses FAISS (Facebook AI Similarity Search) for efficient similarity search
- Stores all training samples in an index optimized for fast retrieval
- Finds the closest training sample for each test sample
- Advantages:
  - More flexible than prototype-based approach
  - Can capture complex class boundaries
  - Better for classes with high intra-class variance

## ⚙️ Configuration

Before running the scripts, you need to set up your configuration in `config/fs.yaml` or create your own configuration file. The configuration file should be in YAML format and must include these parameters:

```yaml
# Required paths
data_path: '/path/to/FungiTastic/dataset'  # Path to the FungiTastic dataset
feature_path: '/path/to/save/features'     # Path where extracted features will be saved

# Optional settings (with defaults)
feature_model: 'dinov2'                     # Feature extraction model: 'clip', 'dinov2', or 'bioclip'
classifier: 'nn'                            # Classification method: 'nn' (nearest neighbor) or 'centroid'
```

All other parameters have sensible defaults. The script will use the configuration specified by the `--config` parameter.

You can further customize the feature extraction model and classifier type in the configuration file. The available options are:
- `feature_model`: Choose between 'clip', 'dinov2', or 'bioclip' for feature extraction
- `classifier`: Choose between 'nn' (nearest neighbor) or 'centroid' (prototype-based) classification

## 🚀 Usage

### 1. Feature Generation

Generate embeddings using one of the pretrained models:

```bash
python feature_generation.py 
```

Parameters:
- `config`: Path to configuration file (YAML format) containing model settings and dataset parameters
The script will use the paths specified in yotheur `config/fs.yaml` file by default.

### 2. Evaluation

After generating features, evaluate the few-shot segmentation performance (using the parameters from the config file):

```bash
python eval.py
```

## 📓 Demo Notebook

A demo notebook is available at `FungiTastic/scripts/baselines/few_shot/demo.ipynb` that demonstrates:
- 📥 Loading and preprocessing the dataset
- 🔍 Feature extraction with different models
- 📊 Evaluation using prototype and nearest neighbor classifiers
- 📈 Visualization of results

## 📊 Output

The evaluation script generates:
- 📄 CSV file with per-image predictions and metrics
- 📋 JSON file with overall performance metrics including:
  - 🎯 Top-1 accuracy
  - 📈 F1 score
  - 🎯 Top-3 accuracy (when applicable)

## 📦 Requirements

Key dependencies include:
- PyTorch
- Transformers
- FAISS
- OpenCLIP
- OpenCV
- Albumentations
- Pandas
- NumPy
- scikit-learn
- timm
- wandb

For a complete list of dependencies with specific versions, please refer to `requirements.txt`.
