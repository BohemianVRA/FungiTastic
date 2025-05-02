# 🍄 Segmentation 

## 🎯 Zero-shot segmentation

We provide a zero-shot segmentation approach based on GroundingDINO and the Segment Anything Model (SAM). 

## 🔬 Technical Details

The segmentation pipeline uses a two-stage approach:

1. **🔍 GroundingDINO Stage**:
   - Uses the text prompt "mushroom" to localize mushrooms in the image
   - Outputs bounding boxes with confidence scores
   - The model is pre-trained on large-scale data and can understand natural language descriptions
   - Default thresholds: box_threshold=0.3, text_threshold=0.25

2. **🎨 SAM Stage**:
   - Takes the bounding boxes from GroundingDINO as input
   - Uses these boxes as prompts to generate precise pixel-level masks
   - SAM is a powerful foundation model that can segment any object given a prompt
   - The model uses the ViT-H architecture by default

The pipeline combines the zero-shot detection capabilities of GroundingDINO with the precise segmentation abilities of SAM, allowing for mushroom segmentation without requiring task-specific training.

## ⚙️ Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install GroundingDINO:
   Follow the official installation guide: [GroundingDINO Installation](https://github.com/IDEA-Research/GroundingDINO#installation)

> ⚠️ **CUDA Installation Note**: If you want to run GroundingDINO with CUDA support, please follow the CUDA installation instructions in the official guide carefully, as the setup can be tricky.

## 📝 Configuration

Before running the mask generation script, you need to configure the paths in the config file. Create a copy of `config/seg.yaml` and update the following paths:

- `data_path`: Path to your FungiTastic dataset
- `mask_path`: Path where generated masks will be saved
- `ckpt_path`: Path where model checkpoints will be downloaded

Example configuration:
```yaml
data_path: '/path/to/FungiTastic'
mask_path: '/path/to/save/masks'
ckpt_path: '/path/to/model/checkpoints'
```

## 🚀 Generating Masks

To generate masks for your dataset, run:

```bash
python generate_masks.py --config_path path/to/your/config.yaml
```

The script will:
1. Download the required model checkpoints if not present
2. Process each image in the dataset
3. Generate binary masks for mushrooms
4. Save the masks in the specified `mask_path` directory

## 👀 Visualizing Results

You can visualize the generated masks using the provided Jupyter notebook:

```bash
jupyter lab demo.ipynb
```

The demo notebook shows:
- Original images
- Generated masks
- Side-by-side comparison

## 📊 Evaluation

To evaluate the quality of the generated masks:

```bash
python eval.py --config_path path/to/your/config.yaml
```

This will compute the IoU (Intersection over Union) metric between the generated masks and ground truth masks (if available).