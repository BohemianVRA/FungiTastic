# Image Caption Generation

This code generates detailed visual=based captions using the [Malmo-7B](https://huggingface.co/allenai/Molmo-7B-D-0924) vision-language model (VLM). 
The generated descriptions focus on visual features such as color, texture, shape, and relative size, aiding in fine-grained fungal identification. 
We use the following prompt:
> *"Describe the visual features of the fungi, such as their colour, shape, texture, and relative size. Focus on the fungi and their parts. Provide a detailed description of the visual features, but avoid speculations."*

### 📦 Requirements installation

```bash
pip install -r requirements.txt
```

### 🚀 Script Usage

```bash
python generate_captions.py \
  --chunk 1 \
  --chunk_total 10 \
  --output_path outputs/ \
  --metadata_path dataset/FungiTastic/metadata/FungiTastic-Mini/FungiTastic-Mini-Train.csv \
  --image_path dataset/FungiTastic/FungiTastic-Mini/train/500p/
```

#### Arguments

- `--chunk`: Index of the current chunk (1-based)
- `--chunk_total`: Total number of chunks
- `--output_path`: Path to store output JSON files
- `--metadata_path`: Path to the CSV file with image filenames
- `--image_path`: Path to the actual image files

Captions are saved as `.json` files under the `./captions` folder in `output_path`, named by their corresponding image filename (e.g. `1-1234567890.JPG.json`).
