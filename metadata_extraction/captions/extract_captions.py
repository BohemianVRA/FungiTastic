import os
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image, UnidentifiedImageError
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

prompt = (
    "Describe the visual features of the fungi, such as their colour, shape, texture and relative size. "
    "Focus on the fungi and its parts. Provide detailed description of the visual features but avoid speculations."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True, help="Current chunk")
    parser.add_argument("--chunk_total", type=int, required=True, help="Total number of chunks")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output folder")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image directory")

    args = parser.parse_args()
    os.makedirs(os.path.join(args.output_path, "captions"), exist_ok=True)

    # Load data
    df = pd.read_csv(args.metadata_path)
    df['full_path'] = df['filename'].apply(lambda x: os.path.join(args.image_path, x))

    # Load the processor and model
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Process assigned chunk
    indexes = np.array_split(np.arange(len(df)), args.chunk_total)[args.chunk - 1]
    for i in tqdm(indexes):
        row = df.iloc[i]
        output_file = os.path.join(args.output_path, "captions", f"{row['filename']}.json")

        if os.path.exists(output_file):
            continue

        try:
            # Attempt to open and verify image
            img = Image.open(row['full_path'])
            img.verify()
            img = Image.open(row['full_path'])

            inputs = processor.process(
                images=[img],
                text=prompt
            )
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=8000, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )

            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        except (OSError, UnidentifiedImageError, Exception) as e:
            print(f"Skipping file {row['filename']} due to error: {e}")
            generated_text = ""

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(generated_text, f)
