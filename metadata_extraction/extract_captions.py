import os
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

prompt =  "Describe the visual features of the fungi, such as their colour, shape, texture and relative size. Focus on the fungi and its parts. Provide detailed description of the visual features but avoid speculations."

invalid_files = [
    # Train + Val
    "1-3861325341.JPG",
    "2-4100092812.JPG",
    "0-2238480458.JPG",
    "0-2238152962.JPG",
    "1-2238552443.JPG",
    "0-2597564674.JPG",
    "1-2238554704.JPG",
    "1-2238483885.JPG",
    "1-2238523717.JPG",
    "1-2237914372.JPG",
    "0-2238496151.JPG",
    "0-2238127565.JPG",
    "0-2238365191.JPG",

    # Test
    '1-4169763613.JPG',
    '0-4465865676.JPG',
    '0-4465900600.JPG',
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True, help="Current chunk")
    parser.add_argument("--chunk_total", type=int, required=True, help="Total number of chunks")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output folder")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata csv, e.g. 'dataset/FungiTastic/metadata/FungiTastic-Mini/FungiTastic-Mini-Train.csv' ")
    parser.add_argument("--image_path", type=str, required=True, help="Path to images, e.g. 'dataset/FungiTastic//FungiTastic-Mini/train/500p/'")

    args = parser.parse_args()
    os.makedirs(f"{args.output_path}/captions", exist_ok=True)

    # Load data
    df = pd.read_csv(args.metadata_path)
    df['full_path'] = args.image_path + df['filename']


    # Load the processor
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    ).to('cuda')


    # Main loop
    indexes = np.array_split(np.arange(len(df)), args.chunk_total)[args.chunk-1]
    for i in tqdm(indexes):
        row = df.iloc[i]
        output_path = args.output_path + "/captions/" + row['filename']

        if os.path.exists(f"{output_path}.json"):
            continue

        if row.filename in invalid_files:
            print(f"Skipping invalid {row.filename}")
            continue

        try:
            img = Image.open(row['full_path'])
            inputs = processor.process(
                images=[img],
                text=prompt
            )

            # move inputs to the correct device and make a batch of size 1
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=8000, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
                )

            # only get generated tokens; decode them to text
            generated_tokens = output[0,inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        except OSError as e:
            generated_text = ''

        
        with open(f"{output_path}.json", 'w', encoding='utf-8') as f:
            json.dump(generated_text, f)
