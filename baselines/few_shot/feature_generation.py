from types import SimpleNamespace
from typing import Sequence
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms as tfms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
from PIL import Image
import open_clip

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


from dataset.fungi import FungiTastic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureExtractor(torch.nn.Module):
    """Base class for feature extraction from images using various vision models.
    
    This abstract class provides common functionality for extracting and normalizing
    image features using different vision models like CLIP, DINOv2, etc.
    """
    def __init__(self, device):
        super(FeatureExtractor, self).__init__()
        self.device = device

    def extract_features(self, image_path):
        """Extract features from an image.
        
        Args:
            image_path: Path to the image or PIL Image object
            
        Returns:
            Normalized feature embeddings
        """
        raise NotImplementedError

    def load(self):
        """Load the model weights and prepare it for inference."""
        raise NotImplementedError

    @staticmethod
    def normalize_embedding(embs):
        """Normalize the embedding vectors to unit length (L2 normalization).
        
        Args:
            embs: Raw embedding tensor
            
        Returns:
            Normalized embeddings in range [-1, 1]
        """
        embs = embs.float()
        norm_features = torch.nn.functional.normalize(embs, dim=1, p=2)
        return norm_features

    @staticmethod
    def quantize_normalized_embedding(embs):
        """Quantize normalized embeddings to 8-bit unsigned integers.
        
        Args:
            embs: Normalized embeddings in range [-1, 1]
            
        Returns:
            Quantized embeddings as numpy array of uint8
        """
        embs = embs.float()

        assert embs.min() >= -1 and embs.max() <= 1, 'Embeddings must be normalized to -1, 1 range'

        # quantize the -1, 1 range to 8 bit u-integers
        image_features = ((embs + 1) * 127.5).to(torch.uint8).detach().cpu().numpy()
        return image_features


class DinoV2(FeatureExtractor):
    """Feature extractor using Facebook's DINOv2 vision transformer model."""
    def __init__(self, device):
        super(DinoV2, self).__init__(device)
        self.model = None
        self.transform = self.get_transform()

    def load(self, model_name='vitb14_reg'):
        """Load DINOv2 model weights.
        
        Args:
            model_name: Name of the DINOv2 model variant to load
        """
        if model_name == 'vitb14_reg':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        else:
            raise ValueError(f'Unknown model kind: {model_name}')

        model.eval()
        model.to(self.device)

        self.model = model

    def extract_features(self, image):
        """Extract features from an image using DINOv2.
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized feature embeddings
        """
        if self.model is None:
            raise ValueError('Model not loaded')

        # get the features
        image_tensor = self.transform(image).unsqueeze(0)
        features = self.model(image_tensor.to(self.device))
        norm_features = self.normalize_embedding(features)
        return norm_features

    @staticmethod
    def get_transform(resize_size: int = 224,
                mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
                std: Sequence[float] = IMAGENET_DEFAULT_STD,
        ):
        """Get the image transformation pipeline for DINOv2.
        
        Args:
            resize_size: Size to resize images to
            mean: Normalization mean values
            std: Normalization standard deviation values
            
        Returns:
            Composition of image transformations
        """
        transforms_list = [
            tfms.Resize((resize_size, resize_size), interpolation=tfms.InterpolationMode.BICUBIC),
            tfms.ToTensor(),
            tfms.Normalize(mean=mean, std=std)
        ]
        return tfms.Compose(transforms_list)


class CLIP(FeatureExtractor):
    """Feature extractor using OpenAI's CLIP model."""
    def __init__(self, device):
        super(CLIP, self).__init__(device)
        self.model = None
        self.processor = None
        # pil image resize to 224, 224
        self.size = 224, 224

    def load(self, model_name='clip-vit-base-patch32'):
        """Load CLIP model weights and processor.
        
        Args:
            model_name: Name of the CLIP model variant to load
        """
        model = CLIPModel.from_pretrained(f"openai/{model_name}")
        processor = CLIPProcessor.from_pretrained(f"openai/{model_name}")

        model.to(self.device)

        self.model = model
        self.processor = processor

    def extract_features(self, image):
        """Extract features from an image using CLIP.
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized feature embeddings
        """
        if self.model is None:
            raise ValueError('Model not loaded')

        image = image.resize(self.size, Image.BICUBIC)
        image_tensor_proc = self.processor(images=image, return_tensors='pt').pixel_values

        # get the features
        features = self.model.get_image_features(pixel_values=image_tensor_proc.to(self.device))
        norm_features = self.normalize_embedding(features)
        return norm_features


class BioCLIP(CLIP):
    """Feature extractor using BioCLIP model, specialized for biological images."""
    def load(self, model_name='bioclip'):
        """Load BioCLIP model weights and processor.
        
        Args:
            model_name: Name of the BioCLIP model variant to load
        """
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        self.processor = preprocess_val
        model.to(self.device)
        self.model = model

    def extract_features(self, image):
        """Extract features from an image using BioCLIP.
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized feature embeddings
        """
        if self.model is None:
            raise ValueError('Model not loaded')

        image = image.resize(self.size, Image.BICUBIC)
        image_tensor_proc = self.processor(image)[None]

        # get the features
        features = self.model.encode_image(image_tensor_proc.to(self.device))
        norm_features = self.normalize_embedding(features)
        return norm_features


def get_model(model_name):
    """Factory function to create and load the appropriate feature extractor model.
    
    Args:
        model_name: Name of the model to load ('clip', 'dinov2', or 'bioclip')
        
    Returns:
        Loaded and configured feature extractor model
    """
    if model_name == 'clip':
        model = CLIP(device=DEVICE)
    elif model_name == 'dinov2':
        model = DinoV2(device=DEVICE)
    elif model_name == 'bioclip':
        model = BioCLIP(device=DEVICE)
    else:
        raise ValueError(f'Unknown model kind: {model_name}')

    model.load()
    model.eval()
    return model


def generate_embeddings(data_path, feature_path, model_name='clip', data_split='val'):
    """Generate and save image embeddings for the FungiTastic dataset.
    
    Args:
        data_path: Path to the dataset root directory
        feature_path: Path where feature embeddings will be saved
        model_name: Name of the model to use for feature extraction
        data_split: Dataset split to process ('val', 'test', 'train', or 'all')
    """
    model = get_model(model_name)

    splits = [data_split] if data_split != 'all' else ['val', 'test', 'train']

    for split in splits:
        dataset = FungiTastic(
            root=data_path,
            split=split,
            size='300',
            task='closed',
            data_subset='FewShot',
            transform=None,
            )

        if model_name == 'dinov2':
            feature_folder = os.path.join(feature_path, f'{model_name}_vit_b')
        else:
            feature_folder = os.path.join(feature_path, model_name)

        save_freq = -1

        # if it doesn't exist, create the feature directory
        Path(feature_folder).mkdir(parents=True, exist_ok=True)

        feature_file_full = os.path.join(feature_folder, f'224x224_{split}.h5')

        # if the file exists, skip the feature generation
        if os.path.exists(feature_file_full):
            print(f'Skipping {feature_file_full} because it already exists')
            continue

        cols = ['im_name', 'embedding']
        df = pd.DataFrame(columns=cols)

        idxs = np.arange(len(dataset))
        im_names, embs = [], []
        for idx in tqdm(idxs):
            im, label, file_path = dataset[idx]

            with torch.no_grad():
                feat = model.extract_features(im)
            feat_quant = model.quantize_normalized_embedding(feat)

            im_names.append(os.path.basename(file_path))
            embs.append(feat_quant)

            if idx % save_freq == 0 and save_freq > 0:
                # concat new dataset to the existing dataframe
                new = pd.DataFrame({'im_name': im_names, 'embedding': embs})
                df = pd.concat([df, new], ignore_index=True)

                # save
                df.to_hdf(feature_file_full, key='df', mode='w')

                # clear the lists
                im_names, embs = [], []

        new = pd.DataFrame({'im_name': im_names, 'embedding': embs})
        df = pd.concat([df, new], ignore_index=True)
        df.to_hdf(feature_file_full, key='df', mode='w')
        print(f'Saved {len(df)} embeddings to {feature_file_full}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings for fungi dataset')
    parser.add_argument('--config_path', type=str, default='/home.stud/janoukl1/projects/fungi_code_public/FungiTastic/scripts/baselines/few_shot/config/fs.yaml',  
                        help='Path to the config file',)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    generate_embeddings(data_path=cfg.data_path, model_name=cfg.model, data_split=cfg.split, feature_path=cfg.feature_path)



