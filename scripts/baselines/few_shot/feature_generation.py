import importlib
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
    def __init__(self, device):
        super(FeatureExtractor, self).__init__()
        self.device = device

    def extract_features(self, image_path):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    @staticmethod
    def normalize_embedding(embs):
        """
        Normalize the embedding to -1, 1 range
        :return:
        """
        embs = embs.float()
        norm_features = torch.nn.functional.normalize(embs, dim=1, p=2)
        return norm_features

    @staticmethod
    def quantize_normalized_embedding(embs):
        """
        Quantize the normalized embedding to 8 bit unsigned integers
        :return:
        """
        embs = embs.float()

        assert embs.min() >= -1 and embs.max() <= 1, 'Embeddings must be normalized to -1, 1 range'

        # quantize the -1, 1 range to 8 bit u-integers
        image_features = ((embs + 1) * 127.5).to(torch.uint8).detach().cpu().numpy()
        return image_features


class DinoV2(FeatureExtractor):
    def __init__(self, device):
        super(DinoV2, self).__init__(device)
        self.model = None
        self.transform = self.get_transform()

    def load(self, model_name='vitb14_reg'):
        if model_name == 'vitb14_reg':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        else:
            raise ValueError(f'Unknown model kind: {model_name}')

        model.eval()
        model.to(self.device)

        self.model = model

    def extract_features(self, image):
        """

        :param model:
        :param image_tensor:
        :return:
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
        transforms_list = [
            tfms.Resize((resize_size, resize_size), interpolation=tfms.InterpolationMode.BICUBIC),
            tfms.ToTensor(),
            tfms.Normalize(mean=mean, std=std)
        ]
        return tfms.Compose(transforms_list)


class CLIP(FeatureExtractor):
    def __init__(self, device):
        super(CLIP, self).__init__(device)
        self.model = None
        self.processor = None
        # pil image resize to 224, 224
        self.size = 224, 224

    def load(self, model_name='clip-vit-base-patch32'):
        # 'clip-vit-base-patch32'
        # clip-vit-large-patch14
        model = CLIPModel.from_pretrained(f"openai/{model_name}")
        processor = CLIPProcessor.from_pretrained(f"openai/{model_name}")

        model.to(self.device)

        self.model = model
        self.processor = processor

    def extract_features(self, image):
        if self.model is None:
            raise ValueError('Model not loaded')

        image = image.resize(self.size, Image.BICUBIC)
        # TODO put image on gpu before processing (check what normalization is expected for tensors)
        image_tensor_proc = self.processor(images=image, return_tensors='pt').pixel_values

        # get the features
        features = self.model.get_image_features(pixel_values=image_tensor_proc.to(self.device))
        norm_features = self.normalize_embedding(features)
        return norm_features


class BioCLIP(CLIP):
    def load(self, model_name='bioclip'):
        # bioclip-vit-b-16-inat-only
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        # tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        self.processor = preprocess_val
        model.to(self.device)
        self.model = model

    def extract_features(self, image):
        if self.model is None:
            raise ValueError('Model not loaded')

        image = image.resize(self.size, Image.BICUBIC)
        image_tensor_proc = self.processor(image)[None]

        # get the features
        features = self.model.encode_image(image_tensor_proc.to(self.device))
        norm_features = self.normalize_embedding(features)
        return norm_features


def get_model(model_name):
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
    # image_path = '/mnt/datagrid/plants/DanishFungiDataset/DanishFungi24/DF24-FS/DF24-FS-test-300'

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

        # TODO: run in batch mode https://github.com/openai/CLIP/issues/175
        save_freq = -1

        # if it doesn't exist, create the feature directory
        Path(feature_folder).mkdir(parents=True, exist_ok=True)

        feature_file_full = os.path.join(feature_folder, f'224x224_{split}.h5')
        cols = ['im_name', 'embedding']
        df = pd.DataFrame(columns=cols)

        # idxs = np.arange(int(0.4 * len(nico)), len(nico))[::-1]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings for fungi dataset')
    parser.add_argument('--model', type=str, default='bioclip', choices=['clip', 'dinov2', 'bioclip'],
                        help='Model to use for feature extraction')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to extract features for', choices=['train', 'val', 'test', 'all'])
    args = parser.parse_args()

    with open('../../../config/path.yaml', "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    generate_embeddings(data_path=cfg.data_path, model_name=args.model, data_split=args.split, feature_path=cfg.feature_path)



