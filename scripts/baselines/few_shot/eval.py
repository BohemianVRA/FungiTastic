from pathlib import Path
from types import SimpleNamespace
from functools import partial
import argparse
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import numpy as np
import torch
from tqdm import tqdm
import yaml
import wandb

from dataset.fetaure_fungi import FeatureFungiTastic
from scripts.baselines.few_shot.classifier import PrototypeClassifier, NNClassifier


def get_dataloader(cfg, test_dataset):
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_dataloader


def get_classifier_cls(cfg):
    if cfg.classifier == 'centroid':
        return PrototypeClassifier
    elif cfg.classifier == 'nn':
        return NNClassifier
    else:
        raise ValueError(f"Classifier {cfg.classifier} not implemented")


def test_fungi(cfg, split):
    # np, python and torch random seed
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.set_float32_matmul_precision('high')

    features_file = os.path.join(cfg.feature_path, cfg.feature_model, f"224x224_no_micro_{split}.h5")

    dataset_train = FeatureFungiTastic(
        root=cfg.data_path,
        features_file=features_file,
        split='train',
        size='300',
        task='closed',
        data_subset='FewShot',
        transform=None,
    )

    dataset_eval = FeatureFungiTastic(
        root=cfg.data_path,
        features_file=features_file,
        split=split,
        size='300',
        task='closed',
        data_subset='FewShot',
        transform=None,
    )

    cfg.exp_name = f"eval_fungi_{cfg.feature_model}_{split}_{cfg.classifier}"

    print(f"Evaluating {cfg.exp_name}")

    if cfg.debug:
        cfg.num_workers = 0

    # train dataloader not working now
    dataloader = get_dataloader(cfg, dataset_eval)

    n_classes = min(torch.inf, dataset_train.n_classes)

    class_embeddings = []
    empty_classes = []
    for cls in range(n_classes):
        cls_embs = dataset_train.get_embeddings_for_class(cls)
        if len(cls_embs) == 0:
            # if no embeddings for class, use zeros
            empty_classes.append(cls)
            class_embeddings.append(torch.zeros(1, dataset_train.emb_dim))
        else:
            class_embeddings.append(torch.tensor(np.vstack(cls_embs.values)))

    classifier = get_classifier_cls(cfg)(cfg, class_embeddings)
    classifier.cuda()

    #  if True, runs 1 train/val batch only in trainer.fit, n batches if set to n
    fast_dev_run = 3 if cfg.debug else False

    result_dir = Path(cfg.path_out) / 'results' / 'fs' / split
    classifier.evaluate(dataloader=dataloader, fast_dev_run=fast_dev_run)
    classifier.save_results(out_dir=result_dir, file_name=f'{cfg.exp_name}')


def main():
    config_path = '../../../config/FungiTastic_FS.yaml'
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    # convert to SimpleNamespace for easier access
    cfg = SimpleNamespace(**dict(cfg))

    test_fungi(cfg, split=cfg.split)


if __name__ == '__main__':
    # main()
    visualize()