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

from dataset.feature_fungi import FeatureFungiTasticDataset
from scripts.baselines.few_shot.classifier import PrototypeClassifier, NNClassifier


def get_dataloader(test_dataset, batch_size=256, num_workers=0):
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_dataloader


def get_classifier_cls(classifier_name):
    if classifier_name == 'centroid':
        return PrototypeClassifier
    elif classifier_name == 'nn':
        return NNClassifier
    else:
        raise ValueError(f"Classifier {classifier_name} not implemented")


def get_classifier_embeddings(dataset_train):
    class_embeddings = []
    empty_classes = []
    n_classes = min(torch.inf, dataset_train.n_classes)
    for cls in range(n_classes):
        cls_embs = dataset_train.get_embeddings_for_class(cls)
        if len(cls_embs) == 0:
            # if no embeddings for class, use zeros
            empty_classes.append(cls)
            class_embeddings.append(torch.zeros(1, dataset_train.emb_dim))
        else:
            class_embeddings.append(torch.tensor(np.vstack(cls_embs.values)))
    return class_embeddings, empty_classes


def test_fungi(path_out, data_path, feature_path, feature_model, classifier_name, split, debug=False):
    features_file_train = os.path.join(feature_path, feature_model, f"224x224_no_micro_train.h5")
    features_file_eval = os.path.join(feature_path, feature_model, f"224x224_no_micro_{split}.h5")

    dataset_train = FeatureFungiTasticDataset(
        root=data_path,
        features_file=features_file_train,
        split='train',
        size='300',
        task='closed',
        data_subset='FewShot',
        transform=None,
    )

    dataset_eval = FeatureFungiTasticDataset(
        root=data_path,
        features_file=features_file_eval,
        split=split,
        size='300',
        task='closed',
        data_subset='FewShot',
        transform=None,
    )

    exp_name = f"eval_{feature_model}_{split}_{classifier_name}"

    print(f"Evaluating {exp_name}")

    dataloader = get_dataloader(test_dataset=dataset_eval)

    class_embeddings, _ = get_classifier_embeddings(dataset_train)

    classifier = get_classifier_cls(classifier_name)(class_embeddings, device='cpu')
    # classifier.cuda()

    #  if True, runs 1 train/val batch only in trainer.fit, n batches if set to n
    fast_dev_run = 3 if debug else False

    result_dir = Path(path_out) / 'results' / 'fs' / split
    classifier.evaluate(dataloader=dataloader, fast_dev_run=fast_dev_run)
    classifier.save_results(out_dir=result_dir, file_name=f'{exp_name}')


def main():
    config_path = '../../../config/FungiTastic_FS.yaml'
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    # # test_fungi(path_out, data_path, feature_path, feature_model, classifier_name, split, debug=False)
    test_fungi(
        path_out=cfg.path_out,
        data_path=cfg.data_path,
        feature_path=cfg.feature_path,
        feature_model=cfg.feature_model,
        classifier_name=cfg.classifier,
        split=cfg.split,
        debug=False)


if __name__ == '__main__':
    main()