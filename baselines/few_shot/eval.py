from pathlib import Path
from types import SimpleNamespace
import argparse
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import numpy as np
import torch
import yaml

from dataset.feature_fungi import FeatureFungiTastic
from scripts.baselines.few_shot.classifier import PrototypeClassifier, NNClassifier


def get_dataloader(test_dataset, batch_size=256, num_workers=0):
    """Creates a DataLoader for the test dataset with specified batch size and workers.
    
    Args:
        test_dataset: Dataset to create loader for
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
    """
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
    """Returns the appropriate classifier class based on the name.
    
    Args:
        classifier_name: Either 'centroid' for PrototypeClassifier or 'nn' for NNClassifier
    """
    if classifier_name == 'centroid':
        return PrototypeClassifier
    elif classifier_name == 'nn':
        return NNClassifier
    else:
        raise ValueError(f"Classifier {classifier_name} not implemented")


def get_classifier_embeddings(dataset_train):
    """Extracts embeddings for each class from the training dataset.
    
    Args:
        dataset_train: Training dataset containing class embeddings
        
    Returns:
        tuple: (class_embeddings, empty_classes)
            - class_embeddings: List of tensors containing embeddings for each class
            - empty_classes: List of class indices that had no embeddings
    """
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
    """Evaluates a few-shot classifier on the Fungi dataset.
    
    Args:
        path_out: Directory to save results
        data_path: Path to the dataset
        feature_path: Path to pre-computed features
        feature_model: Name of the feature model used
        classifier_name: Type of classifier to use ('centroid' or 'nn')
        split: Dataset split to evaluate on
        debug: If True, runs only 3 batches for quick testing
    """
    features_file_train = os.path.join(feature_path, feature_model, "224x224_train.h5")
    features_file_eval = os.path.join(feature_path, feature_model, f"224x224_{split}.h5")

    dataset_train = FeatureFungiTastic(
        root=data_path,
        features_file=features_file_train,
        split='train',
        size='300',
        task='closed',
        data_subset='FewShot',
        transform=None,
    )

    dataset_eval = FeatureFungiTastic(
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


def main(cfg):
    """Main function that runs the evaluation based on configuration.
    
    Args:
        cfg: Configuration object containing all necessary parameters
    """
    test_fungi(path_out=cfg.path_out, data_path=cfg.data_path, feature_path=cfg.feature_path,
               feature_model=cfg.feature_model, classifier_name=cfg.classifier, split=cfg.split,
               debug=cfg.debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--config_path', type=str, default='/home.stud/janoukl1/projects/fungi_code_public/FungiTastic/scripts/baselines/few_shot/config/fs.yaml',  
                        help='Path to the config file',)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    main(cfg)
