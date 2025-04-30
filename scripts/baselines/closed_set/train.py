import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from fgvc.core.training import predict, train
from fgvc.datasets import get_dataloaders
from fgvc.losses import FocalLossWithLogits, SeesawLossWithLogits
from fgvc.utils.experiment import (
    get_optimizer_and_scheduler,
    load_config,
    load_model,
    load_test_metadata,
    load_train_metadata,
    parse_unknown_args,
    save_config,
)
from fgvc.utils.utils import set_cuda_device, set_random_seed
from fgvc.utils.wandb import (
    finish_wandb,
    init_wandb,
    resume_wandb,
    set_best_scores_in_summary,
)
from PIL import Image, ImageFile
from scipy.special import softmax
from torch.utils.data import DataLoader

from utils.hfhub import export_model_to_huggingface_hub_from_checkpoint

# To handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Logger setup
logger = logging.getLogger("script")


def load_args(args: list = None):
    """
    Load training script arguments using argparse.

    Parameters
    ----------
    args : list, optional
        Optional list of arguments for parsing, by default None.

    Returns
    -------
    tuple
        A tuple containing the parsed args (Namespace) and extra arguments (dict).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        help="Path to a training metadata file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val-path", help="Path to a val metadata file.", type=str, required=True
    )
    parser.add_argument(
        "--test-path", help="Path to a test metadata file.", type=str, required=True
    )
    parser.add_argument(
        "--config-path",
        help="Path to a training config yaml file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cuda-devices",
        help="Visible cuda devices (cpu,0,1,2,...).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb-entity",
        help="Entity name for logging experiment to W&B.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--wandb-project",
        help="Project name for logging experiment to W&B.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--resume-exp-name",
        help="Experiment name to resume training.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hfhub-owner",
        help="User or project name for uploading the model to HuggingFace Hub.",
        type=str,
        default=None,
    )

    args, unknown_args = parser.parse_known_args(args)
    extra_args = parse_unknown_args(unknown_args)
    return args, extra_args


def evaluate(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    path: str,
    device: torch.device = "cpu",
    log_images: bool = False,
):
    """
    Evaluate the model and log the evaluation metrics to W&B.

    Parameters
    ----------
    model : nn.Module
        The model to be evaluated.
    trainloader : DataLoader
        DataLoader for training data.
    testloader : DataLoader
        DataLoader for test data.
    path : str
        Directory to store visualizations.
    device : torch.device, optional
        Device to use (CPU or CUDA), by default "cpu".
    log_images : bool, optional
        Flag to log images to W&B, by default False.
    """
    if wandb.run is None:
        return
    
    # if "class_id" not in train_df:
    #     logger.info("No .")
    #     return
    
    # Evaluate model
    logger.info("Creating predictions.")
    preds, targs, _, scores = predict(model, testloader, device=device)
    print(scores)
    argmax_preds = preds.argmax(1)
    max_conf = softmax(preds, 1).max(1)
    softmax_values = softmax(preds, 1)

    # Create W&B prediction table
    train_df = trainloader.dataset.df
    test_df = testloader.dataset.df

    pred_df = pd.DataFrame()
    if log_images:
        pred_df["image"] = test_df["image_path"].apply(
            lambda image_path: wandb.Image(data_or_path=Image.open(image_path))
        )
    
    id2species = dict(zip(train_df["class_id"], train_df["scientificName"]))

    top5_indices = np.argsort(-softmax_values, axis=1)[:, :5]
    top5_species = [str([id2species[i] for i in row]) for row in top5_indices]
    top5_softmax = [
        str(softmax_values[i][top5_indices[i]]) for i in range(len(top5_indices))
    ]

    pred_df["top5-species"] = top5_species
    pred_df["top5-softmax"] = top5_softmax
    # pred_df["species"] = test_df["species"]
    pred_df["species-predicted"] = [id2species[x] for x in argmax_preds]
    # pred_df["class_id"] = test_df["class_id"]
    pred_df["class_id-predicted"] = argmax_preds
    pred_df["max-confidence"] = max_conf

    for col in ["image_path"]:
        pred_df[col] = test_df[col]

    wandb.log({"pred_table": wandb.Table(dataframe=pred_df)})
    wandb.log(
        {
            "test/F1": scores["F1"],
            "test/Accuracy": scores["Accuracy"],
            "test/Recall@3": scores["Recall@3"],
        }
    )


def add_metadata_info_to_config(
    config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    """
    Add metadata information (such as class count) to the configuration file.

    Parameters
    ----------
    config : dict
        Configuration dictionary for training.
    train_df : pd.DataFrame
        Training metadata.
    val_df : pd.DataFrame
        Validation metadata.
    test_df : pd.DataFrame
        Test metadata.

    Returns
    -------
    dict
        Updated configuration with metadata information.
    """
    assert "class_id" in train_df and "class_id" in val_df
    config["number_of_classes"] = len(train_df["class_id"].unique())
    config["validation_classes"] = len(val_df["class_id"].unique())
    config["training_samples"] = len(train_df)
    config["validation_samples"] = len(val_df)
    config["test_samples"] = len(test_df)
    return config


def train_clf(
    *,
    train_metadata: str = None,
    valid_metadata: str = None,
    test_metadata: str = None,
    config_path: str = None,
    cuda_devices: str = None,
    wandb_entity: str = None,
    wandb_project: str = None,
    resume_exp_name: str = None,
    hfhub_owner: str = None,
    **kwargs,
):
    """
    Train a model on a classification task.

    Parameters
    ----------
    train_metadata : str, optional
        Path to training metadata, by default None.
    valid_metadata : str, optional
        Path to validation metadata, by default None.
    test_metadata : str, optional
        Path to test metadata, by default None.
    config_path : str, optional
        Path to the training config file, by default None.
    cuda_devices : str, optional
        Comma-separated list of CUDA devices, by default None.
    wandb_entity : str, optional
        W&B entity name for logging, by default None.
    wandb_project : str, optional
        W&B project name for logging, by default None.
    resume_exp_name : str, optional
        Experiment name to resume from checkpoint, by default None.
    hfhub_owner : str, optional
        HuggingFace Hub user for uploading the model, by default None.
    """
    if train_metadata is None or valid_metadata is None or config_path is None:
        # Load script args if not provided
        args, extra_args = load_args()
        train_metadata = args.train_path
        valid_metadata = args.val_path
        test_metadata = args.test_path
        config_path = args.config_path
        cuda_devices = args.cuda_devices
        wandb_entity = args.wandb_entity
        wandb_project = args.wandb_project
        hfhub_owner = args.hfhub_owner
    else:
        extra_args = kwargs

    # Load training configuration
    logger.info("Loading training config.")
    config = load_config(
        config_path,
        extra_args,
        run_name_fmt="architecture-loss-augmentations",
        resume_exp_name=resume_exp_name,
    )

    # Set device and random seed
    device = set_cuda_device(cuda_devices)
    set_random_seed(config["random_seed"])

    # Load training, validation, and test metadata
    logger.info("Loading training and validation metadata.")
    train_df, valid_df = load_train_metadata(train_metadata, valid_metadata)
    test_df = load_test_metadata(test_metadata)
    config = add_metadata_info_to_config(config, train_df, valid_df, test_df)

    # Load model and optimizer
    logger.info("Creating model, optimizer, and scheduler.")
    model, model_mean, model_std = load_model(config)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)

    # Create DataLoaders
    logger.info("Creating DataLoaders.")
    trainloader, validloader, _, _ = get_dataloaders(
        train_df,
        valid_df,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
    )

    # Loss function selection
    logger.info("Creating loss function.")
    if config["loss"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == "FocalLoss":
        criterion = FocalLossWithLogits()
    elif config["loss"] == "SeeSawLoss":
        class_counts = train_df["class_id"].value_counts().sort_index().values
        criterion = SeesawLossWithLogits(class_counts=class_counts)
    else:
        logger.error(f"Unknown loss: {config['loss']}")
        raise ValueError()

    # Initialize W&B logging
    if wandb_entity is not None and wandb_project is not None:
        if resume_exp_name is None:
            init_wandb(
                config, config["run_name"], entity=wandb_entity, project=wandb_project
            )
        else:
            if "wandb_run_id" not in config:
                raise ValueError("Config is missing 'wandb_run_id' field.")
            resume_wandb(
                run_id=config["wandb_run_id"],
                entity=wandb_entity,
                project=wandb_project,
            )

    # Save config for this run
    if resume_exp_name is None:
        save_config(config)

    # Train the model
    logger.info("Training the model.")
    train(
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        wandb_train_prefix="train/",
        wandb_valid_prefix="val/",
        num_epochs=config["epochs"],
        accumulation_steps=config.get("accumulation_steps", 1),
        clip_grad=config.get("clip_grad"),
        device=device,
        seed=config.get("random_seed", 777),
        path=config["exp_path"],
        resume=resume_exp_name is not None,
        mixup=config.get("mixup"),
        cutmix=config.get("cutmix"),
        mixup_prob=config.get("mixup_prob"),
        # apply_ema=config.get("apply_ema"),
        # ema_start_epoch=config.get("ema_start_epoch", 0),
        # ema_decay=config.get("ema_decay", 0.9999),
    )

    # Load best model and evaluate
    model_filename = os.path.join(config["exp_path"], "best_f1.pth")
    model.load_state_dict(torch.load(model_filename, map_location="cpu"))
    _, testloader, _, _ = get_dataloaders(
        None,
        test_df,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
    )
    
    evaluate(model, trainloader, testloader, path=config["exp_path"], device=device)

    # Finish W&B run
    run_id = finish_wandb()
    if run_id is not None:
        logger.info("Setting the best scores in the W&B run summary.")
        set_best_scores_in_summary(
            run_or_path=f"{wandb_entity}/{wandb_project}/{run_id}",
            primary_score="val/F1",
            scores=lambda df: [col for col in df if col.startswith("val/")],
        )

    # HuggingFace Hub export
    def count_parameters(trained_model):
        return sum(p.numel() for p in trained_model.parameters() if p.requires_grad)

    if hfhub_owner is not None:
        try:
            logger.info("Trying to save the model to HuggingFace hub...")
            num_params = count_parameters(model)
            config["mean"] = model_mean
            config["std"] = model_std
            config["params"] = np.round(num_params / 1_000_000, 1)
            export_model_to_huggingface_hub_from_checkpoint(
                config=config, repo_owner=hfhub_owner, saved_model="f1"
            )
            logger.info("Model sucessfully saved to HuggingFace hub!")
        except Exception as e:
            print(f"Exception during export: {e}")


if __name__ == "__main__":
    train_clf()
