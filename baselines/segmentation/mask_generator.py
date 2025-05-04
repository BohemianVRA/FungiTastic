from typing import List
import os
import numpy as np
from PIL import Image
import torch

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download

from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

# Mapping of SAM model types to their corresponding checkpoint filenames
MODEL2FILENAME = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth"
}

# URLs for downloading SAM model checkpoints
SAM_MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

class GDINOSAM:
    """
    A wrapper class that combines Grounding-DINO and SAM (Segment Anything Model) for end-to-end object detection and segmentation.
    
    This class provides a unified interface for:
    1. Detecting objects using Grounding-DINO based on text prompts
    2. Generating segmentation masks using SAM for the detected objects
    
    Attributes:
        text_prompt (str): The text prompt used for object detection
        dataframes_dir (str): Directory for storing dataframes (if needed)
        name (str): Identifier for the model
        device (str): Device to run the models on ('cuda' or 'cpu')
        sam_predictor (SamPredictor): SAM model predictor instance
        groundingdino: Grounding-DINO model instance
        image_transform: Image transformation pipeline for Grounding-DINO
    """

    def __init__(
        self,
        ckpt_path: str,
        text_prompt: str,
        dataframes_dir: str = "",
        sam_type: str = "vit_h",
        dino_version: str = "B",
        return_prompts: bool = False,
    ):
        """
        Initialize the GDINOSAM model.

        Args:
            ckpt_path (str): Path to store/load model checkpoints
            text_prompt (str): Text prompt for object detection
            dataframes_dir (str, optional): Directory for dataframes. Defaults to "".
            sam_type (str, optional): Type of SAM model to use ('vit_h', 'vit_l', or 'vit_b'). Defaults to "vit_h".
            dino_version (str, optional): Version of Grounding-DINO to use ('B' or 'T'). Defaults to "B".
            return_prompts (bool, optional): Whether to return prompts in predictions. Defaults to False.
        """
        self.text_prompt = text_prompt
        self.dataframes_dir = dataframes_dir
        self.name = "gdino_sam"
        self.device = "cpu"
        self._build_sam(sam_type, ckpt_path)
        self._build_dino(dino_version, return_prompts)
        self.image_transform = get_image_transform()

    def predict(self, image_pil: Image.Image, box_thr: float = 0.3, txt_thr: float = 0.25):
        """
        Generate segmentation masks for objects in the image based on the text prompt.

        Args:
            image_pil (Image.Image): Input PIL image
            box_thr (float, optional): Confidence threshold for box detection. Defaults to 0.3.
            txt_thr (float, optional): Confidence threshold for text matching. Defaults to 0.25.

        Returns:
            tuple: (mask, extra)
                - mask (np.ndarray): Binary mask of detected objects
                - extra (dict): Additional information including:
                    - confs_gdino: Grounding-DINO confidence scores
                    - confs_seg: SAM segmentation confidence scores
                    - n_inst: Number of detected instances
                    - phrases: Detected phrases
                    - boxes: Bounding boxes of detected objects
        """
        # ---- 1. Grounding-DINO: boxes, logits, phrases ----------------- #
        boxes, gdino_logits, phrases = self._predict_gdino(image_pil, self.text_prompt, box_thr, txt_thr)

        # ---- 2. SAM segmentation for each box -------------------------- #
        masks, seg_confs = self._predict_sam(image_pil, boxes)

        # ---- 3. Merge masks (or create empty) -------------------------- #
        if len(masks) == 0 or len(gdino_logits) == 0:
            mask = np.zeros(image_pil.size[::-1], dtype=np.uint8)
        else:
            mask = np.zeros(image_pil.size[::-1], dtype=np.uint8)
            for m in masks:
                mask = np.maximum(mask, m.cpu().numpy())

        extra = {
            "confs_gdino": gdino_logits,
            "confs_seg" : seg_confs,
            "n_inst"    : len(masks),
            "phrases"   : phrases,
            "boxes"     : boxes,
        }
        return mask, extra

    def _build_sam(self, sam_type: str, ckpt_root: str):
        """
        Initialize and load the SAM model.

        Args:
            sam_type (str): Type of SAM model to use
            ckpt_root (str): Root directory for model checkpoints
        """
        ckpt_file = os.path.join(ckpt_root, MODEL2FILENAME[sam_type])
        if not os.path.exists(ckpt_file):
            os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
            torch.hub.download_url_to_file(SAM_MODEL_URLS[sam_type], ckpt_file)

        sam = sam_model_registry[sam_type](ckpt_file).to(self.device)
        self.sam_predictor = SamPredictor(sam)

    def _build_dino(self, version: str, return_prompts: bool):
        """
        Initialize and load the Grounding-DINO model.

        Args:
            version (str): Version of Grounding-DINO to use ('B' or 'T')
            return_prompts (bool): Whether to return prompts in predictions
        """
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = (
            "groundingdino_swinb_cogcoor.pth" if version == "B" else "groundingdino_swint_ogc.pth"
        )
        ckpt_cfg = (
            "GroundingDINO_SwinB.cfg.py" if version == "B" else "GroundingDINO_SwinT_OGC.py"
        )
        self.groundingdino = load_model_hf(
            repo_id=ckpt_repo_id, filename=ckpt_filename, ckpt_config_filename=ckpt_cfg
        )
        self.return_prompts = return_prompts

    def _predict_sam(self, image_pil: Image.Image, boxes: torch.Tensor):
        """
        Generate segmentation masks using SAM for the given bounding boxes.

        Args:
            image_pil (Image.Image): Input PIL image
            boxes (torch.Tensor): Bounding boxes from Grounding-DINO

        Returns:
            tuple: (masks, confidences)
                - masks: Segmentation masks for each box
                - confidences: Confidence scores for each mask
        """
        if len(boxes) == 0:
            return torch.tensor([]), []

        img_arr = np.asarray(image_pil)
        self.sam_predictor.set_image(img_arr)
        tb = self.sam_predictor.transform.apply_boxes_torch(boxes, img_arr.shape[:2])
        masks, iou_preds, _ = self.sam_predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=tb.to(self.device), multimask_output=False
        )
        return masks.cpu().squeeze(1), iou_preds.cpu().numpy()
    
    def _predict_gdino(self, image_pil, text_prompt, box_threshold, text_threshold):
        """
        Detect objects using Grounding-DINO based on the text prompt.

        Args:
            image_pil (Image.Image): Input PIL image
            text_prompt (str): Text prompt for object detection
            box_threshold (float): Confidence threshold for box detection
            text_threshold (float): Confidence threshold for text matching

        Returns:
            tuple: (boxes, logits, phrases)
                - boxes: Detected bounding boxes
                - logits: Confidence scores for detections
                - phrases: Detected phrases
        """
        image_trans, _ = self.image_transform(image_pil, None)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         remove_combined=self.return_prompts,
                                         device='cpu')
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases


def load_model_hf(repo_id, filename, ckpt_config_filename=None, model_config_path=None, device='cpu'):
    """
    Load a model from Hugging Face Hub.

    Args:
        repo_id (str): Hugging Face repository ID
        filename (str): Name of the model checkpoint file
        ckpt_config_filename (str, optional): Name of the config file in the repo. Defaults to None.
        model_config_path (str, optional): Local path to model config. Defaults to None.
        device (str, optional): Device to load the model on. Defaults to 'cpu'.

    Returns:
        model: Loaded model instance
    """
    assert model_config_path is not None or ckpt_config_filename is not None, "Please provide either model config path or checkpoint config filename"
    if model_config_path is None:
        config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    else:
        config_file = model_config_path

    args = SLConfig.fromfile(config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def get_image_transform():
    """
    Get the image transformation pipeline for Grounding-DINO.

    Returns:
        T.Compose: Image transformation pipeline
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform