from typing import List

import numpy as np
from PIL import Image
import os

from scripts.baselines.seg.lang_sam import LangSAM


class MaskGenerator():
    def predict_batch(self, pil_images: List[Image.Image]) -> (List[np.array], List[dict]):
        segs, extras = []
        for pil_im in pil_images:
            mask, extra = self.predict(pil_im)
            segs.append(mask)
            extras.append(extra)
        return segs, extras

    def predict(self, pil_image: Image.Image) -> (np.array, dict):
        pass


class LangSAMMaskGenerator(MaskGenerator):
    def __init__(self, ckpt_path: str, text_prompt: str, dataframes_dir: str):
        sam_ckpt_h = os.path.join(ckpt_path, 'sam_vit_h_4b8939.pth')
        self.langsam_predictor = LangSAM(ckpt_path=sam_ckpt_h)
        self.text_prompt = text_prompt
        self.dataframes_dir = dataframes_dir
        self.name = f"lang_sam"

    def predict(self, image_pil):
        # there may be multiple masks if grounding dino returns multiple boxes
        masks, boxes, phrases, gdino_confs, seg_confs = self.langsam_predictor.predict(image_pil, self.text_prompt)

        if len(masks) == 0 or len(gdino_confs) == 0:
            mask = np.zeros(image_pil.size[::-1], dtype=np.uint8)
        elif len(masks) != 1 or len(gdino_confs) != 1:
            # merge them into a single mask
            mask = np.zeros(image_pil.size[::-1], dtype=np.uint8)
            for m in masks:
                mask += m.cpu().numpy()
        else:
            mask = masks[0].cpu().numpy()

        extra = {
            "confs_gdino": gdino_confs,
            "confs_seg": seg_confs,
            "n_inst": len(masks),
        }
        return mask, extra
