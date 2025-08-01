import argparse
import ast
import os
from typing import Tuple, Any, Dict
import yaml
import numpy as np
from types import SimpleNamespace

import torchvision.transforms as T

import pandas as pd
# get root directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset.fungi import FungiTastic
from dataset.utils.mask_vis import (
    visualize_binary_mask,
    visualize_semantic_masks,
    visualize_instance_masks
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class MaskFungiTastic(FungiTastic):
    def __init__(self, root: str, data_subset: str = 'Mini', split: str = 'val', size: str = '300',
                 task: str = 'closed', transform: T.Compose = None, seg_task: str = 'binary', debug: bool = False, **kwargs):
        """
        Initialize MaskFungiTastic dataset.
        
        Args:
            root: Root directory path
            data_subset: Data subset name (default: 'Mini')
            split: Dataset split ('val' or 'train')
            size: Image size (default: '300')
            task: Task type (default: 'closed')
            transform: Image transformations
            seg_task: Segmentation task type. Options:
                - 'binary': Foreground-background segmentation (merged masks)
                - 'semantic': Semantic segmentation (masks merged by label)
                - 'instance': Instance segmentation (individual mask parts), "Instance-aware Semantic Part Segmentation"
            debug: If True, visualize mask parts during merging
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            root=root,
            data_subset=data_subset,
            split=split,
            size=size,
            task=task,
            transform=transform,
            **kwargs
        )
        
        self.seg_task = seg_task
        self.debug = debug
        split_str = 'Validation' if split == 'val' else 'Train'
        
        # Load multiple semantic segmentation masks per image
        gt_masks = pd.read_parquet(os.path.join(root, 'masks', f'FungiTastic-Mini-{split_str}Masks.parquet'))
        gt_masks.rename(columns={'file_name': 'filename'}, inplace=True)
        # Convert RLE strings to lists if they're stored as strings
        if 'rle' in gt_masks.columns and isinstance(gt_masks['rle'].iloc[0], str):
            gt_masks['rle'] = gt_masks['rle'].apply(ast.literal_eval)
        
        # Group by filename to collect all mask parts for each image
        # Since there are multiple entries per filename, we need to aggregate them
        gt_masks = gt_masks.groupby('filename').agg({
            'rle': list,  # Collect all RLE masks for this image
            'label': list,  # Collect all labels for this image (needed for semantic/instance segmentation)
            'height': 'first',  # Keep first height (should be same for all parts of same image)
            'width': 'first'   # Keep first width (should be same for all parts of same image)
        }).reset_index()
        
        self.df = self.df.merge(gt_masks, on='filename', how='inner')
        
        # Generate consistent color mapping for all unique labels in the dataset
        self.label_colors = self._generate_label_colors()


    def rle_to_mask(self, rle_points: list, height: int, width: int) -> np.ndarray:
        """Decode data compressed with CVAT-based Run-Length Encoding (RLE) and return boolean image mask.
        """
    
        mask = np.zeros(height * width, dtype=np.uint8)

        # Extract the RLE encoding
        rle_counts = rle_points[:-4]  # Exclude the last four points which are bounding box coordinates

        # Decode the RLE into the mask
        current_position = 0
        current_value = 0

        for rle_count in rle_counts:
            mask[current_position:current_position + rle_count] = current_value
            current_position += rle_count
            current_value = 1 - current_value  # Toggle between 0 and 1

        # Reshape the flat mask back to 2D and convert to boolean
        mask = mask.reshape((height, width)).astype(bool)

        return mask

    def _generate_label_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Generate consistent color mapping for all unique labels in the dataset.
        
        Returns:
            Dictionary mapping label names to RGB colors (0-1 range)
        """
        # Collect all unique labels from the dataset
        all_labels = set()
        for _, row in self.df.iterrows():
            if 'label' in row and isinstance(row['label'], list):
                all_labels.update(row['label'])
        
        # Sort labels for consistent ordering
        sorted_labels = sorted(list(all_labels))
        
        # Generate distinct colors for each label
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Use discrete colors from Set3 colormap (no interpolation)
        # Set3 has 12 distinct colors, we'll cycle through them if needed
        base_colors = plt.cm.Set3(np.arange(12))  # Get the 12 discrete colors
        
        # Create color mapping
        label_colors = {}
        for i, label in enumerate(sorted_labels):
            color_idx = i % len(base_colors)  # Cycle through colors if more labels than colors
            label_colors[label] = tuple(base_colors[color_idx][:3])  # RGB values in 0-1 range
        
        print(f"Generated color mapping for {len(sorted_labels)} unique labels: {sorted_labels}")
        return label_colors

    def merge_semantic_masks(self, mask_data: list, height: int, width: int, debug: bool = False) -> np.ndarray:
        """
        Merge multiple semantic segmentation masks into a single boolean foreground mask.
        
        Args:
            mask_data: list of RLE encodings for all mask parts of this image
            height: Image height
            width: Image width
            debug: If True, visualize all mask parts and final merged mask
            
        Returns:
            Boolean mask where foreground pixels are True and background pixels are False
        """
        # Initialize binary mask
        binary_mask = np.zeros((height, width), dtype=bool)
        
        # For debug visualization
        if debug:
            import matplotlib.pyplot as plt
            part_masks = []
        
        # Iterate through all RLE masks for this image
        for i, rle_mask in enumerate(mask_data):
            if len(rle_mask) > 0:
                # array to list
                rle_mask = rle_mask.tolist()
                # Decode this semantic mask part
                part_mask = self.rle_to_mask(rle_mask, height, width)
                
                # Store for debug visualization
                if debug:
                    part_masks.append(part_mask)
                
                # Add to binary mask (any foreground pixel becomes True)
                binary_mask = np.logical_or(binary_mask, part_mask)
        
        # Debug visualization
        if debug and len(part_masks) > 0:
            num_parts = len(part_masks)
            cols = min(4, num_parts + 1)  # Max 4 columns, +1 for final mask
            rows = (num_parts + 1 + cols - 1) // cols  # Calculate needed rows
            
            plt.figure(figsize=(cols * 4, rows * 3))
            
            # Show individual parts
            for i, part_mask in enumerate(part_masks):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(part_mask, cmap='gray')
                plt.title(f'Part {i+1}')
                plt.axis('off')
            
            # Show final merged mask
            plt.subplot(rows, cols, num_parts + 1)
            plt.imshow(binary_mask, cmap='gray')
            plt.title('Final Merged Mask')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return binary_mask

    def merge_masks_by_label(self, mask_data: list, label_data: list, height: int, width: int) -> dict:
        """
        Merge masks by their semantic labels for semantic segmentation.
        
        Args:
            mask_data: list of RLE encodings for all mask parts of this image
            label_data: list of labels corresponding to each mask part
            height: Image height
            width: Image width
            
        Returns:
            Dictionary mapping label to merged mask
        """
        # Group masks by label
        label_to_masks = {}
        for rle_mask, label in zip(mask_data, label_data):
            if label not in label_to_masks:
                label_to_masks[label] = []
            label_to_masks[label].append(rle_mask)
        
        # Merge masks for each label
        merged_masks = {}
        for label, rle_masks in label_to_masks.items():
            # Initialize mask for this label
            label_mask = np.zeros((height, width), dtype=bool)
            
            # Merge all masks for this label
            for rle_mask in rle_masks:
                if len(rle_mask) > 0:
                    if hasattr(rle_mask, 'tolist'):
                        rle_mask = rle_mask.tolist()
                    part_mask = self.rle_to_mask(rle_mask, height, width)
                    label_mask = np.logical_or(label_mask, part_mask)
            
            merged_masks[label] = label_mask
        
        return merged_masks

    def create_instance_masks(self, mask_data: list, label_data: list, height: int, width: int) -> list:
        """
        Create individual instance masks for instance segmentation.
        
        Args:
            mask_data: list of RLE encodings for all mask parts of this image
            label_data: list of labels corresponding to each mask part
            height: Image height
            width: Image width
            
        Returns:
            List of tuples (mask, label) for each instance
        """
        instance_masks = []
        
        for rle_mask, label in zip(mask_data, label_data):
            if len(rle_mask) > 0:
                if hasattr(rle_mask, 'tolist'):
                    rle_mask = rle_mask.tolist()
                mask = self.rle_to_mask(rle_mask, height, width)
                instance_masks.append((mask, label))
        
        return instance_masks

    def __getitem__(self, item):
        image, class_id, file_path = super().__getitem__(item)

        meta = self.df.iloc[item]
        
        # Multiple semantic segmentation masks - use the last one which is already merged
        mask_data = meta['rle']  # This is now a list of RLE masks for this image (aggregated by groupby)
        label_data = meta['label']  # This is now a list of labels for this image (aggregated by groupby)
        
        if self.seg_task == 'binary':
            # For binary segmentation, use the last mask which is already merged
            last_rle = mask_data[-1]  # Last mask is the merged one
            if hasattr(last_rle, 'tolist'):
                last_rle = last_rle.tolist()
            mask = self.rle_to_mask(last_rle, meta.height, meta.width)
            return image, mask, class_id, file_path, label_data
            
        elif self.seg_task == 'semantic':
            # For semantic segmentation, merge masks by label
            semantic_masks = self.merge_masks_by_label(mask_data, label_data, meta.height, meta.width)
            return image, semantic_masks, class_id, file_path, label_data
            
        elif self.seg_task == 'instance':
            # For instance segmentation, return individual masks
            instance_masks = self.create_instance_masks(mask_data, label_data, meta.height, meta.width)
            return image, instance_masks, class_id, file_path, label_data
            
        else:
            raise ValueError(f"Unknown seg_task: {self.seg_task}. Use 'binary', 'semantic', or 'instance'")

    def show_sample(self, idx: int) -> None:
        """
        Display a sample image with its class name and ID.

        Args:
            idx (int): Index of the sample to display.
        """
        image, masks, category_id, file_path, label_data = self.__getitem__(idx)
        class_name = self.category_id2label[category_id] if category_id is not None else '[TEST]'
        
        if self.seg_task == 'binary':
            visualize_binary_mask(image, masks, class_name)
        elif self.seg_task == 'semantic':
            visualize_semantic_masks(image, masks, class_name, label_colors=self.label_colors)
        elif self.seg_task == 'instance':
            visualize_instance_masks(image, masks, class_name, label_colors=self.label_colors)
        else:
            raise ValueError(f"Unknown seg_task: {self.seg_task}")
    




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Use LaTeX-like font for paper visualization if needed and LaTeX is installed
    if False:  # Change to True if you want LaTeX-like font
        plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    # Configuration parameters
    config_path = os.path.join(SCRIPT_DIR, '../baselines/segmentation/config/seg.yaml')
    split = 'val'  # 'val' or 'train'
    seg_task = 'instance'  # 'binary', 'semantic', or 'instance'
    debug = False  # Enable debug visualization of mask parts

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    dataset = MaskFungiTastic(
        root=cfg.data_path,
        split=split,
        size='300',
        task='closed',
        data_subset='Mini',
        transform=None,
        seg_task=seg_task,
        debug=debug,
    )

    for i in range(20, 25):
        dataset.show_sample(i)
    # dataset.show_sample(25)
    # print(dataset.df.iloc[25])


