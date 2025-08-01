import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image
import cv2

def get_label_color(label: str, label_colors: Dict[str, Tuple[float, float, float]] = None, index: int = None) -> Tuple[float, float, float]:
    """
    Get consistent color for a label.
    
    Args:
        label: Label name
        label_colors: Dictionary mapping labels to colors (from dataset)
        index: Index for color selection
        
    Returns:
        RGB color tuple (0-1 range for matplotlib)
    """
    if label_colors and label in label_colors:
        return label_colors[label]
    else:
        # Generate a color based on label hash for consistency
        import hashlib
        hash_val = int(hashlib.md5(label.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        return tuple(np.random.rand(3))


def resize_mask_to_image(mask, image_shape: Tuple[int, int], method: str = 'nearest') -> np.ndarray:
    """
    Resize mask to match image dimensions.
    
    Args:
        mask: Input mask (H, W) - boolean or numeric
        image_shape: Target image shape (H, W)
        method: Resizing method ('nearest', 'bilinear', 'cubic')
        
    Returns:
        Resized mask with same dtype as input
    """
    if mask.shape == image_shape:
        return mask
    
    # Convert to uint8 for resizing (preserve boolean behavior)
    mask_uint8 = mask.astype(np.uint8)
    
    # Choose interpolation method
    if method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'cubic':
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_NEAREST
    
    # Resize mask
    resized_mask = cv2.resize(mask_uint8, (image_shape[1], image_shape[0]), interpolation=interpolation)
    
    # Convert back to original dtype
    if mask.dtype == bool:
        return resized_mask > 0
    else:
        return resized_mask.astype(mask.dtype)


def get_image_shape(image) -> Tuple[int, int]:
    """
    Get the shape (height, width) of an image.
    
    Args:
        image: PIL Image, numpy array, or tensor
        
    Returns:
        Tuple of (height, width)
    """
    if isinstance(image, Image.Image):
        return image.size[1], image.size[0]  # PIL uses (width, height)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            return image.shape[0], image.shape[1]  # (height, width)
        else:
            return image.shape[0], image.shape[1]  # (height, width)
    elif hasattr(image, 'shape'):  # Tensor
        if len(image.shape) == 3:
            return image.shape[1], image.shape[2]  # (height, width) for CHW
        else:
            return image.shape[0], image.shape[1]  # (height, width)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def visualize_binary_mask(image, mask, class_name: str) -> None:
    """
    Visualize binary segmentation mask.
    
    Args:
        image: Input image (PIL Image, numpy array, or tensor)
        mask: Binary mask (H, W) - boolean or numeric
        class_name: Class name for title
    """
    # Resize mask to match image dimensions
    image_shape = get_image_shape(image)
    mask = resize_mask_to_image(mask, image_shape)
    
    from torchvision.utils import draw_segmentation_masks
    import torch
    from PIL import Image
    
    # Convert image to tensor format for torchvision
    if isinstance(image, Image.Image):
        # Convert PIL Image to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # HWC to CHW
    elif isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
    elif torch.is_tensor(image):
        image_tensor = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Convert mask to boolean if needed and then to tensor
    if mask.dtype != bool:
        mask = mask.astype(bool)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Add batch dimension
    result = draw_segmentation_masks(
        image_tensor, 
        mask_tensor, 
        alpha=0.7, 
        colors=["red"]
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title(f"Image - {class_name}")
    axes[0].axis('off')
    
    axes[1].imshow(result.permute(1, 2, 0))  # CHW to HWC
    axes[1].set_title(f"Binary Segmentation - {class_name}")
    axes[1].axis('off')

    plt.show()





def visualize_semantic_masks(image, masks: Dict[str, np.ndarray], 
                           class_name: str, alpha: float = 0.6, 
                           label_colors: Dict[str, Tuple[float, float, float]] = None) -> None:
    """
    Visualize semantic segmentation masks as transparent overlay.
    
    Args:
        image: Input image (PIL Image, numpy array, or tensor)
        masks: Dictionary mapping label to mask (H, W) - boolean or numeric
        class_name: Class name for title
        alpha: Transparency level for overlay
    """
    if len(masks) == 0:
        print("No semantic masks to visualize")
        return
    
    # Resize all masks to match image dimensions
    image_shape = get_image_shape(image)
    resized_masks = {}
    for label, mask in masks.items():
        resized_masks[label] = resize_mask_to_image(mask, image_shape)
    
    # Create colored visualization with transparent overlay
    semantic_overlay = np.zeros((*image_shape, 3), dtype=np.uint8)
    
    # Create figure with original image and overlay
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image - {class_name}")
    axes[0].axis('off')
    
    # Create colored overlay for all masks (excluding the last one - fruiting body)
    for i, (label, mask) in enumerate(list(resized_masks.items())[:-1]):
        color = get_label_color(label, label_colors, i)
        color_uint8 = (np.array(color) * 255).astype(np.uint8)
        semantic_overlay[mask] = color_uint8
        
        # Add contour around the mask for better distinction
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Draw contour with cyan color (BGR format)
            cv2.drawContours(semantic_overlay, [contour], -1, (255, 0, 255), 2)
    
    # Show overlay with transparency
    axes[1].imshow(image)
    axes[1].imshow(semantic_overlay, alpha=alpha)
    axes[1].set_title(f"Semantic Segmentation Overlay")
    axes[1].axis('off')
    
    # Add legend showing labels and their colors (excluding the last one - fruiting body)
    legend_elements = []
    for i, (label, _) in enumerate(list(resized_masks.items())[:-1]):
        color = get_label_color(label, label_colors, i)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=f"{label}"))
    
    # Place legend above the plot
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=min(len(legend_elements), 4))
    
    plt.tight_layout()
    plt.show()


def visualize_instance_masks(image, masks: List[Tuple[np.ndarray, str]], 
                           class_name: str, alpha: float = 0.6,
                           label_colors: Dict[str, Tuple[float, float, float]] = None) -> None:
    """
    Visualize instance segmentation masks as transparent overlay.
    
    Args:
        image: Input image (PIL Image, numpy array, or tensor)
        masks: List of tuples (mask, label) for each instance - masks are boolean or numeric
        class_name: Class name for title
        alpha: Transparency level for overlay
    """
    if len(masks) == 0:
        print("No instance masks to visualize")
        return
    
    # Resize all masks to match image dimensions
    image_shape = get_image_shape(image)
    resized_masks = []
    for mask, label in masks:
        resized_masks.append((resize_mask_to_image(mask, image_shape), label))
    
    # Create colored visualization for instances with transparent overlay
    instance_overlay = np.zeros((*image_shape, 3), dtype=np.uint8)
    
    # Create figure with original image and overlay
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image - {class_name}")
    axes[0].axis('off')
    
    # Create colored overlay for all instances (excluding the last one - fruiting body)
    for i, (mask, label) in enumerate(resized_masks[:-1]):
        color = get_label_color(label, label_colors, i)
        color_uint8 = (np.array(color) * 255).astype(np.uint8)
        instance_overlay[mask] = color_uint8
        
        # Add contour around the mask for better instance distinction
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Draw contour with cyan color (BGR format)
            cv2.drawContours(instance_overlay, [contour], -1, (255, 0, 255), 2)
    
    # Show overlay with transparency
    axes[1].imshow(image)
    axes[1].imshow(instance_overlay, alpha=alpha)
    axes[1].set_title(f"Instance Segmentation Overlay")
    axes[1].axis('off')
    
    # Add legend showing instances and their labels (excluding the last one - fruiting body)
    legend_elements = []
    for i, (_, label) in enumerate(resized_masks[:-1]):
        color = get_label_color(label, label_colors, i)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=f"{label}"))
    
    # Place legend above the plot
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=min(len(legend_elements), 4))
    
    plt.tight_layout()
    plt.show()


def visualize_masks_grid(image, masks: Union[Dict[str, np.ndarray], List[Tuple[np.ndarray, str]]], 
                        class_name: str, seg_task: str = 'semantic', max_cols: int = 4) -> None:
    """
    Visualize masks in a grid layout showing individual masks.
    
    Args:
        image: Input image (PIL Image, numpy array, or tensor)
        masks: Either dict of semantic masks or list of instance masks
        class_name: Class name for title
        seg_task: 'semantic' or 'instance'
        max_cols: Maximum number of columns in grid
    """
    if seg_task == 'semantic':
        if len(masks) == 0:
            print("No semantic masks to visualize")
            return
        
        # Calculate grid layout
        num_masks = len(masks)
        cols = min(max_cols, num_masks)
        rows = (num_masks + cols - 1) // cols
        
        # Create figure
        fig = plt.figure(figsize=(cols * 4 + 8, max(6, rows * 3)))
        
        # Original image
        plt.subplot(rows + 1, cols + 1, 1)
        plt.imshow(image)
        plt.title(f"Image - {class_name}")
        plt.axis('off')
        
        # Show individual semantic masks with labels
        for i, (label, mask) in enumerate(masks.items()):
            plt.subplot(rows + 1, cols + 1, i + 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f"{label}")
            plt.axis('off')
    
    elif seg_task == 'instance':
        if len(masks) == 0:
            print("No instance masks to visualize")
            return
        
        # Calculate grid layout
        num_masks = len(masks)
        cols = min(max_cols, num_masks)
        rows = (num_masks + cols - 1) // cols
        
        # Create figure
        fig = plt.figure(figsize=(cols * 4 + 8, max(6, rows * 3)))
        
        # Original image
        plt.subplot(rows + 1, cols + 1, 1)
        plt.imshow(image)
        plt.title(f"Image - {class_name}")
        plt.axis('off')
        
        # Show individual instance masks with labels
        for i, (mask, label) in enumerate(masks):
            plt.subplot(rows + 1, cols + 1, i + 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Instance {i+1}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_semantic_overlay(masks: Dict[str, np.ndarray], image_shape: Optional[Tuple[int, int]] = None,
                           label_colors: Dict[str, Tuple[float, float, float]] = None) -> np.ndarray:
    """Create colored overlay for semantic masks."""
    if len(masks) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Use provided image shape or infer from first mask
    if image_shape is None:
        image_shape = list(masks.values())[0].shape
    
    overlay = np.zeros((*image_shape, 3), dtype=np.uint8)
    
    for i, (label, mask) in enumerate(masks.items()):
        color = get_label_color(label, label_colors, i)
        color_uint8 = (np.array(color) * 255).astype(np.uint8)
        # Resize mask if needed
        if mask.shape != image_shape:
            mask = resize_mask_to_image(mask, image_shape)
        overlay[mask] = color_uint8
        
        # Add contour around the mask for better distinction
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Draw contour with cyan color (BGR format)
            cv2.drawContours(overlay, [contour], -1, color=(255, 0, 255), thickness=2)
    
    return overlay


def create_instance_overlay(masks: List[Tuple[np.ndarray, str]], image_shape: Optional[Tuple[int, int]] = None,
                           label_colors: Dict[str, Tuple[float, float, float]] = None) -> np.ndarray:
    """Create colored overlay for instance masks."""
    if len(masks) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Use provided image shape or infer from first mask
    if image_shape is None:
        image_shape = masks[0][0].shape
    
    overlay = np.zeros((*image_shape, 3), dtype=np.uint8)
    
    for i, (mask, label) in enumerate(masks):
        color = get_label_color(label, label_colors, i)
        color_uint8 = (np.array(color) * 255).astype(np.uint8)
        # Resize mask if needed
        if mask.shape != image_shape:
            mask = resize_mask_to_image(mask, image_shape)
        overlay[mask] = color_uint8
        
        # Add contour around the mask for better instance distinction
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Draw contour with cyan color (BGR format)
            cv2.drawContours(overlay, [contour], -1, (255, 0, 255), 2)
    
    return overlay 