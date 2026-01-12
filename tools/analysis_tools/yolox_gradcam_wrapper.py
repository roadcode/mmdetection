# Copyright (c) OpenMMLab. All rights reserved.
"""YOLOX Grad-CAM Wrapper.

This module provides wrapper classes and utilities to integrate YOLOX
object detection models with the pytorch-grad-cam library for generating
Class Activation Maps (CAM) visualizations.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed. "
          "Install it with: pip install grad-cam")


class DetectionTarget:
    """Custom target for object detection Grad-CAM.
    
    This class defines the target for gradient computation in object detection.
    For each detection box, we compute gradients with respect to its
    classification score.
    
    Args:
        bboxes (np.ndarray): Detection bounding boxes in [x1, y1, x2, y2] format.
        labels (np.ndarray): Class labels for each detection.
        scores (np.ndarray): Confidence scores for each detection.
        category (int, optional): Specific category to target. If None, uses
            the predicted category for each box. Defaults to None.
    """
    
    def __init__(self, 
                 bboxes: np.ndarray,
                 labels: np.ndarray, 
                 scores: np.ndarray,
                 category: Optional[int] = None):
        self.bboxes = bboxes
        self.labels = labels
        self.scores = scores
        self.category = category
        
    def __call__(self, model_output):
        """Compute target score for gradient computation.
        
        Args:
            model_output: Model output containing detection predictions.
            
        Returns:
            torch.Tensor: Target score for backpropagation.
        """
        # For object detection, we sum the classification scores
        # of all detected objects (or specific category if specified)
        if isinstance(model_output, (list, tuple)):
            # Handle multi-scale outputs
            output = model_output[0] if len(model_output) > 0 else model_output
        else:
            output = model_output
            
        return output.sum()


class YOLOXGradCAMWrapper(torch.nn.Module):
    """Wrapper for YOLOX model to work with pytorch-grad-cam.
    
    This wrapper adapts YOLOX models to be compatible with the
    pytorch-grad-cam library by providing a standard forward interface
    and handling the extraction of intermediate features.
    
    Args:
        model (torch.nn.Module): YOLOX detection model.
        target_layers (List[torch.nn.Module]): List of layers to extract
            activations from for CAM computation.
    """
    
    def __init__(self, model: torch.nn.Module, target_layers: List[torch.nn.Module]):
        super().__init__()
        self.model = model
        self.target_layers = target_layers
        
    def forward(self, x):
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            Output from the model's forward pass.
        """
        return self.model(x, mode='tensor')


def get_target_layer_from_name(model, layer_name: str) -> Optional[torch.nn.Module]:
    """Get target layer from model by name.
    
    Args:
        model: The model to extract layer from.
        layer_name (str): Name of the layer (e.g., 'neck.out_convs.2').
        
    Returns:
        torch.nn.Module or None: The target layer if found.
    """
    parts = layer_name.split('.')
    current = model
    
    try:
        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current
    except (AttributeError, IndexError, TypeError):
        return None


def reshape_transform_yolox(tensor, height=None, width=None):
    """Reshape transform for YOLOX multi-scale features.
    
    YOLOX neck outputs multi-scale features. This function handles
    the transformation of these features for CAM computation.
    
    Args:
        tensor: Feature tensor from YOLOX.
        height (int, optional): Target height for resizing.
        width (int, optional): Target width for resizing.
        
    Returns:
        Reshaped tensor suitable for CAM computation.
    """
    if isinstance(tensor, (list, tuple)):
        # Handle multi-scale features - use the largest resolution
        tensor = tensor[0] if len(tensor) > 0 else tensor
    
    # Input tensor shape: (batch, channels, height, width)
    # No reshape needed for standard conv layers
    return tensor


def show_cam_on_detection(image: np.ndarray,
                          grayscale_cam: np.ndarray,
                          bboxes: np.ndarray,
                          labels: np.ndarray,
                          scores: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          use_rgb: bool = True) -> np.ndarray:
    """Overlay CAM heatmap on image with detection boxes.
    
    Args:
        image (np.ndarray): Original image in RGB or BGR format.
        grayscale_cam (np.ndarray): Grayscale CAM heatmap [0-1].
        bboxes (np.ndarray): Detection boxes in [x1, y1, x2, y2] format.
        labels (np.ndarray): Class labels for each box.
        scores (np.ndarray): Confidence scores for each box.
        class_names (List[str], optional): List of class names.
        use_rgb (bool): Whether image is in RGB format.
        
    Returns:
        np.ndarray: Visualization image with CAM and detections.
    """
    # Normalize image to [0, 1]
    if image.dtype == np.uint8:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.astype(np.float32)
    
    # Overlay CAM on image
    cam_image = show_cam_on_image(image_float, grayscale_cam, use_rgb=use_rgb)
    
    # Draw bounding boxes and labels
    for bbox, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Draw box
        color = (0, 255, 0)  # Green in RGB
        cv2.rectangle(cam_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f'{class_names[label] if class_names else label}: {score:.2f}'
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw background rectangle
        cv2.rectangle(cam_image, 
                     (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1),
                     color, -1)
        
        # Draw text
        cv2.putText(cam_image, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return cam_image


def renormalize_cam_in_bounding_boxes(boxes: np.ndarray,
                                     image_float_np: np.ndarray,
                                     grayscale_cam: np.ndarray) -> np.ndarray:
    """Renormalize CAM within each bounding box for better visualization.
    
    This is a recommended practice for object detection CAM visualization.
    It re-normalizes the CAM values within each bounding box to [0, 1],
    making the important regions more visible.
    
    Args:
        boxes (np.ndarray): Bounding boxes in [x1, y1, x2, y2] format.
        image_float_np (np.ndarray): Original image as float [0-1].
        grayscale_cam (np.ndarray): Original grayscale CAM.
        
    Returns:
        np.ndarray: Renormalized CAM within bounding boxes.
    """
    renormalized_cam = np.zeros_like(grayscale_cam, dtype=np.float32)
    
    for x1, y1, x2, y2 in boxes.astype(int):
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(grayscale_cam.shape[1], x2)
        y2 = min(grayscale_cam.shape[0], y2)
        
        # Extract CAM within box
        cam_box = grayscale_cam[y1:y2, x1:x2]
        
        if cam_box.size > 0:
            # Renormalize within box
            cam_min = cam_box.min()
            cam_max = cam_box.max()
            
            if cam_max > cam_min:
                renormalized_cam[y1:y2, x1:x2] = \
                    (cam_box - cam_min) / (cam_max - cam_min)
            else:
                renormalized_cam[y1:y2, x1:x2] = cam_box
    
    return renormalized_cam


def get_default_target_layers(model, model_type='yolox'):
    """Get default target layers for YOLOX models.
    
    Args:
        model: The YOLOX model.
        model_type (str): Type of model (currently only 'yolox' supported).
        
    Returns:
        List[torch.nn.Module]: List of recommended target layers.
    """
    if model_type.lower() == 'yolox':
        # For YOLOX, use the last output conv layer from neck
        # This gives us the final feature maps before the detection head
        try:
            # Try to get the last out_conv from YOLOXPAFPN neck
            if hasattr(model, 'neck') and hasattr(model.neck, 'out_convs'):
                # Return the last scale's output conv
                return [model.neck.out_convs[-1]]
            elif hasattr(model, 'backbone'):
                # Fallback to last backbone layer
                if hasattr(model.backbone, 'stage4'):
                    return [model.backbone.stage4]
        except AttributeError:
            pass
    
    # General fallback - try to find a reasonable layer
    print("Warning: Could not find default target layer. "
          "Please specify target layers manually.")
    return []
