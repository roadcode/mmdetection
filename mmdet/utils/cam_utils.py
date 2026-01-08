# Copyright (c) OpenMMLab. All rights reserved.
"""CAM (Class Activation Mapping) utilities for visualization.

This module provides implementations of various CAM methods including:
- EigenCAM: PCA-based activation mapping (gradient-free, fast)
- GradCAM: Gradient-weighted class activation mapping

References:
    - EigenCAM: Based on Principal Component Analysis of feature maps
    - GradCAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union


class BaseCAM:
    """Base class for CAM methods.

    Args:
        model: The neural network model
        target_layer (str): Name of the target layer to visualize
    """

    def __init__(self, model, target_layer: str = 'neck'):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture intermediate features.

        Returns:
            list: List of hook handles
        """
        # Get target layer
        target_module = self._get_target_layer()

        # Register forward hook
        forward_handle = target_module.register_forward_hook(
            self._forward_hook)
        self.hooks.append(forward_handle)

        return self.hooks

    def _forward_hook(self, module, input, output):
        """Forward hook to save features.

        Args:
            module: The layer module
            input: Input to the layer
            output: Output from the layer
        """
        self.features = output

    def _get_target_layer(self):
        """Get the target layer module by name.

        Returns:
            torch.nn.Module: The target layer

        Raises:
            ValueError: If target layer is not found
        """
        for name, module in self.model.named_modules():
            if self.target_layer in name:
                return module
        raise ValueError(f"Layer {self.target_layer} not found. "
                        f"Available layers: {[name for name, _ in self.model.named_modules()]}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __del__(self):
        """Clean up hooks when object is destroyed."""
        self.remove_hooks()


class EigenCAM(BaseCAM):
    """EigenCAM implementation using PCA.

    EigenCAM applies Principal Component Analysis (PCA) to feature maps
    and uses the first principal component as the activation map.

    Advantages:
        - No gradient computation required (fast)
        - Model-agnostic
        - Works well with object detectors

    Reference:
        Based on the concept of using PCA for visualization in CNNs.

    Examples:
        >>> model = init_model('yolox_s.py', 'yolox_s.pth')
        >>> cam = EigenCAM(model, target_layer='neck')
        >>> img = torch.rand(1, 3, 640, 640)
        >>> cam_map = cam(img)
        >>> print(cam_map.shape)  # (640, 640) or similar
    """

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None):
        """Compute EigenCAM.

        Args:
            x (torch.Tensor): Input tensor of shape (1, 3, H, W)
            target_class (int, optional): Target class index.
                Not used in EigenCAM but kept for interface consistency.

        Returns:
            np.ndarray: CAM heatmap of shape (H, W)
        """
        # Register hooks
        self.register_hooks()

        # Forward pass
        with torch.no_grad():
            _ = self.model(x)

        # Get features
        features = self.features

        # Remove hooks
        self.remove_hooks()

        # Handle multi-scale features (YOLOX neck outputs 3 scales)
        if isinstance(features, (list, tuple)):
            cam_map = self._compute_multiscale_eigen_cam(features)
        else:
            cam_map = self._compute_single_eigen_cam(features)

        return cam_map

    def _compute_single_eigen_cam(self, features: torch.Tensor) -> np.ndarray:
        """Compute EigenCAM for single-scale features.

        Args:
            features (torch.Tensor): Feature map of shape (1, C, H, W)

        Returns:
            np.ndarray: CAM heatmap of shape (H, W), normalized to [0, 1]
        """
        from sklearn.decomposition import PCA

        # Remove batch dimension and convert to numpy
        features = features.squeeze(0).cpu().numpy()  # (C, H, W)

        C, H, W = features.shape

        # Reshape: (C, H, W) -> (C, H*W)
        features_2d = features.reshape(C, -1)

        # Transpose: (C, H*W) -> (H*W, C) for PCA
        features_2d = features_2d.T

        # Apply PCA to get first principal component
        pca = PCA(n_components=1)
        cam = pca.fit_transform(features_2d)

        # Reshape back to (H, W)
        cam = cam.reshape(H, W)

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def _compute_multiscale_eigen_cam(self, features_list: List[torch.Tensor]) -> np.ndarray:
        """Compute EigenCAM for multi-scale features with fusion.

        YOLOX neck outputs 3 scales:
            - P3: (1, C, 80, 80) for 640x640 input
            - P4: (1, C, 40, 40)
            - P5: (1, C, 20, 20)

        Args:
            features_list (list): List of feature maps at different scales

        Returns:
            np.ndarray: Fused CAM heatmap
        """
        # Get max size (usually the finest feature map)
        max_size = features_list[0].shape[-2:]  # (H, W)

        upsampled_cams = []
        for feat in features_list:
            # feat: (1, C, H_i, W_i)
            cam = self._compute_single_eigen_cam(feat)  # (H_i, W_i)
            cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()

            # Upsample to max size
            cam_upsampled = F.interpolate(
                cam_tensor, size=max_size, mode='bilinear', align_corners=False
            )
            upsampled_cams.append(cam_upsampled.squeeze())

        # Weighted average (finer features get higher weight)
        # P3 (finest) > P4 > P5 (coarsest)
        weights = torch.tensor([1.0, 0.8, 0.6])  # P3, P4, P5
        weights = weights[:, None, None].to(upsampled_cams[0].device)

        stacked = torch.stack(upsampled_cams)
        weighted = stacked * weights
        fused = (weighted.sum(dim=0) / weights.sum()).numpy()

        return fused


class GradCAM(BaseCAM):
    """Grad-CAM implementation.

    Grad-CAM uses gradients of the target class with respect to feature maps
    to weight and combine them into a class-specific activation map.

    Advantages:
        - Class-specific explanations
        - Strong interpretability
        - Widely used and validated

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017

    Examples:
        >>> model = init_model('yolox_s.py', 'yolox_s.pth')
        >>> cam = GradCAM(model, target_layer='neck')
        >>> img = torch.rand(1, 3, 640, 640)
        >>> cam_map = cam(img, target_class=0)  # for class 0
        >>> print(cam_map.shape)  # (H, W)
    """

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None):
        """Compute Grad-CAM.

        Args:
            x (torch.Tensor): Input tensor of shape (1, 3, H, W)
            target_class (int, optional): Target class index.
                If None, uses the class with highest score.

        Returns:
            np.ndarray: CAM heatmap of shape (H, W)
        """
        # Register forward and backward hooks
        forward_handle, backward_handle = self.register_backward_hooks()

        # Forward pass
        self.model.eval()
        output = self.model(x)

        # Get target class
        if target_class is None:
            # Use the class with highest score
            if hasattr(output[0], 'pred_instances'):
                scores = output[0].pred_instances.scores
                if len(scores) > 0:
                    target_class = scores.argmax().item()
                else:
                    # No detections, use first channel
                    target_class = 0
            else:
                # For non-detection outputs
                target_class = 0

        # Get target score for backward pass
        if hasattr(output[0], 'pred_instances'):
            scores = output[0].pred_instances.scores
            if len(scores) > 0 and target_class < len(scores):
                target_score = scores[target_class]
            else:
                # Fallback: use first detection score
                target_score = scores[0] if len(scores) > 0 else torch.tensor(1.0)
        else:
            # For raw model outputs
            target_score = output[0].sum()  # Use sum as fallback

        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # Get gradients and features
        grads = self.gradients  # (1, C, H, W) or similar
        feats = self.features   # (1, C, H, W) or similar

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Compute CAM
        cam_map = self._compute_grad_cam(feats, grads)

        return cam_map

    def register_backward_hooks(self):
        """Register forward and backward hooks.

        Returns:
            tuple: (forward_handle, backward_handle)
        """
        target_module = self._get_target_layer()

        forward_handle = target_module.register_forward_hook(self._forward_hook)
        backward_handle = target_module.register_full_backward_hook(self._backward_hook)

        return forward_handle, backward_handle

    def _backward_hook(self, module, grad_in, grad_out):
        """Backward hook to save gradients.

        Args:
            module: The layer module
            grad_in: Gradients from the input
            grad_out: Gradients to the output
        """
        self.gradients = grad_out[0]

    def _compute_grad_cam(self, features: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM from features and gradients.

        Args:
            features (torch.Tensor): Feature maps of shape (1, C, H, W)
            gradients (torch.Tensor): Gradients of shape (1, C, H, W)

        Returns:
            np.ndarray: CAM heatmap normalized to [0, 1]
        """
        # Global average pooling on gradients to get weights
        # Shape: (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        # Shape: (1, 1, H, W)
        cam = (weights * features).sum(dim=1, keepdim=True)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


# Utility functions

def apply_colormap(heatmap: np.ndarray, colormap: str = 'jet') -> np.ndarray:
    """Apply color mapping to heatmap.

    Args:
        heatmap (np.ndarray): Normalized heatmap of shape (H, W) with values in [0, 1]
        colormap (str): Matplotlib colormap name (e.g., 'jet', 'hot', 'viridis')

    Returns:
        np.ndarray: RGB image of shape (H, W, 3) with values in [0, 255]

    Examples:
        >>> heatmap = np.random.rand(100, 100)
        >>> colored = apply_colormap(heatmap, 'jet')
        >>> print(colored.shape)  # (100, 100, 3)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for colormap application. "
                         "Install it with: pip install matplotlib")

    # Convert to 0-255 range
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(heatmap_uint8)[:, :, :3]  # (H, W, 3) RGBA -> RGB

    # Convert to 0-255
    colored = (colored * 255).astype(np.uint8)

    return colored


def overlay_cam(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay CAM heatmap on original image.

    Args:
        image (np.ndarray): Original RGB image of shape (H, W, 3)
        heatmap (np.ndarray): RGB heatmap of shape (H, W, 3)
        alpha (float): Transparency for heatmap overlay (0-1)

    Returns:
        np.ndarray: Overlayed image of shape (H, W, 3)

    Examples:
        >>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> heat = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> overlay = overlay_cam(img, heat, alpha=0.5)
        >>> print(overlay.shape)  # (100, 100, 3)
    """
    # Ensure float32 for blending
    image = image.astype(np.float32)
    heatmap = heatmap.astype(np.float32)

    # Blend
    overlayed = (1 - alpha) * image + alpha * heatmap

    return overlayed.astype(np.uint8)


def resize_cam(cam_map: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize CAM heatmap to target size.

    Args:
        cam_map (np.ndarray): CAM heatmap of shape (H, W)
        target_size (tuple): Target size (H_new, W_new)

    Returns:
        np.ndarray: Resized heatmap of shape (H_new, W_new)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for resizing. "
                         "Install it with: pip install opencv-python")

    return cv2.resize(cam_map, (target_size[1], target_size[0]),
                     interpolation=cv2.INTER_LINEAR)


def extract_bbox_cam(cam_map: np.ndarray, bbox: np.ndarray,
                     img_shape: tuple) -> np.ndarray:
    """Extract CAM region corresponding to a bounding box.

    Args:
        cam_map (np.ndarray): Full CAM heatmap of shape (H_cam, W_cam)
        bbox (np.ndarray): Bounding box [x1, y1, x2, y2] in original image coordinates
        img_shape (tuple): Original image shape (H_img, W_img)

    Returns:
        np.ndarray: Extracted CAM region

    Examples:
        >>> cam = np.random.rand(160, 160)  # CAM at 1/4 resolution
        >>> bbox = np.array([100, 100, 200, 200])  # bbox in original 640x640 image
        >>> region = extract_bbox_cam(cam, bbox, (640, 640))
        >>> print(region.shape)  # (~25, 25)
    """
    x1, y1, x2, y2 = bbox
    h_cam, w_cam = cam_map.shape
    h_img, w_img = img_shape

    # Scale bbox coordinates to CAM size
    scale_x = w_cam / w_img
    scale_y = h_cam / h_img

    x1_cam = int(x1 * scale_x)
    y1_cam = int(y1 * scale_y)
    x2_cam = int(x2 * scale_x)
    y2_cam = int(y2 * scale_y)

    # Extract region
    region = cam_map[y1_cam:y2_cam, x1_cam:x2_cam]

    return region
