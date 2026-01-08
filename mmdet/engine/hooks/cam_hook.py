# Copyright (c) OpenMMLab. All rights reserved.
"""Hook for CAM visualization during validation/testing."""

import os
import warnings
from typing import Optional

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmdet.utils import cam_utils
from mmdet.visualization import CAMVisualizer


@HOOKS.register_module()
class CAMVisualizationHook(Hook):
    """Hook to generate Class Activation Mapping (CAM) visualizations during validation/testing.

    This hook automatically computes and saves CAM visualizations for each batch
    during testing, showing which regions of the image the model focuses on
    when making predictions.

    Args:
        enabled (bool): Whether to enable CAM visualization. Defaults to True
        method (str): CAM method to use. Options: 'eigen', 'grad'. Defaults to 'eigen'
        target_layer (str): Target layer for CAM computation.
            Common options: 'backbone', 'neck', 'head'. Defaults to 'neck'
        alpha (float): Transparency for CAM overlay (0-1). Defaults to 0.5
        colormap (str): Matplotlib colormap name. Defaults to 'jet'
        score_thr (float): Score threshold for filtering detections. Defaults to 0.3
        save_dir (str, optional): Directory to save visualizations.
            If None, uses runner's default visualization directory. Defaults to None
        save_heatmap (bool): Whether to save separate heatmaps without overlay.
            Defaults to False
        overlay_only (bool): Whether to only save overlayed images. Defaults to True
        per_bbox (bool): Whether to generate CAM for each bbox separately.
            Defaults to False
        comparison (bool): Whether to generate side-by-side comparison view.
            Defaults to False
        interval (int): Process every N batches. Defaults to 1 (process all)

    Examples:
        >>> # In config file
        >>> custom_hooks = [
        ...     dict(
        ...         type='CAMVisualizationHook',
        ...         method='eigen',
        ...         target_layer='neck',
        ...         alpha=0.5,
        ...         score_thr=0.3
        ...     )
        ... ]

        >>> # Or programmatically
        >>> hook = CAMVisualizationHook(
        ...     method='eigen',
        ...     target_layer='neck',
        ...     save_dir='cam_results'
        ... )
        >>> runner.register_hook(hook, priority='LOW')
    """

    def __init__(self,
                 enabled: bool = True,
                 method: str = 'eigen',
                 target_layer: str = 'neck',
                 alpha: float = 0.5,
                 colormap: str = 'jet',
                 score_thr: float = 0.3,
                 save_dir: Optional[str] = None,
                 save_heatmap: bool = False,
                 overlay_only: bool = True,
                 per_bbox: bool = False,
                 comparison: bool = False,
                 interval: int = 1):
        self.enabled = enabled
        self.method = method
        self.target_layer = target_layer
        self.alpha = alpha
        self.colormap = colormap
        self.score_thr = score_thr
        self.save_dir = save_dir
        self.save_heatmap = save_heatmap
        self.overlay_only = overlay_only
        self.per_bbox = per_bbox
        self.comparison = comparison
        self.interval = interval

        # Validate method
        if self.method not in ['eigen', 'grad']:
            raise ValueError(f"Unsupported method: {self.method}. "
                           f"Choose from 'eigen' or 'grad'")

        # Initialize CAM instance
        self.cam = None
        self.visualizer = None

    def before_val(self, runner: Runner) -> None:
        """Initialize CAM and visualizer before validation.

        Args:
            runner (Runner): The runner
        """
        if not self.enabled:
            return

        # Get save directory
        if self.save_dir is None:
            self.save_dir = os.path.join(
                runner._log_dir if hasattr(runner, '_log_dir') else 'work_dirs',
                'cam_vis'
            )

        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize CAM
        if self.method == 'eigen':
            self.cam = cam_utils.EigenCAM(runner.model, self.target_layer)
        else:
            self.cam = cam_utils.GradCAM(runner.model, self.target_layer)

        # Initialize visualizer
        self.visualizer = CAMVisualizer(
            alpha=self.alpha,
            colormap=self.colormap
        )

        runner.logger.info(f"CAM visualization enabled. Method: {self.method}, "
                          f"Layer: {self.target_layer}, Save dir: {self.save_dir}")

    def after_test_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch: dict,
                       outputs: list) -> None:
        """Generate CAM visualization after each test iteration.

        Args:
            runner (Runner): The runner
            batch_idx (int): Batch index
            data_batch (dict): Data batch containing inputs and data_samples
            outputs (list): Model outputs
        """
        if not self.enabled:
            return

        # Process every N batches
        if batch_idx % self.interval != 0:
            return

        # Get inputs and data samples
        inputs = data_batch.get('inputs', data_batch.get('img'))
        data_samples = data_batch.get('data_samples', [])

        if inputs is None or len(data_samples) == 0:
            return

        # Handle batch dimension
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        # Process each sample in the batch
        for i, (img_tensor, data_sample) in enumerate(zip(inputs, data_samples)):
            try:
                self._process_single_sample(
                    runner, img_tensor, data_sample, batch_idx, i
                )
            except Exception as e:
                runner.logger.warning(
                    f"Failed to generate CAM for batch {batch_idx}, sample {i}: {e}"
                )

    def _process_single_sample(self,
                               runner: Runner,
                               img_tensor: torch.Tensor,
                               data_sample,
                               batch_idx: int,
                               sample_idx: int) -> None:
        """Process a single sample.

        Args:
            runner (Runner): The runner
            img_tensor (torch.Tensor): Input image tensor
            data_sample: Data sample
            batch_idx (int): Batch index
            sample_idx (int): Sample index within batch
        """
        # Get predictions
        if not hasattr(data_sample, 'pred_instances'):
            return

        pred_instances = data_sample.pred_instances

        # Filter by score threshold
        scores = pred_instances.scores.cpu().numpy()
        keep = scores >= self.score_thr

        if keep.sum() == 0:
            return

        # Get image (convert to numpy)
        img_np = self._tensor_to_image(img_tensor)

        # Compute CAM
        with torch.set_grad_enabled(self.method == 'grad'):
            cam_map = self.cam(img_tensor.unsqueeze(0))

        # Resize CAM to match image size if needed
        if cam_map.shape != img_np.shape[:2]:
            cam_map = cam_utils.resize_cam(cam_map, img_np.shape[:2])

        # Get metadata
        img_name = self._get_image_name(data_sample, batch_idx, sample_idx)
        classes = getattr(runner.model, 'CLASSES', None)

        # Generate visualizations
        bboxes = pred_instances.bboxes[keep].cpu().numpy()
        labels = pred_instances.labels[keep].cpu().numpy()
        scores_filtered = scores[keep]

        if self.comparison:
            # Generate comparison view
            result_img = self.visualizer.get_cam_comparison(
                img_np, cam_map, bboxes
            )
            save_path = os.path.join(self.save_dir, f'{img_name}_comparison.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        elif self.per_bbox:
            # Generate per-bbox visualization
            result_img = self.visualizer.draw_cam_per_bbox(
                img_np, cam_map, bboxes, labels, scores_filtered, classes
            )
            save_path = os.path.join(self.save_dir, f'{img_name}_per_bbox.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        else:
            # Standard overlay visualization
            result_img = self.visualizer.draw_cam(
                img_np, cam_map, bboxes, labels, scores_filtered, classes
            )

            # Save overlay
            save_path = os.path.join(self.save_dir, f'{img_name}_cam.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

            # Save separate heatmap if requested
            if self.save_heatmap and not self.overlay_only:
                heatmap = cam_utils.apply_colormap(cam_map, self.colormap)
                if heatmap.shape[:2] != img_np.shape[:2]:
                    try:
                        import cv2
                        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                    except ImportError:
                        pass
                heatmap_path = os.path.join(self.save_dir, f'{img_name}_heatmap.jpg')
                cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    def _tensor_to_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Convert image tensor to numpy array.

        Args:
            img_tensor (torch.Tensor): Image tensor of shape (C, H, W) or (1, C, H, W)

        Returns:
            np.ndarray: Image array of shape (H, W, 3) in RGB format
        """
        # Remove batch dimension if present
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)

        # Convert to numpy
        img_np = img_tensor.cpu().numpy()

        # Handle different data ranges
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # Transpose from (C, H, W) to (H, W, C)
        img_np = np.transpose(img_np, (1, 2, 0))

        # Ensure RGB format (if grayscale, convert to RGB)
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)

        return img_np

    def _get_image_name(self, data_sample, batch_idx: int, sample_idx: int) -> str:
        """Get image name from data sample.

        Args:
            data_sample: Data sample
            batch_idx (int): Batch index
            sample_idx (int): Sample index

        Returns:
            str: Image name without extension
        """
        # Try to get image path from data sample
        img_path = getattr(data_sample, 'img_path', None)
        if img_path is not None:
            # Extract filename without extension
            img_name = os.path.basename(img_path).rsplit('.', 1)[0]
        else:
            # Use batch and sample index
            img_name = f'batch{batch_idx:04d}_sample{sample_idx:02d}'

        return img_name

    def after_val(self, runner: Runner) -> None:
        """Clean up after validation.

        Args:
            runner (Runner): The runner
        """
        if not self.enabled:
            return

        runner.logger.info(f"CAM visualizations saved to: {self.save_dir}")

    def after_test(self, runner: Runner) -> None:
        """Clean up after testing.

        Args:
            runner (Runner): The runner
        """
        if not self.enabled:
            return

        runner.logger.info(f"CAM visualizations saved to: {self.save_dir}")


# Import cv2 at module level for convenience
try:
    import cv2
except ImportError:
    warnings.warn("opencv-python is required for CAM visualization. "
                  "Install it with: pip install opencv-python")
