# Copyright (c) OpenMMLab. All rights reserved.
"""CAM Visualizer for object detection models.

This module provides visualization utilities for Class Activation Mapping (CAM)
heatmaps, allowing users to overlay activation maps on detection results.
"""

import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
from mmengine.visualization import Visualizer

from mmdet.registry import VISUALIZERS
from ..utils.cam_utils import apply_colormap, overlay_cam, resize_cam


@VISUALIZERS.register_module()
class CAMVisualizer(Visualizer):
    """CAM Visualizer for object detection.

    This visualizer extends the base Visualizer to support CAM heatmap
    visualization on top of detection results.

    Args:
        name (str): Name of the visualizer. Defaults to 'cam_visualizer'
        alpha (float): Transparency for CAM overlay (0-1). Defaults to 0.5
        colormap (str): Matplotlib colormap name. Defaults to 'jet'
        show_colorbar (bool): Whether to show colorbar. Defaults to False
        **kwargs: Other arguments passed to Visualizer base class

    Examples:
        >>> from mmdet.visualization import CAMVisualizer
        >>> import numpy as np
        >>> visualizer = CAMVisualizer(alpha=0.5, colormap='jet')
        >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> cam_map = np.random.rand(120, 160)  # CAM at 1/4 resolution
        >>> result = visualizer.draw_cam(image, cam_map)
        >>> visualizer.save('cam_result.jpg')
    """

    def __init__(self,
                 name: str = 'cam_visualizer',
                 alpha: float = 0.5,
                 colormap: str = 'jet',
                 show_colorbar: bool = False,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.colormap = colormap
        self.show_colorbar = show_colorbar

    def draw_cam(self,
                 image: np.ndarray,
                 cam_map: np.ndarray,
                 bboxes: Optional[np.ndarray] = None,
                 labels: Optional[np.ndarray] = None,
                 scores: Optional[np.ndarray] = None,
                 classes: Optional[List[str]] = None) -> np.ndarray:
        """Draw CAM heatmap on image with optional detections.

        Args:
            image (np.ndarray): Original RGB image of shape (H, W, 3)
            cam_map (np.ndarray): CAM heatmap of shape (H_cam, W_cam)
                Values should be in range [0, 1]
            bboxes (np.ndarray, optional): Bounding boxes of shape (N, 4)
                in format [x1, y1, x2, y2]
            labels (np.ndarray, optional): Class labels of shape (N,)
            scores (np.ndarray, optional): Confidence scores of shape (N,)
            classes (list, optional): Class name list for label mapping

        Returns:
            np.ndarray: Image with CAM overlay and detections, shape (H, W, 3)

        Examples:
            >>> visualizer = CAMVisualizer()
            >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> cam = np.random.rand(480, 640)  # Same size as image
            >>> bboxes = np.array([[100, 100, 200, 200]])
            >>> labels = np.array([0])
            >>> result = visualizer.draw_cam(img, cam, bboxes, labels)
        """
        # Apply colormap
        heatmap = apply_colormap(cam_map, self.colormap)

        # Resize heatmap to match image size
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        # Overlay CAM on image
        overlayed = overlay_cam(image, heatmap, self.alpha)

        # Set image for drawing
        self.set_image(overlayed)

        # Draw bounding boxes if provided
        if bboxes is not None and len(bboxes) > 0:
            self.draw_instances(cam_image=overlayed,
                               bboxes=bboxes,
                               labels=labels,
                               scores=scores,
                               classes=classes)

        return self.get_image()

    def draw_instances(self,
                       cam_image: np.ndarray,
                       bboxes: np.ndarray,
                       labels: Optional[np.ndarray] = None,
                       scores: Optional[np.ndarray] = None,
                       classes: Optional[List[str]] = None):
        """Draw detection instances on CAM overlay.

        Args:
            cam_image (np.ndarray): Image with CAM overlay
            bboxes (np.ndarray): Bounding boxes of shape (N, 4)
            labels (np.ndarray, optional): Class labels
            scores (np.ndarray, optional): Confidence scores
            classes (list, optional): Class name list
        """
        # Use parent's draw_bboxes method
        self.draw_bboxes(bboxes, edge_colors='green', line_widths=2)

        # Add labels if provided
        if labels is not None and classes is not None:
            label_names = [classes[l] for l in labels]
            if scores is not None:
                label_texts = [f'{n} {s:.2f}' for n, s in zip(label_names, scores)]
            else:
                label_texts = label_names

            self.draw_texts(label_texts, bboxes[:, :2], colors='white')

    def draw_cam_per_bbox(self,
                          image: np.ndarray,
                          cam_map: np.ndarray,
                          bboxes: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          scores: Optional[np.ndarray] = None,
                          classes: Optional[List[str]] = None) -> np.ndarray:
        """Draw CAM for each bounding box separately.

        This method extracts the CAM region corresponding to each bbox
        and creates a visualization showing the activation for each detection.

        Args:
            image (np.ndarray): Original RGB image
            cam_map (np.ndarray): Full CAM heatmap
            bboxes (np.ndarray): Bounding boxes of shape (N, 4)
            labels (np.ndarray, optional): Class labels
            scores (np.ndarray, optional): Confidence scores
            classes (list, optional): Class name list

        Returns:
            np.ndarray: Image with bbox-specific CAM visualizations

        Examples:
            >>> visualizer = CAMVisualizer()
            >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> cam = np.random.rand(120, 160)  # CAM at 1/4 resolution
            >>> bboxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
            >>> result = visualizer.draw_cam_per_bbox(img, cam, bboxes)
        """
        from ..utils.cam_utils import extract_bbox_cam

        result_img = image.copy()
        img_h, img_w = image.shape[:2]

        for i, bbox in enumerate(bboxes):
            # Extract CAM region for this bbox
            bbox_cam = extract_bbox_cam(cam_map, bbox, (img_h, img_w))

            # Get bbox region in image
            x1, y1, x2, y2 = bbox.astype(int)
            bbox_img = image[y1:y2, x1:x2]

            # Resize CAM to bbox size
            bbox_h, bbox_w = y2 - y1, x2 - x1
            if bbox_cam.size > 0:
                bbox_cam_resized = cv2.resize(bbox_cam, (bbox_w, bbox_h))
            else:
                bbox_cam_resized = bbox_cam

            # Apply colormap
            bbox_heatmap = apply_colormap(bbox_cam_resized, self.colormap)

            # Overlay on bbox region
            bbox_overlay = overlay_cam(bbox_img, bbox_heatmap, self.alpha)

            # Place back on result image
            result_img[y1:y2, x1:x2] = bbox_overlay

            # Draw bbox border
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            if labels is not None and classes is not None:
                label = classes[labels[i]]
                text = f'{label}'
                if scores is not None:
                    text += f' {scores[i]:.2f}'
                cv2.putText(result_img, text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return result_img

    def add_datasample_with_cam(self,
                                name: str,
                                image: np.ndarray,
                                data_sample,
                                cam_map: np.ndarray,
                                draw_gt: bool = False,
                                draw_pred: bool = True,
                                show: bool = False,
                                wait_time: float = 0,
                                out_file: Optional[str] = None,
                                score_thr: float = 0.3):
        """Add data sample with CAM visualization.

        This is a convenience method that integrates with MMDetection's
        data sample structure.

        Args:
            name (str): Image name
            image (np.ndarray): Original image
            data_sample: Data sample containing predictions
            cam_map (np.ndarray): CAM heatmap
            draw_gt (bool): Whether to draw ground truth. Defaults to False
            draw_pred (bool): Whether to draw predictions. Defaults to True
            show (bool): Whether to display the image. Defaults to False
            wait_time (float): Display wait time in seconds. Defaults to 0
            out_file (str, optional): Output file path
            score_thr (float): Score threshold for filtering detections

        Examples:
            >>> from mmdet.structures import DetDataSample
            >>> visualizer = CAMVisualizer()
            >>> data_sample = DetDataSample()
            >>> # Set up data_sample with predictions
            >>> cam = np.random.rand(160, 160)
            >>> visualizer.add_datasample_with_cam('image', img, data_sample, cam)
        """
        # Get predictions
        pred_instances = data_sample.pred_instances if hasattr(data_sample, 'pred_instances') else None

        # Filter by score
        bboxes = None
        labels = None
        scores = None

        if pred_instances is not None and draw_pred:
            scores_np = pred_instances.scores.cpu().numpy()
            keep = scores_np >= score_thr

            if keep.sum() > 0:
                bboxes = pred_instances.bboxes[keep].cpu().numpy()
                labels = pred_instances.labels[keep].cpu().numpy()
                scores = scores_np[keep]

        # Draw CAM with detections
        result_img = self.draw_cam(image, cam_map, bboxes, labels, scores)

        # Set image
        self.set_image(result_img)

        # Save or show
        if out_file is not None:
            self.save(out_file)

        if show:
            self.show(wait_time=wait_time)

    def get_cam_comparison(self,
                          image: np.ndarray,
                          cam_map: np.ndarray,
                          bboxes: Optional[np.ndarray] = None) -> np.ndarray:
        """Create a side-by-side comparison view.

        Creates a 3-panel image showing:
        1. Original image with bboxes
        2. CAM heatmap only
        3. CAM overlay on image with bboxes

        Args:
            image (np.ndarray): Original image
            cam_map (np.ndarray): CAM heatmap
            bboxes (np.ndarray, optional): Bounding boxes

        Returns:
            np.ndarray: Comparison image of shape (H, 3*W, 3)

        Examples:
            >>> visualizer = CAMVisualizer()
            >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> cam = np.random.rand(480, 640)
            >>> comparison = visualizer.get_cam_comparison(img, cam)
        """
        h, w = image.shape[:2]

        # Panel 1: Original image with bboxes
        img1 = image.copy()
        if bboxes is not None:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Panel 2: CAM heatmap only
        heatmap = apply_colormap(cam_map, self.colormap)
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))
        img2 = heatmap

        # Panel 3: CAM overlay
        img3 = self.draw_cam(image, cam_map, bboxes)

        # Concatenate horizontally
        comparison = np.hstack([img1, img2, img3])

        # Add labels
        cv2.putText(comparison, 'Original', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'CAM Heatmap', (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'CAM Overlay', (2*w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return comparison
