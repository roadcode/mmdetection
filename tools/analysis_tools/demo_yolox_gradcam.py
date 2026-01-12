#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Demo script for YOLOX Grad-CAM visualization.

This script demonstrates how to generate Grad-CAM visualizations for
YOLOX object detection models. It supports various CAM methods and
provides comprehensive visualization options.

Example:
    # Basic usage with default settings
    python demo_yolox_gradcam.py \\
        configs/yolox/yolox_s_8xb8-300e_coco.py \\
        https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth \\
        demo/demo.jpg \\
        --out-dir outputs/gradcam

    # Use GradCAM++ method with custom target layer
    python demo_yolox_gradcam.py \\
        configs/yolox/yolox_s_8xb8-300e_coco.py \\
        checkpoint.pth \\
        demo/demo.jpg \\
        --method gradcam++ \\
        --target-layers neck.out_convs.2 \\
        --score-thr 0.5
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

# Add mmdetection to path
mmdet_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(mmdet_path))

from mmdet.apis import init_detector, inference_detector
from mmdet.registry import MODELS
from mmdet.utils import get_test_pipeline_cfg

# Check and import grad-cam
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("\n" + "="*60)
    print("ERROR: pytorch-grad-cam is not installed!")
    print("Please install it with: pip install grad-cam")
    print("="*60 + "\n")

# Import our wrapper utilities
from yolox_gradcam_wrapper import (
    DetectionTarget, YOLOXGradCAMWrapper, get_default_target_layers,
    show_cam_on_detection, renormalize_cam_in_bounding_boxes,
    get_target_layer_from_name, GRAD_CAM_AVAILABLE as WRAPPER_GRAD_CAM
)


CAM_METHODS = {
    'gradcam': GradCAM if GRAD_CAM_AVAILABLE else None,
    'gradcam++': GradCAMPlusPlus if GRAD_CAM_AVAILABLE else None,
    'xgradcam': XGradCAM if GRAD_CAM_AVAILABLE else None,
    'eigencam': EigenCAM if GRAD_CAM_AVAILABLE else None,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM visualizations for YOLOX detections')
    
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path or URL')
    parser.add_argument('img', help='Image file path')
    parser.add_argument(
        '--out-dir',
        default='outputs/gradcam',
        help='Directory to save output visualizations')
    parser.add_argument(
        '--method',
        default='gradcam',
        choices=['gradcam', 'gradcam++', 'xgradcam', 'eigencam'],
        help='CAM method to use')
    parser.add_argument(
        '--target-layers',
        nargs='+',
        default=None,
        help='Target layer names for CAM (e.g., neck.out_convs.2). '
             'If not specified, uses default layers.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Score threshold for detections')
    parser.add_argument(
        '--topk',
        type=int,
        default=None,
        help='Only show CAM for top-k detections. If None, show all.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument(
        '--renormalize',
        action='store_true',
        help='Renormalize CAM within bounding boxes for better visualization')
    parser.add_argument(
        '--save-all',
        action='store_true',
        help='Save all intermediate results (original, detection, CAM)')
    
    args = parser.parse_args()
    return args


def get_target_layers_from_args(model, layer_names):
    """Get target layers from model using layer names.
    
    Args:
        model: The detection model.
        layer_names (list): List of layer names.
        
    Returns:
        list: List of target layer modules.
    """
    if layer_names is None:
        # Use default layers
        layers = get_default_target_layers(model)
        if not layers:
            print("Using fallback: neck's last output conv")
            try:
                layers = [model.neck.out_convs[-1]]
            except:
                raise ValueError(
                    "Could not find default target layers. "
                    "Please specify with --target-layers")
        return layers
    
    layers = []
    for name in layer_names:
        layer = get_target_layer_from_name(model, name)
        if layer is None:
            raise ValueError(f"Could not find layer: {name}")
        layers.append(layer)
    
    return layers


def run_gradcam_on_image(model, image_path, cam_algorithm, target_layers,
                         score_thr=0.3, topk=None, renormalize=False,
                         device='cuda:0'):
    """Run Grad-CAM on a single image.
    
    Args:
        model: Detection model.
        image_path (str): Path to input image.
        cam_algorithm: CAM algorithm class (e.g., GradCAM).
        target_layers (list): List of target layers.
        score_thr (float): Score threshold for detections.
        topk (int, optional): Only process top-k detections.
        renormalize (bool): Whether to renormalize CAM in bounding boxes.
        device (str): Device for inference.
        
    Returns:
        tuple: (cam_image, detections, original_image)
    """
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0
    
    # Run detection
    with torch.no_grad():
        result = inference_detector(model, image_path)
    
    # Extract detection results
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    
    # Filter by score threshold
    keep = scores >= score_thr
    bboxes = bboxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    if len(bboxes) == 0:
        print(f"No detections found with score >= {score_thr}")
        return image_rgb, None, image_rgb
    
    # Keep only top-k detections if specified
    if topk is not None and len(bboxes) > topk:
        top_indices = np.argsort(scores)[-topk:]
        bboxes = bboxes[top_indices]
        labels = labels[top_indices]
        scores = scores[top_indices]
    
    print(f"Generating CAM for {len(bboxes)} detections...")
    
    # Prepare input tensor
    # We need to use the model's data pipeline
    from mmdet.apis import init_detector
    from mmengine.dataset import Compose
    from mmdet.structures import DetDataSample
    
    # Build data pipeline
    test_pipeline = get_test_pipeline_cfg(model.cfg)
    test_pipeline = Compose(test_pipeline)
    
    # Prepare data
    data = dict(img=image_path, img_id=0)
    data = test_pipeline(data)
    
    # Get input tensor
    input_tensor = data['inputs'].unsqueeze(0).to(device)
    
    # Setup Grad-CAM
    cam = cam_algorithm(
        model=model,
        target_layers=target_layers,
        use_cuda=device.startswith('cuda')
    )
    
    # Create detection target
    targets = [DetectionTarget(bboxes, labels, scores)]
    
    # Generate CAM
    try:
        # Run CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Get first image in batch
        
        # Resize CAM to match original image size
        grayscale_cam = cv2.resize(
            grayscale_cam,
            (image_rgb.shape[1], image_rgb.shape[0])
        )
        
        # Optionally renormalize within bounding boxes
        if renormalize:
            grayscale_cam = renormalize_cam_in_bounding_boxes(
                bboxes, image_float, grayscale_cam
            )
        
        # Get class names
        class_names = model.dataset_meta.get('classes', None)
        
        # Create visualization
        cam_image = show_cam_on_detection(
            image_rgb,
            grayscale_cam,
            bboxes,
            labels,
            scores,
            class_names=class_names,
            use_rgb=True
        )
        
        return cam_image, (bboxes, labels, scores), image_rgb
        
    except Exception as e:
        print(f"Error generating CAM: {e}")
        import traceback
        traceback.print_exc()
        return image_rgb, (bboxes, labels, scores), image_rgb


def draw_detections(image, bboxes, labels, scores, class_names=None):
    """Draw detection boxes on image.
    
    Args:
        image (np.ndarray): Input image.
        bboxes (np.ndarray): Bounding boxes.
        labels (np.ndarray): Class labels.
        scores (np.ndarray): Confidence scores.
        class_names (list, optional): Class names.
        
    Returns:
        np.ndarray: Image with drawn detections.
    """
    img_with_boxes = image.copy()
    
    for bbox, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Draw box
        color = (0, 255, 0)  # Green
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f'{class_names[label] if class_names else label}: {score:.2f}'
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw background
        cv2.rectangle(img_with_boxes,
                     (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1),
                     color, -1)
        
        # Draw text
        cv2.putText(img_with_boxes, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img_with_boxes


def main():
    """Main function."""
    args = parse_args()
    
    if not GRAD_CAM_AVAILABLE:
        print("Exiting: pytorch-grad-cam is required but not installed.")
        return
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Initialize model
    print(f"Loading model from {args.checkpoint}...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    
    # Get target layers
    print("Setting up target layers...")
    target_layers = get_target_layers_from_args(model, args.target_layers)
    print(f"Using target layers: {[str(layer) for layer in target_layers]}")
    
    # Get CAM method
    cam_method = CAM_METHODS[args.method]
    print(f"Using CAM method: {args.method}")
    
    # Process image
    print(f"\nProcessing image: {args.img}")
    cam_image, detections, original_image = run_gradcam_on_image(
        model=model,
        image_path=args.img,
        cam_algorithm=cam_method,
        target_layers=target_layers,
        score_thr=args.score_thr,
        topk=args.topk,
        renormalize=args.renormalize,
        device=args.device
    )
    
    # Save results
    img_name = Path(args.img).stem
    
    # Always save CAM visualization
    cam_output_path = os.path.join(args.out_dir, f'{img_name}_gradcam.jpg')
    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(cam_output_path, cam_image_bgr)
    print(f"Saved Grad-CAM visualization to: {cam_output_path}")
    
    # Optionally save all intermediate results
    if args.save_all and detections is not None:
        # Save original image
        orig_output_path = os.path.join(args.out_dir, f'{img_name}_original.jpg')
        cv2.imwrite(orig_output_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        
        # Save detection result
        bboxes, labels, scores = detections
        class_names = model.dataset_meta.get('classes', None)
        det_image = draw_detections(
            original_image, bboxes, labels, scores, class_names
        )
        det_output_path = os.path.join(args.out_dir, f'{img_name}_detection.jpg')
        cv2.imwrite(det_output_path, cv2.cvtColor(det_image, cv2.COLOR_RGB2BGR))
        print(f"Saved detection result to: {det_output_path}")
        print(f"Saved original image to: {orig_output_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
