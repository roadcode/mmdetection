# Copyright (c) OpenMMLab. All rights reserved.
"""Standalone CAM visualization tool for YOLOX models.

This script provides a command-line interface for generating Class Activation
Mapping (CAM) visualizations on trained YOLOX models.
"""

import argparse
import os
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS
from mmdet.utils import cam_utils
from mmdet.visualization import CAMVisualizer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate CAM visualizations for YOLOX models'
    )
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--img', help='Input image path', required=True)
    parser.add_argument('--out-dir', default='cam_results',
                       help='Directory to save results')
    parser.add_argument('--method', default='eigen', choices=['eigen', 'grad'],
                       help='CAM method: eigen (fast) or grad (class-specific)')
    parser.add_argument('--target-layer', default='neck',
                       help='Target layer for CAM (backbone/neck)')
    parser.add_argument('--score-thr', type=float, default=0.3,
                       help='Score threshold for filtering detections')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Transparency for CAM overlay (0-1)')
    parser.add_argument('--colormap', default='jet',
                       help='Matplotlib colormap name')
    parser.add_argument('--device', default='cuda:0',
                       help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--per-bbox', action='store_true',
                       help='Generate CAM for each bbox separately')
    parser.add_argument('--comparison', action='store_true',
                       help='Generate side-by-side comparison view')
    parser.add_argument('--show', action='store_true',
                       help='Display the result')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load config
    print(f"Loading config from: {args.config}")
    cfg = Config.fromfile(args.config)

    # Build model
    print(f"Building model: {cfg.model.type}")
    model = MODELS.build(cfg.model)
    model.to(args.device)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # Get class names
    CLASSES = getattr(cfg, 'classes', getattr(cfg, 'CLASSES',
                   getattr(cfg.get('model', {}), 'CLASSES', None)))
    if CLASSES is None:
        # Try to get from metainfo in checkpoint
        if 'dataset_meta' in checkpoint.get('message', {}):
            CLASSES = checkpoint['message']['dataset_meta'].get('classes')
        if CLASSES is None:
            print("Warning: Class names not found. Using generic names.")
            CLASSES = [f'class_{i}' for i in range(80)]
    print(f"Number of classes: {len(CLASSES)}")

    # Load image
    print(f"Loading image from: {args.img}")
    img = mmcv.imread(args.img)
    img_rgb = mmcv.bgr2rgb(img)

    # Prepare input tensor
    from mmdet.registry import MODELS
    from mmengine.registry import DATA_PREPROCESSORS

    # Get data preprocessor
    if 'data_preprocessor' in cfg.model:
        preprocessor = DATA_PREPROCESSORS.build(cfg.model.data_preprocessor)
    else:
        from mmdet.models.data_preprocessors import DetDataPreprocessor
        preprocessor = DetDataPreprocessor(mean=None, std=None, bgr_to_rgb=True,
                                           pad_size_divisor=32)

    # Preprocess image
    data = {'inputs': img_rgb, 'batch_input_shape': (img_rgb.shape[0], img_rgb.shape[1])}
    preprocessed = preprocessor([data], training=False)
    img_tensor = preprocessed['inputs'].to(args.device)

    # Initialize CAM
    print(f"Initializing {args.method.upper()} CAM...")
    if args.method == 'eigen':
        cam = cam_utils.EigenCAM(model, args.target_layer)
    else:
        cam = cam_utils.GradCAM(model, args.target_layer)

    # Run inference to get predictions
    print("Running inference...")
    with torch.no_grad():
        output = model(img_tensor, mode='predict')

    # Compute CAM
    print("Computing CAM...")
    with torch.set_grad_enabled(args.method == 'grad'):
        cam_map = cam(img_tensor)

    # Resize CAM to match image size
    if cam_map.shape != img_rgb.shape[:2]:
        cam_map = cam_utils.resize_cam(cam_map, img_rgb.shape[:2])

    # Get predictions
    pred_instances = output[0].pred_instances
    scores = pred_instances.scores.cpu().numpy()
    keep = scores >= args.score_thr

    if keep.sum() > 0:
        bboxes = pred_instances.bboxes[keep].cpu().numpy()
        labels = pred_instances.labels[keep].cpu().numpy()
        scores_filtered = scores[keep]
        print(f"Found {keep.sum()} detections above threshold {args.score_thr}")
    else:
        print(f"No detections found above threshold {args.score_thr}")
        bboxes = None
        labels = None
        scores_filtered = None

    # Initialize visualizer
    visualizer = CAMVisualizer(alpha=args.alpha, colormap=args.colormap)

    # Generate visualization
    print("Generating visualization...")
    img_name = Path(args.img).stem

    if args.comparison:
        # Comparison view
        result_img = visualizer.get_cam_comparison(img_rgb, cam_map, bboxes)
        save_path = os.path.join(args.out_dir, f'{img_name}_comparison.jpg')
        mmcv.imwrite(mmcv.rgb2bgr(result_img), save_path)
        print(f"Saved comparison to: {save_path}")
    elif args.per_bbox:
        # Per-bbox visualization
        result_img = visualizer.draw_cam_per_bbox(
            img_rgb, cam_map, bboxes, labels, scores_filtered, CLASSES
        )
        save_path = os.path.join(args.out_dir, f'{img_name}_per_bbox.jpg')
        mmcv.imwrite(mmcv.rgb2bgr(result_img), save_path)
        print(f"Saved per-bbox visualization to: {save_path}")
    else:
        # Standard overlay
        result_img = visualizer.draw_cam(
            img_rgb, cam_map, bboxes, labels, scores_filtered, CLASSES
        )
        save_path = os.path.join(args.out_dir, f'{img_name}_cam.jpg')
        mmcv.imwrite(mmcv.rgb2bgr(result_img), save_path)
        print(f"Saved CAM overlay to: {save_path}")

        # Also save raw heatmap
        heatmap = cam_utils.apply_colormap(cam_map, args.colormap)
        if heatmap.shape[:2] != img_rgb.shape[:2]:
            try:
                import cv2
                heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
            except ImportError:
                pass
        heatmap_path = os.path.join(args.out_dir, f'{img_name}_heatmap.jpg')
        mmcv.imwrite(mmcv.rgb2bgr(heatmap), heatmap_path)
        print(f"Saved heatmap to: {heatmap_path}")

    # Show if requested
    if args.show:
        try:
            import cv2
            cv2.imshow('CAM Visualization', mmcv.rgb2bgr(result_img))
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except ImportError:
            print("opencv-python is required for display. Install it with: pip install opencv-python")

    print("Done!")


if __name__ == '__main__':
    main()
