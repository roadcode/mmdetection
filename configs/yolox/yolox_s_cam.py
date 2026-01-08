# Copyright (c) OpenMMLab. All rights reserved.
"""YOLOX-CAM configuration with CAM visualization enabled.

This config extends the standard YOLOX-S configuration with CAM visualization
hooks for analyzing model predictions during validation/testing.

Usage:
    # Test with CAM visualization
    python tools/test.py configs/yolox/yolox_s_cam.py checkpoints/yolox_s.pth

    # Standalone CAM visualization on a single image
    python tools/analysis_tools/visualize_cam.py \
        configs/yolox/yolox_s_cam.py \
        checkpoints/yolox_s.pth \
        --img path/to/image.jpg \
        --out-dir cam_results
"""

_base_ = ['./yolox_s_8xb8-300e_coco.py']

# CAM Visualization Hook Configuration
# This hook automatically generates CAM visualizations during validation/testing
custom_hooks = [
    # YOLOX mode switch (existing)
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=15,
        priority=48),

    # SyncNorm (existing)
    dict(type='SyncNormHook', priority=48),

    # EMA (existing)
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),

    # CAM Visualization (NEW)
    dict(
        type='CAMVisualizationHook',
        enabled=True,              # Enable/disable CAM generation
        method='eigen',            # CAM method: 'eigen' (fast) or 'grad' (class-specific)
        target_layer='neck',       # Layer to visualize: 'backbone' or 'neck'
        alpha=0.5,                 # Transparency for CAM overlay (0-1)
        colormap='jet',            # Matplotlib colormap: 'jet', 'hot', 'viridis', etc.
        score_thr=0.3,             # Score threshold for filtering detections
        save_dir='work_dirs/cam_vis',  # Directory to save CAM visualizations
        save_heatmap=False,        # Also save raw heatmaps without overlay
        overlay_only=True,         # Only save overlayed images
        per_bbox=False,            # Generate CAM for each bbox separately
        comparison=False,          # Generate side-by-side comparison view
        interval=1,                # Process every N batches (1 = all)
        priority='NORMAL')         # Hook priority
]

# Test-time CAM Visualization Settings
# You can override CAM settings specifically for test phase
test_cfg = dict(
    score_thr=0.01,
    nms=dict(type='nms', iou_threshold=0.65)
)

# Alternative Configurations for Different Use Cases

# ===== Configuration 1: EigenCAM (Fast, Gradient-Free) =====
# Best for: Quick exploration, real-time analysis
# custom_hooks = [
#     dict(
#         type='CAMVisualizationHook',
#         enabled=True,
#         method='eigen',
#         target_layer='neck',
#         alpha=0.5,
#         colormap='jet',
#         save_dir='work_dirs/cam_eigen'
#     )
# ]

# ===== Configuration 2: Grad-CAM (Class-Specific) =====
# Best for: Understanding class-specific activations
# custom_hooks = [
#     dict(
#         type='CAMVisualizationHook',
#         enabled=True,
#         method='grad',
#         target_layer='neck',
#         alpha=0.5,
#         colormap='hot',
#         save_dir='work_dirs/cam_grad'
#     )
# ]

# ===== Configuration 3: Per-Bbox Visualization =====
# Best for: Analyzing individual detections
# custom_hooks = [
#     dict(
#         type='CAMVisualizationHook',
#         enabled=True,
#         method='eigen',
#         per_bbox=True,          # Enable per-bbox CAM
#         save_dir='work_dirs/cam_per_bbox'
#     )
# ]

# ===== Configuration 4: Comparison View =====
# Best for: Presentation, debugging
# custom_hooks = [
#     dict(
#         type='CAMVisualizationHook',
#         enabled=True,
#         method='eigen',
#         comparison=True,        # Enable comparison view
#         save_dir='work_dirs/cam_comparison'
#     )
# ]

# ===== Configuration 5: Backbone-Level CAM =====
# Best for: Understanding low-level features
# custom_hooks = [
#     dict(
#         type='CAMVisualizationHook',
#         enabled=True,
#         method='eigen',
#         target_layer='backbone',  # Use backbone instead of neck
#         alpha=0.6,
#         save_dir='work_dirs/cam_backbone'
#     )
# ]

# ===== Configuration 6: High-Quality Output =====
# Best for: Publication, detailed analysis
# custom_hooks = [
#     dict(
#         type='CAMVisualizationHook',
#         enabled=True,
#         method='grad',
#         target_layer='neck',
#         alpha=0.6,
#         colormap='viridis',
#         score_thr=0.5,
#         save_heatmap=True,      # Save both overlay and raw heatmap
#         overlay_only=False,
#         save_dir='work_dirs/cam_high_quality'
#     )
# ]

# ===== Configuration 7: Batch Processing (Reduce Output) =====
# Best for: Large dataset analysis
# custom_hooks = [
#     dict(
#         type='CAMVisualizationHook',
#         enabled=True,
#         method='eigen',
#         interval=10,            # Process every 10 batches
#         save_dir='work_dirs/cam_batch'
#     )
# ]

# Visualization Tips:
# 1. Use 'target_layer=neck' for balanced localization (default)
# 2. Use 'target_layer=backbone' for coarse, low-level features
# 3. Adjust 'alpha' for overlay transparency (0.3-0.7 recommended)
# 4. Try different colormaps: 'jet', 'hot', 'cool', 'viridis', 'rainbow'
# 5. Lower 'score_thr' to see more detections (may be noisy)
# 6. Use 'per_bbox=True' to analyze individual object activations
# 7. Use 'comparison=True' for comprehensive analysis (3-panel view)
