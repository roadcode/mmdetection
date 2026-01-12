# YOLOX Grad-CAM Visualization

This tool provides Grad-CAM (Gradient-weighted Class Activation Mapping) visualization for YOLOX object detection models in MMDetection. It helps visualize which regions of the image the model focuses on when making detections.

## Features

- ✅ Multiple CAM methods (GradCAM, GradCAM++, XGradCAM, EigenCAM)
- ✅ Support for all YOLOX model variants (nano, tiny, s, m, l, x)
- ✅ Customizable target layers
- ✅ Per-detection heatmap generation
- ✅ Bounding box renormalization for clearer visualization
- ✅ Easy-to-use command-line interface

## Installation

First, install the required dependency:

```bash
pip install grad-cam
```

## Quick Start

### Basic Usage

Generate Grad-CAM visualization with default settings:

```bash
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth \
    demo/demo.jpg \
    --out-dir outputs/gradcam
```

### Advanced Usage

#### Use Different CAM Methods

```bash
# GradCAM++ (often produces sharper results)
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    checkpoint.pth \
    demo/demo.jpg \
    --method gradcam++ \
    --out-dir outputs/gradcam_plusplus
```

Available methods:
- `gradcam` - Standard Grad-CAM (default)
- `gradcam++` - Grad-CAM++ (improved version)
- `xgradcam` - XGrad-CAM  
- `eigencam` - Eigen-CAM (gradient-free)

#### Specify Target Layers

By default, the tool uses `neck.out_convs[-1]` (the last output layer of YOLOXPAFPN). You can specify custom layers:

```bash
# Use backbone's last stage
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    checkpoint.pth \
    demo/demo.jpg \
    --target-layers backbone.stage4

# Use multiple layers (will average their CAMs)
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    checkpoint.pth \
    demo/demo.jpg \
    --target-layers neck.out_convs.0 neck.out_convs.1 neck.out_convs.2
```

Recommended layers for YOLOX:
- `neck.out_convs.2` - Highest resolution features (stride 8)
- `neck.out_convs.1` - Medium resolution features (stride 16)  
- `neck.out_convs.0` - Lowest resolution features (stride 32)
- `neck.top_down_blocks.0` - After top-down path
- `backbone.stage4` - Backbone output

#### Filter Detections

```bash
# Only show high-confidence detections
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    checkpoint.pth \
    demo/demo.jpg \
    --score-thr 0.5

# Only visualize top-3 detections
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    checkpoint.pth \
    demo/demo.jpg \
    --topk 3
```

#### Renormalize CAM in Bounding Boxes

For better visualization, you can renormalize the CAM values within each bounding box:

```bash
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    checkpoint.pth \
    demo/demo.jpg \
    --renormalize
```

This makes the important features within each detection more visible.

#### Save All Intermediate Results

```bash
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    checkpoint.pth \
    demo/demo.jpg \
    --save-all
```

This will save:
- `*_original.jpg` - Original image
- `*_detection.jpg` - Detection results with bounding boxes
- `*_gradcam.jpg` - Grad-CAM visualization

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config` | str | - | Path to model config file (required) |
| `checkpoint` | str | - | Path or URL to checkpoint file (required) |
| `img` | str | - | Path to input image (required) |
| `--out-dir` | str | `outputs/gradcam` | Output directory for visualizations |
| `--method` | str | `gradcam` | CAM method: gradcam, gradcam++, xgradcam, eigencam |
| `--target-layers` | str | None | Target layer names (space-separated) |
| `--score-thr` | float | 0.3 | Detection score threshold |
| `--topk` | int | None | Only show CAM for top-k detections |
| `--device` | str | `cuda:0` | Device for inference (cuda:0, cpu) |
| `--renormalize` | flag | False | Renormalize CAM within bounding boxes |
| `--save-all` | flag | False | Save all intermediate results |

## Understanding the Output

The output visualization shows:
- **Heatmap colors**: 
  - 🔴 Red regions = High activation (model focuses here)
  - 🔵 Blue regions = Low activation (model ignores these)
- **Bounding boxes**: Green boxes show detected objects
- **Labels**: Class name and confidence score for each detection

### Interpreting Grad-CAM

Grad-CAM helps you understand:
1. **What the model "sees"**: High activation areas show which parts of the image are important for detection
2. **Model attention**: Different objects may activate different regions
3. **Debugging**: If activations are in wrong places, it may indicate model issues

## Examples

### Example 1: Person Detection

```bash
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth \
    demo/demo.jpg
```

Expected: Heatmap will highlight person-shaped regions.

### Example 2: Multi-object Scene with GradCAM++

```bash
python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_m_8xb8-300e_coco.py \
    yolox_m_checkpoint.pth \
    my_image.jpg \
    --method gradcam++ \
    --renormalize \
    --score-thr 0.5
```

Expected: Sharper heatmaps with better object localization.

## Troubleshooting

### ImportError: No module named 'pytorch_grad_cam'

**Solution**: Install the grad-cam library:
```bash
pip install grad-cam
```

### CUDA out of memory

**Solution**: Use CPU instead:
```bash
python tools/analysis_tools/demo_yolox_gradcam.py ... --device cpu
```

### No detections found

**Solution**: Lower the score threshold:
```bash
python tools/analysis_tools/demo_yolox_gradcam.py ... --score-thr 0.1
```

### Cannot find target layer

**Solution**: Check available layers by inspecting the model, or use default layers (don't specify `--target-layers`).

## Technical Details

### How It Works

1. **Forward Pass**: Input image through YOLOX model
2. **Detection**: Get bounding boxes, classes, and scores
3. **Gradient Computation**: Backpropagate from detection scores to target layer
4. **Weight Calculation**: Global average pooling of gradients
5. **CAM Generation**: Weighted sum of feature maps → ReLU → Resize
6. **Visualization**: Overlay heatmap on original image with detections

### Recommended Practices

1. **Layer Selection**: Use later layers (e.g., neck outputs) for localization-focused CAM
2. **Renormalization**: Enable `--renormalize` for multi-object scenes
3. **Method Selection**: Try GradCAM++ for sharper results
4. **Threshold Tuning**: Adjust `--score-thr` based on your application needs

## References

- [Original Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [pytorch-grad-cam Library](https://github.com/jacobgil/pytorch-grad-cam)
- [YOLOX Paper](https://arxiv.org/abs/2107.08430)
- [MMDetection Documentation](https://mmdetection.readthedocs.io/)

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}

@inproceedings{selvaraju2017grad,
  title={Grad-cam: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle={ICCV},
  year={2017}
}
```
