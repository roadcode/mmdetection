#!/bin/bash
# Quick test script for YOLOX Grad-CAM (Linux/Mac)
# This script demonstrates basic usage and can be run manually

echo "========================================"
echo "YOLOX Grad-CAM Test Script"
echo "========================================"
echo ""

# Step 1: Check if grad-cam is installed
echo "Step 1: Checking if pytorch-grad-cam is installed..."
python -c "import pytorch_grad_cam; print('✓ pytorch-grad-cam is installed')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "× pytorch-grad-cam is not installed"
    echo "Installing pytorch-grad-cam..."
    pip install grad-cam
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install grad-cam"
        echo "Please run: pip install grad-cam"
        exit 1
    fi
else
    echo "✓ pytorch-grad-cam is already installed"
fi
echo ""

# Step 2: Run basic Grad-CAM demo
echo "Step 2: Running Grad-CAM demo on demo.jpg..."
echo "Command: python tools/analysis_tools/demo_yolox_gradcam.py ..."
echo ""

python tools/analysis_tools/demo_yolox_gradcam.py \
    configs/yolox/yolox_s_8xb8-300e_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth \
    demo/demo.jpg \
    --out-dir outputs/gradcam \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo ""
    echo "× Test failed with CUDA. Trying with CPU..."
    python tools/analysis_tools/demo_yolox_gradcam.py \
        configs/yolox/yolox_s_8xb8-300e_coco.py \
        https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth \
        demo/demo.jpg \
        --out-dir outputs/gradcam \
        --device cpu
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Test completed successfully!"
    echo "Results saved to: outputs/gradcam/"
    echo "========================================"
    echo ""
    echo "You can now check the generated visualization:"
    echo "- outputs/gradcam/demo_gradcam.jpg"
    echo ""
else
    echo ""
    echo "========================================"
    echo "× Test failed"
    echo "========================================"
    echo "Please check the error messages above."
fi
