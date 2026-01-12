@echo off
REM Quick test script for YOLOX Grad-CAM
REM This script demonstrates basic usage and can be run manually

echo ========================================
echo YOLOX Grad-CAM Test Script
echo ========================================
echo.

REM Step 1: Check if grad-cam is installed
echo Step 1: Checking if pytorch-grad-cam is installed...
python -c "import pytorch_grad_cam; print('✓ pytorch-grad-cam is installed')" 2>nul
if %errorlevel% neq 0 (
    echo × pytorch-grad-cam is not installed
    echo Installing pytorch-grad-cam...
    pip install grad-cam
    if %errorlevel% neq 0 (
        echo Error: Failed to install grad-cam
        echo Please run: pip install grad-cam
        pause
        exit /b 1
    )
) else (
    echo ✓ pytorch-grad-cam is already installed
)
echo.

REM Step 2: Run basic Grad-CAM demo
echo Step 2: Running Grad-CAM demo on demo.jpg...
echo Command: python tools/analysis_tools/demo_yolox_gradcam.py configs/yolox/yolox_s_8xb8-300e_coco.py https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth demo/demo.jpg --out-dir outputs/gradcam
echo.

python tools/analysis_tools/demo_yolox_gradcam.py ^
    configs/yolox/yolox_s_8xb8-300e_coco.py ^
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth ^
    demo/demo.jpg ^
    --out-dir outputs/gradcam ^
    --device cuda:0

if %errorlevel% neq 0 (
    echo.
    echo × Test failed. Trying with CPU...
    python tools/analysis_tools/demo_yolox_gradcam.py ^
        configs/yolox/yolox_s_8xb8-300e_coco.py ^
        https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth ^
        demo/demo.jpg ^
        --out-dir outputs/gradcam ^
        --device cpu
)

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo ✓ Test completed successfully!
    echo Results saved to: outputs/gradcam/
    echo ========================================
    echo.
    echo You can now check the generated visualization:
    echo - outputs/gradcam/demo_gradcam.jpg
    echo.
) else (
    echo.
    echo ========================================
    echo × Test failed
    echo ========================================
    echo Please check the error messages above.
)

pause
