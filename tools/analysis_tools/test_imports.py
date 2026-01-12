"""
简单的导入测试脚本
测试所有必要的模块是否可以正确导入
"""
import sys
from pathlib import Path

print("="*60)
print("YOLOX Grad-CAM 导入测试")
print("="*60)
print()

# Test 1: Basic imports
print("测试 1: 检查基础库...")
try:
    import torch
    print(f"  ✓ PyTorch版本: {torch.__version__}")
    print(f"  ✓ CUDA可用: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  × PyTorch导入失败: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  ✓ OpenCV版本: {cv2.__version__}")
except ImportError as e:
    print(f"  × OpenCV导入失败: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✓ NumPy版本: {np.__version__}")
except ImportError as e:
    print(f"  × NumPy导入失败: {e}")
    sys.exit(1)

print()

# Test 2: MMDetection
print("测试 2: 检查MMDetection...")
try:
    import mmdet
    print(f"  ✓ MMDetection版本: {mmdet.__version__}")
except ImportError as e:
    print(f"  × MMDetection导入失败: {e}")
    print("    请确保已安装mmdetection")
    sys.exit(1)

try:
    from mmdet.apis import init_detector, inference_detector
    print("  ✓ MMDetection APIs可用")
except ImportError as e:
    print(f"  × MMDetection APIs导入失败: {e}")
    sys.exit(1)

print()

# Test 3: pytorch-grad-cam
print("测试 3: 检查pytorch-grad-cam...")
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    print("  ✓ pytorch-grad-cam已安装")
    print("  ✓ 可用方法: GradCAM, GradCAM++, XGradCAM, EigenCAM")
except ImportError:
    print("  × pytorch-grad-cam未安装")
    print("  → 运行: pip install grad-cam")
    grad_cam_available = False
else:
    grad_cam_available = True

print()

# Test 4: Our wrapper module
print("测试 4: 检查Grad-CAM包装器...")
try:
    # Add mmdetection tools to path
    wrapper_path = Path(__file__).parent / 'yolox_gradcam_wrapper.py'
    if wrapper_path.exists():
        print(f"  ✓ 找到wrapper文件: {wrapper_path}")
    else:
        print(f"  × 未找到wrapper文件: {wrapper_path}")
        sys.exit(1)
    
    from yolox_gradcam_wrapper import (
        DetectionTarget,
        YOLOXGradCAMWrapper,
        get_default_target_layers,
        show_cam_on_detection
    )
    print("  ✓ 所有wrapper组件导入成功")
except ImportError as e:
    print(f"  × Wrapper导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Demo script
print("测试 5: 检查演示脚本...")
demo_path = Path(__file__).parent / 'demo_yolox_gradcam.py'
if demo_path.exists():
    print(f"  ✓ 找到演示脚本: {demo_path}")
else:
    print(f"  × 未找到演示脚本: {demo_path}")

print()

# Summary
print("="*60)
print("测试总结")
print("="*60)
if grad_cam_available:
    print("✓ 所有组件已就绪！")
    print()
    print("您现在可以运行:")
    print("  python tools/analysis_tools/demo_yolox_gradcam.py \\")
    print("      configs/yolox/yolox_s_8xb8-300e_coco.py \\")
    print("      [checkpoint] \\")
    print("      [image] \\")
    print("      --out-dir outputs/gradcam")
else:
    print("⚠ 几乎就绪，但需要安装pytorch-grad-cam:")
    print("  pip install grad-cam")
    print()
    print("安装后即可使用Grad-CAM功能。")

print("="*60)
