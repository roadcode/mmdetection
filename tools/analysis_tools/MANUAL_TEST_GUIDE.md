# 手动测试指南 - YOLOX Grad-CAM

由于环境限制，请按照以下步骤手动测试代码：

## 步骤 1: 检查Python环境

在命令行中运行：
```cmd
python --version
```

应该显示Python版本（建议3.7+）

## 步骤 2: 安装依赖

```cmd
pip install grad-cam
```

如果已安装mmdetection，应该已经有torch, torchvision, numpy, opencv等依赖。

## 步骤 3: 检查demo图片是否存在

确认文件存在：
```cmd
dir demo\demo.jpg
```

## 步骤 4: 运行基础测试

### 方式A: 使用测试脚本（推荐）

**Windows:**
```cmd
cd d:\code\grad_cam\mmdetection
.\tools\analysis_tools\test_gradcam.bat
```

**如果遇到权限问题，以管理员身份运行PowerShell:**
```powershell
cd d:\code\grad_cam\mmdetection
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\tools\analysis_tools\test_gradcam.bat
```

### 方式B: 直接运行Python脚本

```cmd
cd d:\code\grad_cam\mmdetection

python tools/analysis_tools/demo_yolox_gradcam.py ^
    configs/yolox/yolox_s_8xb8-300e_coco.py ^
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth ^
    demo/demo.jpg ^
    --out-dir outputs/gradcam ^
    --device cuda:0
```

**如果没有CUDA，使用CPU:**
```cmd
python tools/analysis_tools/demo_yolox_gradcam.py ^
    configs/yolox/yolox_s_8xb8-300e_coco.py ^
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth ^
    demo/demo.jpg ^
    --out-dir outputs/gradcam ^
    --device cpu
```

## 预期输出

### 成功的情况：

1. **控制台输出应该显示：**
```
Loading model from [checkpoint URL]...
Setting up target layers...
Using target layers: [...]
Using CAM method: gradcam

Processing image: demo/demo.jpg
Generating CAM for X detections...
Saved Grad-CAM visualization to: outputs/gradcam/demo_gradcam.jpg

Done!
```

2. **生成的文件：**
- `outputs/gradcam/demo_gradcam.jpg` - 带Grad-CAM热力图的可视化结果

3. **可视化应该显示：**
- 原始图像
- 红色/暖色区域 = 模型高度激活的区域（关注的地方）
- 蓝色/冷色区域 = 模型低激活的区域
- 绿色边界框围绕检测到的物体
- 标签显示类别名称和置信度分数

## 常见错误排查

### 错误 1: ImportError: No module named 'pytorch_grad_cam'

**解决方法：**
```cmd
pip install grad-cam
```

### 错误 2: CUDA out of memory

**解决方法：** 使用CPU
```cmd
python tools/analysis_tools/demo_yolox_gradcam.py ... --device cpu
```

### 错误 3: No detections found with score >= 0.3

**解决方法：** 降低阈值
```cmd
python tools/analysis_tools/demo_yolox_gradcam.py ... --score-thr 0.1
```

### 错误 4: Could not find layer: xxx

**解决方法：** 不指定target-layers，使用默认值
```cmd
# 移除 --target-layers 参数
python tools/analysis_tools/demo_yolox_gradcam.py [config] [checkpoint] [img]
```

### 错误 5: FileNotFoundError: demo/demo.jpg

**解决方法：** 使用您自己的图片
```cmd
python tools/analysis_tools/demo_yolox_gradcam.py ... your_image.jpg
```

## 高级测试

### 测试 1: 使用GradCAM++

```cmd
python tools/analysis_tools/demo_yolox_gradcam.py ^
    configs/yolox/yolox_s_8xb8-300e_coco.py ^
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth ^
    demo/demo.jpg ^
    --method gradcam++ ^
    --out-dir outputs/gradcam_plusplus
```

### 测试 2: 边界框重新归一化

```cmd
python tools/analysis_tools/demo_yolox_gradcam.py ^
    configs/yolox/yolox_s_8xb8-300e_coco.py ^
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth ^
    demo/demo.jpg ^
    --renormalize ^
    --out-dir outputs/gradcam_renorm
```

### 测试 3: 保存所有中间结果

```cmd
python tools/analysis_tools/demo_yolox_gradcam.py ^
    configs/yolox/yolox_s_8xb8-300e_coco.py ^
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20211121_095711-4592a793.pth ^
    demo/demo.jpg ^
    --save-all ^
    --out-dir outputs/gradcam_full
```

应该生成：
- `demo_original.jpg` - 原始图像
- `demo_detection.jpg` - 仅检测结果
- `demo_gradcam.jpg` - Grad-CAM可视化

## 验证清单

请测试后确认：

- [ ] 代码成功运行无报错
- [ ] 生成了输出图像
- [ ] 热力图正确显示在检测物体上
- [ ] 边界框和标签清晰可见
- [ ] 不同CAM方法都能工作（gradcam, gradcam++）
- [ ] --renormalize参数有效果
- [ ] --save-all保存了所有文件

## 需要报告的信息

如果遇到问题，请提供：

1. 完整的错误信息
2. Python版本 (`python --version`)
3. PyTorch版本 (`python -c "import torch; print(torch.__version__)"`)
4. CUDA是否可用 (`python -c "import torch; print(torch.cuda.is_available())"`)
5. 使用的命令
6. 您的图像特征（大小、内容等）

## 测试完成后

如果一切正常，您可以：

1. 尝试在您自己的图像上运行
2. 试验不同的YOLOX模型（tiny, s, m, l, x）
3. 探索不同的目标层
4. 比较不同CAM方法的效果

祝测试顺利！🎉
