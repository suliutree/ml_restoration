# AGENTS.md

## 项目名称
ml-restoration：Noise + JPEG 图像复原（从 0 到 1）

## 项目目标
训练并交付一个图像复原模型：输入经过 **高斯噪声 + JPEG 压缩**退化的图像，输出尽可能接近原始干净图像的复原结果；最终提供可复现训练、可量化评估、可交互 Demo 与可部署推理产物。

## 项目范围与非目标
### 范围
- 合成退化数据：`clean -> (gaussian noise) -> (jpeg) -> degraded`
- 训练复原网络（先从轻量 CNN/ResNet 或 UNet 起步）
- 指标评估：PSNR / SSIM（后续可扩展 LPIPS）
- 工程交付：推理脚本（单图/批量）、模型导出（TorchScript/ONNX 至少一种）、Gradio/Streamlit Demo

### 非目标（阶段 1 不做）
- 追求 SOTA（Transformer / GAN 等复杂路线暂不作为主线）
- 真实相机 RAW/ISP 全链路（当前只针对 RGB 图像的合成退化）
- 多任务混合（超分/去模糊可作为后续扩展，但不影响阶段 1 主线）

## 你需要实现的工作总览
本项目最终需要完成 5 个模块：

1. **数据与退化（Data & Degradation）**
   - 准备干净图像数据集（本地图片或标准数据集）
   - 实现在线退化：随机裁剪 patch + 随机噪声强度 sigma + 随机 JPEG quality
   - 输出训练对：`(degraded, clean)`，并可记录退化参数（sigma/quality）

2. **模型（Model）**
   - 选择稳定、易训练、可控的复原网络（优先小 ResNet/UNet）
   - 输入 degraded，输出 pred（推荐残差学习：预测修复残差）

3. **训练与实验管理（Training & Experimentation）**
   - 实现训练循环：forward/loss/backward/optimizer/scheduler
   - 固定随机种子与配置化（configs），保存 checkpoint
   - 记录训练日志（TensorBoard/W&B 任一）

4. **评估与可视化（Evaluation & Visualization）**
   - 计算 PSNR/SSIM
   - 定期输出对比图：degraded / pred / gt
   - 汇总失败案例（纹理、边缘、平坦区域、强退化参数区间）

5. **推理与交付（Inference & Delivery）**
   - 推理脚本：单图/批量
   - 导出模型：TorchScript/ONNX（至少一种）
   - Demo：Gradio/Streamlit 支持上传图片与结果展示

## 当前进展（截至目前）
已完成“启动与验证阶段（Bootstrap & Sanity Check）”：

- **开发环境可用**
  - 在 MacBook Pro（Apple M4 Pro）上验证 PyTorch MPS 可用，可进行 GPU 加速训练。
- **工程骨架已建立**
  - 已建立项目目录结构：`src/ / scripts/ / data/ / outputs/` 等。
  - 已解决 `src` 模块导入问题（通过将 `src` 设为包并以 `python -m ...` 方式运行脚本）。
- **数据准备就绪**
  - `data/clean_train/` 已放入至少若干张干净图像，满足最小训练数据输入需求。
- **退化管线已实现且验证通过**
  - `src/datasets/degradation.py`：实现高斯噪声 + JPEG 压缩的退化函数，sigma/quality 可控随机采样。
  - `scripts/preview_degradation.py`：已能输出 `outputs/clean.png` 与 `outputs/degraded.png`，退化效果可视化验证通过。

当前状态：已跑通“读图 → 退化 → 输出”的最小闭环，具备进入训练闭环的全部前置条件。

## 环境与运行约定
### 硬件/系统
- 主要开发环境：macOS（Apple M4 Pro），使用 PyTorch MPS 后端进行训练/推理。
- 可选训练环境：Ubuntu 20.04 + NVIDIA GTX 1060 6GB（后续如需长训练/更大实验可迁移）。

### Python 与依赖管理（当前推荐）
- 由于未安装 conda，当前建议使用 `python -m venv .venv` 创建项目隔离环境。
- 每次进入项目后先激活虚拟环境：`source .venv/bin/activate`

### 运行方式约定
- 从项目根目录使用模块方式运行脚本（保证 `src` 可导入）：
  - `python -m scripts.preview_degradation`
  - `python -m scripts.check_mps`

## 代码结构（当前/计划）
- `src/datasets/`：数据集与退化逻辑（已存在 degradation.py）
- `src/models/`：模型定义（待实现）
- `src/metrics/`：PSNR/SSIM 等（待实现）
- `src/utils/`：通用工具（seed、io、logging、config）（待实现）
- `scripts/`：入口脚本（check_mps / preview_degradation 已存在；train/eval/infer/demo 待实现）
- `configs/`：训练/评估配置（待实现）
- `outputs/`：输出图片、日志、checkpoint（已存在用于保存预览结果）
- `data/`：数据集目录（`clean_train` 已存在）

## 下一阶段计划（最近里程碑）
进入“可训练闭环（Trainable Loop）”，按优先级推进：

1. **Dataset/DataLoader**
   - 实现在线 patch 随机裁剪、退化、张量化，返回 `(degraded, clean)`
   - 最小可用：训练集 + 验证集划分（可先用同源数据随机切分）

2. **最小模型 + 训练脚本**
   - 实现一个轻量 ResNet 复原网络（先稳定，再升级）
   - 实现 `scripts/train.py`：训练循环、日志、checkpoint

3. **评估与可视化**
   - 实现 `scripts/eval.py`：PSNR/SSIM
   - 训练中定期保存对比图，便于肉眼检查

4. **推理与交付**
   - `scripts/infer.py` 支持单图/批量
   - 导出 TorchScript/ONNX
   - Gradio Demo（可选作为阶段 1 收尾）

## 关键工程约束与注意事项
- 首阶段优先保证“可复现 + 可评估 + 可交付”，不要提前引入 GAN/Transformer 等高不确定性路线。
- 退化参数范围应配置化（sigma_range / quality_range），并在日志中记录采样分布与关键样例。
- 训练中必须定期做 sanity check：
  - 训练 loss 下降
  - 验证 PSNR/SSIM 上升
  - 可视化对比图中，pred 相比 degraded 有明确改善
- 控制显存/性能：
  - 先用 patch=256、batch 从小到大试；MPS 上优先保证稳定运行。
- 所有脚本以 `python -m scripts.xxx` 方式运行，确保 import 行为一致。

