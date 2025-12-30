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
已完成“可训练闭环（Trainable Loop）”与基础交付：

- **开发环境可用**
  - 在 MacBook Pro（Apple M4 Pro）上验证 PyTorch MPS 可用，可进行 GPU 加速训练。
- **数据与退化**
  - `src/datasets/degradation.py`：高斯噪声 + JPEG 压缩退化函数已实现并验证。
  - `scripts/preview_degradation.py`：可视化退化样例输出验证通过。
  - `data/clean_train/`：已追加高清图片（当前约 252 张）。
- **Dataset/DataLoader**
  - `src/datasets/paired_dataset.py`：随机 patch 裁剪 + 在线退化 + 张量化 + 训练/验证划分。
- **模型**
  - `src/models/resnet.py`：轻量 ResNet 复原网络（残差学习）。
- **训练与实验管理**
  - `scripts/train.py`：训练循环、checkpoint、固定样例保存、按 epoch 记录 PSNR/SSIM 到 `metrics.jsonl`、保存 best checkpoint。
- **评估**
  - `src/metrics/psnr_ssim.py`：PSNR/SSIM 实现。
  - `scripts/eval.py`：评估并输出 `outputs/eval/metrics.json`。
- **推理与交付**
  - `scripts/infer.py`：单图/批量推理。
  - `scripts/export_torchscript.py`：TorchScript 导出。
  - `scripts/demo_gradio.py`：Gradio 本地 Demo（已禁用公网 share）。

当前状态：训练/评估/推理/导出/本地 Demo 已打通，可进入质量提升与实验迭代。

## 环境与运行约定
### 硬件/系统
- 主要开发环境：macOS（Apple M4 Pro），使用 PyTorch MPS 后端进行训练/推理。
- 可选训练环境：Ubuntu 20.04 + NVIDIA GTX 1060 6GB（后续如需长训练/更大实验可迁移）。

### Python 与依赖管理（当前推荐）
- 由于未安装 conda，当前建议使用 `python -m venv .venv` 创建项目隔离环境。
- 每次进入项目后先激活虚拟环境：`source .venv/bin/activate`

### 运行方式约定
- 从项目根目录使用模块方式运行脚本（保证 `src` 可导入）：
  - `python3 -m scripts.preview_degradation`
  - `python3 -m scripts.check_mps`
  - `python3 -m scripts.train`
  - `python3 -m scripts.eval`
  - `python3 -m scripts.infer`
  - `python3 -m scripts.export_torchscript`
  - `python3 -m scripts.demo_gradio`
- 如遇到系统 Python/venv 混用问题，使用绝对路径：
  - `/Users/***/.venv/bin/python3 -m scripts.xxx`

## 代码结构（当前/计划）
- `src/datasets/`：数据集与退化逻辑（`degradation.py`，`paired_dataset.py`）
- `src/models/`：模型定义（`resnet.py`）
- `src/metrics/`：PSNR/SSIM（`psnr_ssim.py`）
- `src/utils/`：通用工具（seed、io、logging、config）（待补充）
- `scripts/`：
  - 已有：`check_mps.py`、`preview_degradation.py`、`train.py`、`eval.py`、`infer.py`、`export_torchscript.py`、`demo_gradio.py`
- `configs/`：训练/评估配置（待补充）
- `outputs/`：输出图片、日志、checkpoint（训练输出在 `outputs/train/`）
- `data/`：数据集目录（`clean_train`）

## 下一阶段计划（最近里程碑）
进入“质量提升与实验迭代”阶段，按优先级推进：

1. **训练质量提升**
   - 增加训练数据与训练轮数
   - 退化参数课程学习（先轻后重）
   - 尝试更强损失（L1 + SSIM）与学习率调度

2. **配置化与复现**
   - 引入 `configs/`（训练/评估配置文件）
   - 记录更完整的训练日志（csv/jsonl 或 TensorBoard）

3. **推理与交付增强**
   - 支持批量推理、目录输入输出、可选保存中间结果
   - 增加 ONNX 导出（可选）
   - Demo 增强（多图批量、可视化对比、禁用公网分享）

## 关键工程约束与注意事项
- 首阶段优先保证“可复现 + 可评估 + 可交付”，不要提前引入 GAN/Transformer 等高不确定性路线。
- 退化参数范围应配置化（sigma_range / quality_range），并在日志中记录采样分布与关键样例。
- 训练中必须定期做 sanity check：
  - 训练 loss 下降
  - 验证 PSNR/SSIM 上升
  - 可视化对比图中，pred 相比 degraded 有明确改善
- 控制显存/性能：
  - 先用 patch=256、batch 从小到大试；MPS 上优先保证稳定运行。
- 所有脚本以 `python3 -m scripts.xxx` 方式运行，确保 import 行为一致。
