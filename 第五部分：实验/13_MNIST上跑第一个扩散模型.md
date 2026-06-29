# MNIST 上跑第一个扩散模型

> **一句话总结**：本章是动手实验的核心——运行训练脚本，观察损失曲线，并从纯噪声中生成第一批手写数字。

## 环境准备

### 安装依赖

```bash
# 激活你的 conda 环境
conda activate diffusion

# 安装依赖
cd /Users/jie/Documents/多维感知/多维感知/diffusion/code
pip install -r requirements.txt
```

依赖包括：
- `torch>=1.13.0` — PyTorch 深度学习框架
- `torchvision` — 数据集和图像处理
- `matplotlib` — 绘图和可视化
- `tqdm` — 进度条

### 验证安装

```bash
python -c "import torch; import torchvision; print('PyTorch', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

如果没有 GPU，代码会自动在 CPU 上运行（MNIST 28×28 图片 CPU 也够用）。

## 首次训练

### 使用默认参数

```bash
cd /Users/jie/Documents/多维感知/多维感知/diffusion/code
python train_mnist.py
```

**参数说明**（默认值）：
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 50 | 训练轮数 |
| `--batch` | 128 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--T` | 1000 | 扩散步数 |
| `--schedule` | linear | 噪声调度 |
| `--save_dir` | output | 输出目录 |

### 预计运行时间

| 配置 | CPU（Mac M1） | 预计 |
|---|---|---|
| 50 epochs, T=1000 | ~45 分钟 | 约 1 小时 |
| 50 epochs, T=500 | ~25 分钟 | 更快的实验 |
| 10 epochs（快速测试） | ~5 分钟 | 看看效果 |

### 训练过程中的输出

```
设备: cpu
图像尺寸: 28×28
扩散步数 T: 1000
噪声调度: linear
批次大小: 128
训练轮数: 50

[1/4] 准备数据...
  数据集大小: 60000 张图，共 469 个批次
[2/4] 创建模型和调度器...
  模型参数量: 3,623,425
[3/4] 开始训练...
  Epoch [1/50] Step [0/469] Loss: 1.0452  Time: 0.0s
  Epoch [1/50] Step [50/469] Loss: 0.2377  Time: 65.3s
  ...
  → Epoch 1 平均损失: 0.2083
  Epoch [2/50] Step [0/469] Loss: 0.0867  Time: 345.1s
  ...
```

**损失下降轨迹**（典型值）：
```
Epoch 1:  ≈0.20  ← 网络开始理解"噪声的长相"
Epoch 5:  ≈0.08  ← 恢复出模糊的数字轮廓
Epoch 10: ≈0.05  ← 细节开始出现
Epoch 25: ≈0.03  ← 大部分数字已经清晰
Epoch 50: ≈0.02  ← 接近收敛
```

## 训练结束后

训练完成后会自动：

1. **保存模型权重** → `output/diffusion_model_final.pt`
2. **绘制损失曲线** → `output/loss_curve.png`
3. **生成第一批样本** → `output/generated_samples.png`
4. **可视化前向加噪** → `output/forward_process.png`
5. **可视化采样过程** → `output/sampling_process.png`

```
output/
├── diffusion_model_final.pt   ← 训练好的模型
├── loss_curve.png             ← 损失曲线
├── generated_samples.png      ← 生成的样本
├── forward_process.png        ← 前向加噪过程
├── sampling_process.png       ← 去噪过程
└── checkpoints/               ← 中间检查点
    ├── model_epoch10.pt
    ├── model_epoch20.pt
    ├── model_epoch30.pt
    ├── model_epoch40.pt
    └── model_epoch50.pt
```

## 损失曲线解读

训练脚本会生成损失曲线图，横轴是训练步数，纵轴是 MSE 损失。

```mermaid
flowchart LR
    A["损失≈1.0\n初始：还没学会"] --> B["损失≈0.2\n几轮后：学会了\n基本去噪"] --> C["损失≈0.05\n十轮后：精细\n去噪"] --> D["损失≈0.02\n收敛：接近\n最优"]
```

**正常损失曲线应该是**：
- 从 **~1.0** 急剧下降到 **~0.2**（第 1 轮）
- 缓慢下降到 **~0.05**（第 10 轮）
- 趋于平稳在 **~0.02**（第 50 轮）

如果损失没有下降或下降异常，检查：
- 学习率是否太小/太大（建议 1e-3 到 1e-4）
- 数据是否归一化到了 [-1, 1]
- 模型是否创建成功（检查参数量）

## 观察第一批生成结果

训练完成后，`generated_samples.png` 看起来应该类似：

```
┌─────┬─────┬─────┬─────┐
│  3  │  7  │  0  │  8  │
├─────┼─────┼─────┼─────┤
│  2  │  4  │  5  │  9  │
├─────┼─────┼─────┼─────┤
│  1  │  6  │  8  │  0  │
├─────┼─────┼─────┼─────┤
│  3  │  7  │  4  │  2  │
└─────┴─────┴─────┴─────┘
```

**预期效果**：
- 50 epochs 训练后，大部分数字**可以辨认**
- 少数可能模糊、变形（正常现象，毕竟只有 50 轮）
- 没有两个完全一样的数字（说明生成的是"新图"）

## 生成更多样本

```bash
python sample_and_visualize.py --checkpoint output/diffusion_model_final.pt --num_batches 5
```

这会生成 5 批（每批 16 张）不同的手写数字。

## 快速实验：只训练 10 轮

想快速看看效果可以：

```bash
python train_mnist.py --epochs 10 --save_dir output_quick
```

10 轮训练虽然最终效果不如 50 轮，但你可以看到**模型在第几轮开始出现可辨认的数字**。

## 要点回顾

1. 用 `python train_mnist.py` 启动训练，所有输出保存在 `output/` 目录
2. 训练过程中观察**损失曲线**，正常应该平滑下降
3. 训练结束后自动生成**样本**，可以看到手写数字
4. 训练时间约 **45 分钟**（CPU）到 **5 分钟**（GPU）
5. 可以用 `--epochs` 快速测试效果

---

**继续阅读**：[[14_生成效果可视化]]
