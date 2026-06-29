"""
MNIST 训练脚本
----------------
在 MNIST 手写数字数据集上训练一个扩散模型

使用方法：
    python train_mnist.py              # 默认参数训练
    python train_mnist.py --epochs 100  # 训练 100 轮
    python train_mnist.py --schedule cosine  # 用余弦调度

训练完成后会自动生成一次采样结果和损失曲线图
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib
matplotlib.use("Agg")  # 在无 GUI 环境下可用
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DiffusionConfig
from scheduler import NoiseScheduler
from model import UNet
from diffusion import DiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="训练扩散模型")
    parser.add_argument("--epochs",   type=int,   default=None, help="训练轮数")
    parser.add_argument("--batch",    type=int,   default=None, help="批次大小")
    parser.add_argument("--lr",       type=float, default=None, help="学习率")
    parser.add_argument("--T",        type=int,   default=None, help="扩散步数")
    parser.add_argument("--schedule", type=str,   default=None,
                        choices=["linear", "cosine"], help="噪声调度方式")
    parser.add_argument("--save_dir", type=str,   default="output",
                        help="模型和图片保存目录")
    return parser.parse_args()


def get_data_loader(config):
    """
    加载 MNIST 数据集

    数据预处理：
    - 缩放到 [-1, 1] 范围：
      (0~1 的像素值) * 2 - 1 = (-1~1)
      因为扩散模型假设输入在 [-1, 1]，噪声 N(0,1) 也是这个范围
    """
    transform = transforms.Compose([
        transforms.ToTensor(),               # [0, 1]
        transforms.Lambda(lambda x: x * 2 - 1),  # [-1, 1]
    ])

    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    return loader


def save_loss_curve(losses, save_path):
    """
    绘制并保存训练损失曲线
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.6, label="每步损失")

    # 平滑曲线（移动平均）
    window = max(len(losses) // 100, 1)
    smoothed = [sum(losses[max(0,i-window):i+1]) / min(i+1, window+1) for i in range(len(losses))]
    plt.plot(smoothed, linewidth=2, label="平滑曲线", color="red")

    plt.xlabel("训练步数")
    plt.ylabel("MSE 损失")
    plt.title("扩散模型训练损失")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  损失曲线已保存: {save_path}")


def save_sample_grid(pipeline, save_path, batch_size=16, label="generated"):
    """
    生成一批样本并保存为网格图
    """
    samples = pipeline.sample(batch_size=batch_size)
    # 从 [-1, 1] 映射回 [0, 1] 用于保存
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    save_image(samples, save_path, nrow=int(batch_size ** 0.5))
    print(f"  生成样本已保存: {save_path}")


def save_forward_process(pipeline, dataset, save_path, num_steps=10):
    """
    可视化前向加噪过程：展示一张图从干净到纯噪声
    """
    scheduler = pipeline.scheduler
    device = pipeline.device

    # 取第一张图
    x_0 = dataset[0][0].unsqueeze(0).to(device)

    T = scheduler.T
    step_size = T // (num_steps - 1)
    all_steps = [0] + [i * step_size for i in range(1, num_steps)]

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
    for i, t_val in enumerate(all_steps):
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        x_t, _ = scheduler.q_sample(x_0, t)
        img = (x_t[0].cpu() + 1) / 2  # 映射到 [0, 1]
        img = img.clamp(0, 1).squeeze()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"t={t_val}" if t_val < T else "纯噪声")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  加噪过程图已保存: {save_path}")


def main():
    args = parse_args()

    # ---------- 1. 配置 ----------
    config = DiffusionConfig()

    # 命令行参数覆盖
    if args.epochs:   config.epochs = args.epochs
    if args.batch:    config.batch_size = args.batch
    if args.lr:       config.lr = args.lr
    if args.T:        config.T = args.T
    if args.schedule: config.beta_schedule = args.schedule

    # 保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  设备: {device}")
    print(f"  图像尺寸: {config.image_size}×{config.image_size}")
    print(f"  扩散步数 T: {config.T}")
    print(f"  噪声调度: {config.beta_schedule}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  训练轮数: {config.epochs}")
    print(f"  学习率: {config.lr}")
    print(f"{'='*50}\n")

    # ---------- 2. 创建组件 ----------
    print("[1/4] 准备数据...")
    loader = get_data_loader(config)
    print(f"  数据集大小: {len(loader.dataset)} 张图，共 {len(loader)} 个批次")

    print("[2/4] 创建模型和调度器...")
    scheduler = NoiseScheduler(
        T=config.T,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        schedule=config.beta_schedule,
    )
    model = UNet(config).to(device)

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    pipeline = DiffusionPipeline(model, scheduler, config, device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # ---------- 3. 训练 ----------
    print("[3/4] 开始训练...")
    losses = []
    total_steps = len(loader)
    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_loss = 0.0

        for step, (images, _) in enumerate(loader):
            images = images.to(device)

            loss = pipeline.train_step(images, optimizer)
            losses.append(loss)
            epoch_loss += loss

            # 打印进度
            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Epoch [{epoch+1}/{config.epochs}] "
                    f"Step [{step}/{total_steps}] "
                    f"Loss: {loss:.6f} "
                    f"Time: {elapsed:.1f}s"
                )

        avg_loss = epoch_loss / total_steps
        print(f"  → Epoch {epoch+1} 平均损失: {avg_loss:.6f}")

        # 每个 epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_dir, "checkpoints", f"model_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)

    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time:.1f}s")

    # ---------- 4. 保存模型和结果 ----------
    print("[4/4] 保存结果...")

    # 保存最终模型
    model_path = os.path.join(save_dir, "diffusion_model_final.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  模型已保存: {model_path}")

    # 绘制损失曲线
    loss_curve_path = os.path.join(save_dir, "loss_curve.png")
    save_loss_curve(losses, loss_curve_path)

    # 生成样本
    sample_path = os.path.join(save_dir, "generated_samples.png")
    save_sample_grid(pipeline, sample_path, batch_size=config.sample_batch_size)

    # 展示前向加噪过程
    dataset = loader.dataset
    forward_path = os.path.join(save_dir, "forward_process.png")
    save_forward_process(pipeline, dataset, forward_path)

    # 生成多步采样过程
    print("\n生成采样过程可视化...")
    all_steps = pipeline.sample(batch_size=4, return_all_steps=True)
    all_steps = (all_steps + 1) / 2  # 映射到 [0, 1]
    all_steps = all_steps.clamp(0, 1)

    # 选取几个关键时间步
    T = config.T
    key_steps = [0, T//4, T//2, 3*T//4, T-1]
    fig, axes = plt.subplots(len(key_steps), 4, figsize=(8, 2 * len(key_steps)))
    for row, step_idx in enumerate(key_steps):
        for col in range(4):
            img = all_steps[step_idx, col].squeeze()
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"t={step_idx}", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sampling_process.png"), dpi=150)
    plt.close()
    print(f"  采样过程图已保存: {os.path.join(save_dir, 'sampling_process.png')}")

    print(f"\n所有输出保存在: {os.path.abspath(save_dir)}/")
    print("Done! 🎉")


if __name__ == "__main__":
    main()
