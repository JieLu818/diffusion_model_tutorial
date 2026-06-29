"""
采样与可视化脚本
------------------
这个脚本加载训练好的模型，进行采样生成和可视化。

使用场景：
1. 加载训练好的模型 → 生成一批样本
2. 可视化去噪全过程
3. 对比不同调度器的效果
4. 生成 GIF 动图
"""

import os
import sys
import argparse

import torch
from torchvision.utils import save_image, make_grid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DiffusionConfig
from scheduler import NoiseScheduler
from model import UNet
from diffusion import DiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="采样与可视化")
    parser.add_argument("--checkpoint",  type=str, default="output/diffusion_model_final.pt",
                        help="模型权重路径")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="生成批次大小")
    parser.add_argument("--num_batches", type=int, default=1,
                        help="生成几批")
    parser.add_argument("--gif", action="store_true",
                        help="生成去噪过程 GIF")
    return parser.parse_args()


@torch.no_grad()
def generate_samples(pipeline, config, output_dir, batch_size=16, num_batches=1):
    """
    生成样本并保存为网格图
    """
    print("生成样本中...")
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx in range(num_batches):
        samples = pipeline.sample(batch_size=batch_size)
        samples = (samples + 1) / 2  # 映射到 [0, 1]
        samples = samples.clamp(0, 1)

        # 保存单张网格图
        grid = make_grid(samples, nrow=int(batch_size ** 0.5))
        save_image(grid, os.path.join(output_dir, f"samples_batch{batch_idx}.png"))

        # 保存每张单独的小图
        for i in range(min(batch_size, 16)):
            save_image(samples[i],
                       os.path.join(output_dir, f"sample_{batch_idx}_{i}.png"))

        print(f"  批次 {batch_idx + 1}/{num_batches} 完成")


@torch.no_grad()
def visualize_denoising_process(pipeline, config, output_dir, save_gif=False):
    """
    可视化从纯噪声逐步去噪的过程

    在第 0, 50, 100, 200, 500, 750, 999 步分别保存状态，
    展示"噪声 → 数字"的完整过程
    """
    print("可视化去噪过程...")

    # 获取完整的采样过程（所有中间步骤）
    all_steps = pipeline.sample(batch_size=8, return_all_steps=True)
    # all_steps shape: [T+1, B, C, H, W]

    T = config.T
    # 选取关键时间步
    key_steps = [0, T-1, T-5, T-10, T-50, T-100, T-200, T-500, T-800]
    key_steps = sorted(set([s for s in key_steps if s <= T]))
    key_steps = key_steps[::-1]  # 从噪声到清晰（T→0）

    fig, axes = plt.subplots(len(key_steps), 8, figsize=(16, 2 * len(key_steps)))

    for row, step_idx in enumerate(key_steps):
        # 从 [-1, 1] 映射到 [0, 1]
        imgs = (all_steps[step_idx] + 1) / 2
        imgs = imgs.clamp(0, 1)

        for col in range(8):
            img = imgs[col].squeeze().cpu().numpy()
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"t={step_idx}", fontsize=10)

    plt.suptitle("去噪过程（从上到下：纯噪声 → 清晰图像）", fontsize=14)
    plt.tight_layout()

    # 保存倒序版（噪声→清晰）
    save_path = os.path.join(output_dir, "denoising_process.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  去噪过程图已保存: {save_path}")

    # 也保存一张正序版（清晰→噪声，看得更直观）
    key_steps_forward = sorted(key_steps)  # 0 → T
    fig, axes = plt.subplots(len(key_steps_forward), 8,
                             figsize=(16, 2 * len(key_steps_forward)))

    for row, step_idx in enumerate(key_steps_forward):
        imgs = (all_steps[step_idx] + 1) / 2
        imgs = imgs.clamp(0, 1)

        for col in range(8):
            img = imgs[col].squeeze().cpu().numpy()
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"t={step_idx}", fontsize=10)

    plt.suptitle("加噪过程（从上到下：清晰图像 → 纯噪声）", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forward_process_vis.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  加噪过程图已保存: {os.path.join(output_dir, 'forward_process_vis.png')}")

    # ---------- GIF 支持 ----------
    if save_gif:
        try:
            from matplotlib.animation import FuncAnimation
            print("  生成去噪过程 GIF...")

            # 每 10 步取一帧
            frame_step = max(T // 50, 1)
            frame_indices = list(range(T, -1, -frame_step))

            fig, axes = plt.subplots(2, 4, figsize=(8, 4))
            axes_flat = axes.flatten()

            def update(frame_idx):
                step = frame_indices[frame_idx]
                imgs = (all_steps[step] + 1) / 2
                imgs = imgs.clamp(0, 1)
                for i, ax in enumerate(axes_flat):
                    if i < imgs.shape[0]:
                        ax.clear()
                        ax.imshow(imgs[i].squeeze(), cmap="gray", vmin=0, vmax=1)
                        ax.axis("off")
                        if i == 0:
                            ax.set_title(f"t={step}", fontsize=8)
                return axes_flat

            anim = FuncAnimation(fig, update, frames=len(frame_indices),
                                 interval=50, blit=False)
            gif_path = os.path.join(output_dir, "denoising_process.gif")
            anim.save(gif_path, writer="pillow", fps=20)
            plt.close()
            print(f"  GIF 已保存: {gif_path}")

        except Exception as e:
            print(f"  GIF 生成失败（可选功能）: {e}")


def compare_schedules(config, output_dir):
    """
    对比线性调度和余弦调度的 beta 和 alpha_bar 曲线
    """
    print("对比噪声调度...")

    linear_scheduler = NoiseScheduler(
        T=config.T, beta_start=config.beta_start, beta_end=config.beta_end,
        schedule="linear"
    )
    cosine_scheduler = NoiseScheduler(
        T=config.T, beta_start=config.beta_start, beta_end=config.beta_end,
        schedule="cosine"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Beta 曲线
    axes[0].plot(linear_scheduler.betas, label="线性调度", alpha=0.8)
    axes[0].plot(cosine_scheduler.betas, label="余弦调度", alpha=0.8)
    axes[0].set_xlabel("时间步 t")
    axes[0].set_ylabel("β_t（噪声率）")
    axes[0].set_title("β 调度对比")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Alpha bar 曲线
    axes[1].plot(linear_scheduler.alpha_bars, label="线性调度", alpha=0.8)
    axes[1].plot(cosine_scheduler.alpha_bars, label="余弦调度", alpha=0.8)
    axes[1].set_xlabel("时间步 t")
    axes[1].set_ylabel("ᾱ_t（信号保留率）")
    axes[1].set_title("累积信号保留率对比")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. SNR 曲线（信噪比 ≈ ᾱ_t / (1 - ᾱ_t)）
    linear_snr = linear_scheduler.alpha_bars / (1 - linear_scheduler.alpha_bars + 1e-8)
    cosine_snr = cosine_scheduler.alpha_bars / (1 - cosine_scheduler.alpha_bars + 1e-8)
    axes[2].plot(linear_snr, label="线性调度", alpha=0.8)
    axes[2].plot(cosine_snr, label="余弦调度", alpha=0.8)
    axes[2].set_xlabel("时间步 t")
    axes[2].set_ylabel("信噪比")
    axes[2].set_yscale("log")
    axes[2].set_title("信噪比对比")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    compare_path = os.path.join(output_dir, "schedule_comparison.png")
    plt.savefig(compare_path, dpi=150)
    plt.close()
    print(f"  调度对比图已保存: {compare_path}")


def main():
    args = parse_args()
    config = DiffusionConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    scheduler = NoiseScheduler(
        T=config.T,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        schedule=config.beta_schedule,
    )
    model = UNet(config).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  已加载权重: {args.checkpoint}")
    else:
        print(f"  警告：未找到权重文件 {args.checkpoint}，使用随机初始化")

    pipeline = DiffusionPipeline(model, scheduler, config, device)

    # 2. 生成样本
    print("\n[2/4] 生成样本...")
    generate_samples(pipeline, config, output_dir,
                     batch_size=args.batch_size, num_batches=args.num_batches)

    # 3. 可视化去噪过程
    print("\n[3/4] 可视化去噪过程...")
    visualize_denoising_process(pipeline, config, output_dir, save_gif=args.gif)

    # 4. 对比调度器
    print("\n[4/4] 对比调度器...")
    compare_schedules(config, output_dir)

    print(f"\n所有输出保存在: {os.path.abspath(output_dir)}/")


if __name__ == "__main__":
    main()
