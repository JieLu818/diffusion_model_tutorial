"""
生成前向加噪过程示意图 + 噪声调度曲线图 + 时间嵌入可视化 + 重参数化示意
完全不依赖 torch，只用 numpy 和 matplotlib
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_schedule_linear(T, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alpha_bars


def compute_schedule_cosine(T, s=0.008):
    steps = T + 1
    t = np.linspace(0, T, steps)
    f = np.cos((t / T + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas = f[1:] / f[:-1]
    alphas = np.clip(alphas, 0.0, 1.0)
    betas = 1.0 - alphas
    alpha_bars = np.cumprod(alphas)
    return betas, alpha_bars


def draw_forward_process():
    """绘制前向加噪过程示意"""
    print("绘制前向加噪过程示意...")

    T = 1000
    betas, alpha_bars = compute_schedule_linear(T)

    # 生成一个模拟的"8"
    np.random.seed(42)
    img_size = 28
    x = np.linspace(-3, 3, img_size)
    y = np.linspace(-3, 3, img_size)
    X, Y = np.meshgrid(x, y)
    circle1 = np.exp(-((X) ** 2 + (Y - 1.5) ** 2))
    circle2 = np.exp(-((X) ** 2 + (Y + 1.5) ** 2))
    img = (circle1 + circle2).clip(0, 1)
    img = img * 2 - 1  # 映射到 [-1, 1]

    steps = [0, 100, 300, 500, 700, 900, 999]

    fig, axes = plt.subplots(1, len(steps), figsize=(len(steps) * 1.8, 2.5))
    for i, t_val in enumerate(steps):
        noise = np.random.randn(*img.shape) * 0.5
        x_t = np.sqrt(alpha_bars[t_val]) * img + np.sqrt(1 - alpha_bars[t_val]) * noise
        x_t = (x_t + 1) / 2
        x_t = x_t.clip(0, 1)
        axes[i].imshow(x_t, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"t = {t_val}", fontsize=12)
        axes[i].axis("off")

    plt.suptitle("前向加噪过程：一张数字图逐渐被噪声淹没", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "forward_process_illustration.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


def draw_schedule_curves():
    """绘制噪声调度曲线"""
    print("绘制噪声调度曲线...")

    T = 1000
    beta_start, beta_end = 1e-4, 0.02

    betas_lin, alpha_bars_lin = compute_schedule_linear(T, beta_start, beta_end)
    betas_cos, alpha_bars_cos = compute_schedule_cosine(T)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t = np.arange(T)

    # Beta
    axes[0].plot(t, betas_lin, label="线性调度", linewidth=2, color="#2196F3")
    axes[0].plot(t, betas_cos, label="余弦调度", linewidth=2, color="#FF5722")
    axes[0].axhline(y=beta_end, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("时间步 t", fontsize=12)
    axes[0].set_ylabel("β_t（噪声率）", fontsize=12)
    axes[0].set_title("噪声率调度对比", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Alpha bar
    axes[1].plot(t, alpha_bars_lin, label="线性调度", linewidth=2, color="#2196F3")
    axes[1].plot(t, alpha_bars_cos, label="余弦调度", linewidth=2, color="#FF5722")
    axes[1].set_xlabel("时间步 t", fontsize=12)
    axes[1].set_ylabel("ᾱ_t（信号保留率）", fontsize=12)
    axes[1].set_title("原始信号保留率", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    # SNR
    axes[2].plot(t, alpha_bars_lin / (1 - alpha_bars_lin + 1e-8), label="线性", linewidth=2, color="#2196F3")
    axes[2].plot(t, alpha_bars_cos / (1 - alpha_bars_cos + 1e-8), label="余弦", linewidth=2, color="#FF5722")
    axes[2].set_xlabel("时间步 t", fontsize=12)
    axes[2].set_ylabel("信噪比", fontsize=12)
    axes[2].set_yscale("log")
    axes[2].set_title("信噪比变化", fontsize=13)
    axes[2].legend(fontsize=11)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "schedule_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


def draw_reparam_trick():
    """绘制重参数化技巧示意图"""
    print("绘制重参数化技巧示意图...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.text(0.5, 0.7, "重参数化前", fontsize=14, ha="center", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#FFCDD2", alpha=0.6))
    ax.text(0.5, 0.5, "x ∼ N(μ, σ²)\n\n↳ 从分布中采样\n↳ 采样操作不可导！",
            fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#E53935"))
    ax.text(0.5, 0.2, "❌ 无法反向传播", fontsize=12, ha="center", color="#E53935")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax = axes[1]
    ax.text(0.5, 0.7, "重参数化后", fontsize=14, ha="center", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#C8E6C9", alpha=0.6))
    ax.text(0.5, 0.5, "x = μ + σ · ε,   ε ∼ N(0, 1)\n\n↳ 采样和求导分离\n↳ μ, σ 可导, ε 是固定噪声",
            fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#43A047"))
    ax.text(0.5, 0.2, "✅ 梯度可以正常传播", fontsize=12, ha="center", color="#43A047")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "reparameterization_trick.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


def draw_time_embedding():
    """绘制时间嵌入可视化"""
    print("绘制时间嵌入可视化...")

    T_vals, dim = 100, 64
    half_dim = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half_dim) / half_dim)
    t = np.arange(T_vals)[:, None]
    angles = t * freqs[None, :]
    embeddings = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    im = axes[0].imshow(embeddings, aspect="auto", cmap="RdBu",
                        interpolation="nearest", vmin=-1, vmax=1)
    axes[0].set_xlabel("嵌入维度", fontsize=12)
    axes[0].set_ylabel("时间步 t", fontsize=12)
    axes[0].set_title("时间嵌入（正弦编码）热力图", fontsize=13)
    plt.colorbar(im, ax=axes[0])

    for t_sel in [0, 10, 30, 50, 80]:
        axes[1].plot(embeddings[t_sel], label=f"t={t_sel}", alpha=0.8)
    axes[1].set_xlabel("嵌入维度", fontsize=12)
    axes[1].set_ylabel("值", fontsize=12)
    axes[1].set_title("不同时间步的嵌入曲线", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "time_embedding.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


if __name__ == "__main__":
    draw_forward_process()
    draw_schedule_curves()
    draw_reparam_trick()
    draw_time_embedding()
    print("\n所有配图生成完成！")
