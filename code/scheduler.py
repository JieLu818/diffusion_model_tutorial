"""
噪声调度器（Noise Scheduler）
--------------------------------
核心工作：
1. 定义 beta 调度（线性 / 余弦）
2. 计算前向扩散所需的所有系数：alpha, alpha_bar 等
3. 提供前向加噪函数 q_sample
4. 提供反向去噪的系数计算函数

术语对照：
  beta_t     = 第 t 步的噪声方差
  alpha_t    = 1 - beta_t（信号保留比例）
  alpha_bar_t = ∏(alpha_1 * alpha_2 * ... * alpha_t) 累积信号保留比例
"""

import torch
import math


class NoiseScheduler:
    """
    噪声调度器
    管理扩散过程的所有系数计算

    大白话理解：
    - beta 控制「每一步加多少噪声」，越大噪得越狠
    - alpha_bar 控制「原始图还剩多少」，从 1 一路降到接近 0
    - 前向加噪 q_sample: 给定 x0 和时间 t，直接算出 xt
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, schedule="linear"):
        self.T = T

        # ---------- 1. 定义 beta 调度 ----------
        if schedule == "linear":
            # 线性调度：从 beta_start 均匀增加到 beta_end
            #   DDPM 原版用的就是线性调度，简单粗暴效果好
            self.betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":
            # 余弦调度：beta 变化更平滑，前几步噪声增加更慢
            #   改进版调度，让图像在早期保留更多信息
            self.betas = self._cosine_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # ---------- 2. 计算核心系数 ----------
        # alphas: 信号保留率（每一步）
        self.alphas = 1.0 - self.betas

        # alpha_bars: 累积信号保留率（从第 1 步到第 t 步的乘积）
        #   大白话：到第 t 步时，原始图像还剩多少比例
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # ---------- 3. 后验分布系数（用于反向去噪） ----------
        # 这些是 q(x_{t-1} | x_t, x_0) 的均值和方差系数
        #   DDPM 公式 (6) 和 (7)
        self.posterior_variances = self.betas * (1 - self.alpha_bars[:-1]) / (1 - self.alpha_bars)

    @staticmethod
    def _cosine_schedule(T, s=0.008):
        """
        余弦调度（cosine schedule）
        Improved DDPM 中提出的更平滑的噪声调度

        公式：
            beta_t = 1 - alpha_t
            alpha_t = f(t) / f(t-1)
            f(t) = cos((t/T + s) / (1 + s) * π/2)^2
        """
        steps = T + 1
        t = torch.linspace(0, T, steps)
        # 余弦函数：在 [0, T] 上从接近 1 下降到接近 0
        f = torch.cos((t / T + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas = f[1:] / f[:-1]  # alpha_t = f(t) / f(t-1)
        # 裁剪防止极端值
        alphas = torch.clamp(alphas, 0.0, 1.0)
        return 1.0 - alphas

    def q_sample(self, x_0, t, noise=None):
        """
        前向加噪：给定原始图像 x_0，直接算出第 t 步的噪声图像 x_t

        数学公式：
            x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
            其中 ε ~ N(0, I) 是高斯噪声

        参数：
            x_0:  原始图像 [B, C, H, W]
            t:    时间步 [B]（每个样本可以有不同的 t）
            noise: 可选的噪声（不传就随机生成）

        返回：
            x_t:  加噪后的图像
            noise: 使用的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 提取 alpha_bar_t
        #   gather: 从 alpha_bars 中取出对应 t 位置的值
        #   然后 reshape 成 [B, 1, 1, 1] 方便广播
        sqrt_alpha_bar = self.alpha_bars[t].sqrt()
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)

        sqrt_one_minus_alpha_bar = (1 - self.alpha_bars[t]).sqrt()
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)

        # 核心公式：x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def get_variance(self, t):
        """
        反向去噪的方差（schedule 中已计算好，这里直接取出）

        DDPM 中后验方差 q(x_{t-1} | x_t, x_0) 的两种选择：
            1. β_t（直接使用噪声率）
            2. β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t（更精确）
        """
        if t == 0:
            return 0.0  # 第 0 步不需要方差
        return self.posterior_variances[t - 1]

    def get_pred_mean_coeffs(self, x_t, t, pred_noise):
        """
        反向去噪的均值系数

        给定 x_t 和模型预测的噪声 pred_noise，
        计算 x_{t-1} 的均值

        DDPM 公式 (15) 的分解：
            μ_θ(x_t, t) = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ)

        但我们不在这里直接计算 x_{t-1}，
        而是返回所有系数，让采样器灵活组合
        """
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)

        # 反向均值系数 1/α_t
        coeff1 = 1.0 / alpha_t.sqrt()

        # 噪声缩放系数 β_t / √(1-ᾱ_t)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        coeff2 = beta_t / sqrt_one_minus_alpha_bar

        return coeff1, coeff2
