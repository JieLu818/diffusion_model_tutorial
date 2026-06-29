"""
扩散流程管理（Diffusion Pipeline）
--------------------------------------
把 NoiseScheduler 和 UNet 组装起来，提供：
1. 训练一步的损失计算（train_step）
2. 完整的采样生成流程（sample）

大白话：
  训练时：拿一张图 → 随机加噪声 → 让 U-Net 猜噪声 → 算损失 → 反向传播
  生成时：从纯噪声开始 → 让 U-Net 一步步猜噪声 → 逐步去噪 → 得到新图
"""

import torch
import torch.nn.functional as F


class DiffusionPipeline:
    """
    扩散流程管理类
    组装 scheduler + model，提供训练和采样接口
    """
    def __init__(self, model, scheduler, config, device="cpu"):
        self.model     = model
        self.scheduler = scheduler
        self.config    = config
        self.device    = device

    def train_step(self, x_0, optimizer):
        """
        单步训练逻辑

        流程：
        1. 随机采样时间步 t（告诉网络去噪到第几步了）
        2. 对 x_0 加噪得到 x_t
        3. U-Net 预测噪声 ε_θ(x_t, t)
        4. 计算 MSE 损失

        参数：
            x_0:      [B, C, H, W] 原始图像
            optimizer: PyTorch 优化器

        返回：
            loss: 标量损失值
        """
        B = x_0.shape[0]

        # ---------- 1. 随机采样时间步 ----------
        # 每个样本随机选一个去噪进度
        t = torch.randint(0, self.scheduler.T, (B,), device=self.device)

        # ---------- 2. 前向加噪 ----------
        # 直接从 x_0 跳到 x_t（不需要一步步走）
        noise = torch.randn_like(x_0)
        x_t, _ = self.scheduler.q_sample(x_0, t, noise)

        # ---------- 3. 预测噪声 ----------
        # U-Net 输入：噪声图 + 时间步
        # U-Net 输出：预测的噪声（要与真实噪声 noise 计算 MSE）
        pred_noise = self.model(x_t, t)

        # ---------- 4. 计算损失 ----------
        # DDPM 的简化损失：L_simple = MSE(ε, ε_θ(x_t, t))
        #   不需要算 KL 散度，直接让网络学猜噪声就行
        loss = F.mse_loss(pred_noise, noise)

        # ---------- 5. 反向传播 ----------
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_steps=False):
        """
        采样生成（反向去噪过程）

        流程：
        从纯噪声 x_T 开始，逐步去噪直到 x_0：

        for t = T, T-1, ..., 1:
            1. U-Net 预测当前噪声 ε_θ(x_t, t)
            2. 用公式计算 x_{t-1} 的均值
            3. 加上随机噪声（t > 1 时才加）
            4. 得到 x_{t-1}

        参数：
            batch_size:     一次生成多少张图
            return_all_steps: 是否返回所有中间结果（用于可视化）

        返回：
            x_0:         生成结果 [B, C, H, W]
            或 all_steps: 每一步的图像 [steps+1, B, C, H, W]
        """
        self.model.eval()
        T = self.scheduler.T
        device = self.device

        # ---------- 1. 从纯噪声开始 ----------
        # x_T ~ N(0, I)
        shape = (batch_size, self.config.image_channels,
                 self.config.image_size, self.config.image_size)
        x = torch.randn(shape, device=device)

        if return_all_steps:
            all_steps = [x.cpu()]

        # ---------- 2. 逐步去噪 ----------
        for t_step in range(T - 1, -1, -1):
            t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)

            # 预测噪声
            pred_noise = self.model(x, t)

            # 获取系数
            coeff1, coeff2 = self.scheduler.get_pred_mean_coeffs(x, t, pred_noise)

            # 反向步骤均值 μ_θ = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ)
            pred_mean = coeff1 * (x - coeff2 * pred_noise)

            # 加噪声（最后一步 t=0 不加）
            if t_step > 0:
                variance = self.scheduler.get_variance(t_step)
                noise = torch.randn_like(x)
                x = pred_mean + torch.sqrt(variance) * noise
            else:
                x = pred_mean

            if return_all_steps:
                all_steps.append(x.cpu())

        self.model.train()

        # 确保输出在有效像素范围 [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)

        if return_all_steps:
            return torch.stack(all_steps, dim=0)  # [T+1, B, C, H, W]
        return x

    @torch.no_grad()
    def sample_ddim(self, batch_size=16, ddim_steps=50, eta=0.0):
        """
        DDIM 采样（加速版）
        Denoising Diffusion Implicit Models

        和 DDPM 的核心区别：
        - 不需要走完 T 步，只走 ddim_steps 步
        - 确定性的（eta=0）或带噪的（eta>0）

        参数：
            ddim_steps: 加速后的步数（比如 T=1000，ddim_steps=50 就是 20 倍加速）
            eta:        噪声系数，0=确定性，1=和 DDPM 一样
        """
        self.model.eval()
        T = self.scheduler.T
        device = self.device

        # 计算跳步索引
        step_ratio = T // ddim_steps
        timesteps = torch.linspace(T - 1, 0, ddim_steps, device=device).long()

        shape = (batch_size, self.config.image_channels,
                 self.config.image_size, self.config.image_size)
        x = torch.randn(shape, device=device)

        for i, t_step in enumerate(timesteps):
            t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)

            pred_noise = self.model(x, t)

            # DDIM 公式
            alpha_bar = self.scheduler.alpha_bars[t_step].to(device)
            alpha_bar_prev = self.scheduler.alpha_bars[t_step - step_ratio].to(device) if t_step > step_ratio else torch.tensor(1.0, device=device)

            # 预测 x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # 方向噪声
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

            # 更新 x
            c1 = torch.sqrt(alpha_bar_prev)  # x_0 系数
            c2 = torch.sqrt(1 - alpha_bar_prev - sigma ** 2)  # 方向系数
            noise = torch.randn_like(x)

            x = c1 * pred_x0 + c2 * pred_noise + sigma * noise

        self.model.train()
        return torch.clamp(x, -1.0, 1.0)
