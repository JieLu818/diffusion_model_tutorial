"""
U-Net 模型（扩散模型的主干网络）
------------------------------------
扩散模型用的 U-Net 和传统图像分割用的 U-Net 有三个关键区别：

1. 时间嵌入（Time Embedding）
   - 告诉网络当前是去噪过程的第几步
   - 类似 Transformer 中的位置编码，用正弦/余弦函数

2. 下采样和上采样保持对称
   - 编码器逐步降维提取特征
   - 解码器逐步升维恢复原图尺寸
   - 中间有跳跃连接（skip connection）保留细节

3. 输出预测噪声而不是直接预测图像
   - 网络输入：噪声图 x_t + 时间步 t
   - 网络输出：预测的噪声 ε_θ(x_t, t)

网络结构（自上而下）：
  输入 [B, 1, 28, 28]
    │
  卷积嵌入 [B, 128, 28, 28]
    │
  下采样块1 [B, 128, 28, 28] ────→ 跳跃连接 ────→ 上采样块1 [B, 128, 28, 28]
    │ (下采样 ×2)                                 │ (上采样 ×2)
  下采样块2 [B, 256, 14, 14] ────→ 跳跃连接 ────→ 上采样块2 [B, 128, 14, 14]
    │ (下采样 ×2)                                 │ (上采样 ×2)
  中间块 [B, 512, 7, 7] ──────────────────────→ 上采样块3 [B, 256, 7, 7]
                                                       │ (上采样 ×2)
                                                  输出层 [B, 1, 28, 28]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================================================================
# 工具模块
# ====================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间嵌入（Sinusoidal Time Embedding）
    和 Transformer 的位置编码同一套思想

    大白话：
    用一个固定公式生成不同频率的正弦/余弦波，
    让网络"一眼认出来"当前是去噪的第几步

    公式：
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B] 时间步（0 到 T-1 的整数）
        返回: [B, dim] 时间嵌入向量
        """
        device = t.device
        half_dim = self.dim // 2

        # 计算频率分母：10000^(2i/d)
        #   用 exp 和 log 算更稳定
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )

        # t 变成 [B, 1] 方便广播
        t = t.float().unsqueeze(1)  # [B, 1]
        # freqs 变成 [1, half_dim]
        freqs = freqs.unsqueeze(0)  # [1, half_dim]

        # 相乘得到角度：t * freqs
        angles = t * freqs  # [B, half_dim]

        # 拼接 sin 和 cos
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, dim]

        return emb


class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）
    ┌─────────────────┐
    │ 输入 → Conv → SiLU → Conv → + → 输出
    │                ↑                   │
    │          时间嵌入 → 线性层          │
    └─────────────────────────────────────┘

    加了时间嵌入的调节：把时间信息加到特征图上
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()

        # 第一个卷积组：归一化 → 激活 → 卷积
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # 时间嵌入的映射层：把时间向量映射到通道维度
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        # 第二个卷积组
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # 跳跃连接：如果输入输出通道数不同，用 1×1 卷积对齐
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        """
        x:     [B, in_ch, H, W]  特征图
        t_emb: [B, time_emb_dim] 时间嵌入
        """
        # 第一次处理
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # 加入时间信息
        time_scale = self.time_mlp(t_emb)     # [B, out_ch]
        time_scale = time_scale.unsqueeze(-1).unsqueeze(-1)  # [B, out_ch, 1, 1]
        h = h + time_scale

        # 第二次处理
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # 残差连接
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    自注意力模块（Self-Attention）
    让每个像素关注其他所有像素，捕捉全局依赖

    流程：
    输入 → QKV 投影 → 注意力分数 → 加权求和 → 输出
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.GroupNorm(32, channels)

        # QKV 投影
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        # 生成 Q, K, V
        qkv = self.qkv(h)                          # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)              # 三个 [B, C, H, W]

        # 变形为多头格式
        q = q.view(B, self.num_heads, C // self.num_heads, -1)  # [B, heads, head_dim, H*W]
        k = k.view(B, self.num_heads, C // self.num_heads, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, -1)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, H*W, H*W]
        attn = attn.softmax(dim=-1)

        # 加权求和
        out = attn @ v                                 # [B, heads, head_dim, H*W]
        out = out.transpose(1, 2).reshape(B, C, H, W)  # 恢复原形状

        return x + self.proj(out)


# ====================================================================
# U-Net 主体
# ====================================================================

class UNet(nn.Module):
    """
    扩散模型专用的 U-Net

    结构：
    1. 输入卷积 → 嵌入到 base_channels
    2. 下采样路径（Encoder）：3 个阶段，通道数翻倍，尺寸减半
    3. 中间层：残差块 + 注意力
    4. 上采样路径（Decoder）：3 个阶段，通过跳跃连接融合编码器特征
    5. 输出卷积 → 映射回原通道数
    """
    def __init__(self, config):
        super().__init__()
        in_channels = config.image_channels        # 1（灰度图）
        base_ch     = config.base_channels         # 128
        time_dim    = config.time_emb_dim          # 256
        num_blocks  = config.num_res_blocks        # 2
        self.image_size = config.image_size

        # ---------- 时间嵌入网 ----------
        # （照相机记录时间信息给所有残差块使用）
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ---------- 输入层 ----------
        self.input_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # ---------- 编码器（下采样） ----------
        self.encoder_blocks = nn.ModuleList()
        encoder_channels = []
        ch = base_ch

        # 3 个阶段，每个阶段通道数翻倍
        for i, mult in enumerate([1, 2, 4]):
            in_ch = ch
            out_ch = base_ch * mult
            encoder_channels.append(out_ch)

            # 每个阶段包含 num_blocks 个残差块
            blocks = nn.ModuleList()
            for j in range(num_blocks):
                blocks.append(ResidualBlock(in_ch if j == 0 else out_ch, out_ch, time_dim))
            self.encoder_blocks.append(blocks)

            # 下采样（除了最后一个阶段）
            if i < 2:  # 128→256→512，每一步下采样
                self.encoder_blocks.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.encoder_blocks.append(nn.Identity())

        # ---------- 中间层 ----------
        self.mid_block1 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid_attn  = AttentionBlock(base_ch * 4)
        self.mid_block2 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)

        # ---------- 解码器（上采样） ----------
        self.decoder_blocks = nn.ModuleList()

        # 3 个阶段，与编码器对称，通道数逐步减半
        for i, (mult, enc_ch) in enumerate(zip([4, 2, 1], reversed(encoder_channels))):
            in_ch = base_ch * mult
            # 融合跳跃连接后通道数 = 自身特征 + 编码器特征
            blocks = nn.ModuleList()
            for j in range(num_blocks):
                block_in = (in_ch if j == 0 else out_ch) + (enc_ch if j == 0 else 0)
                blocks.append(ResidualBlock(block_in, base_ch * mult // 2, time_dim))
                out_ch = base_ch * mult // 2
            self.decoder_blocks.append(blocks)

            # 上采样（除了最后一个阶段）
            if i < 2:
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    )
                )
            else:
                self.decoder_blocks.append(nn.Identity())

        # ---------- 输出层 ----------
        self.output_norm = nn.GroupNorm(32, base_ch)
        self.output_conv = nn.Conv2d(base_ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        """
        x: [B, C, H, W] 带噪声的图像
        t: [B] 时间步（整数 0 到 T-1）

        返回: [B, C, H, W] 预测的噪声
        """
        # 1. 时间嵌入
        t_emb = self.time_embed(t)

        # 2. 输入卷积
        h = self.input_conv(x)  # [B, base_ch, H, W]

        # 3. 编码器路径（记录跳跃连接）
        skips = []
        for i in range(0, len(self.encoder_blocks), 2):
            blocks = self.encoder_blocks[i]
            downsample = self.encoder_blocks[i + 1]

            # 残差块
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)  # 保存跳跃连接

            # 下采样
            h = downsample(h)

        # 4. 中间层
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # 5. 解码器路径（融合跳跃连接）
        for i in range(0, len(self.decoder_blocks), 2):
            blocks = self.decoder_blocks[i]
            upsample = self.decoder_blocks[i + 1]
            skip = skips.pop()  # 取出对应的编码器特征

            # 融合跳跃连接（在通道维度拼接）
            h = torch.cat([h, skip], dim=1)

            for block in blocks:
                h = block(h, t_emb)

            # 上采样
            h = upsample(h)

        # 6. 输出层
        h = self.output_norm(h)
        h = F.silu(h)
        return self.output_conv(h)

    def predict_noise(self, x_t, timesteps):
        """
        别名方法，语义更清晰
        输入噪声图像 x_t 和时间步 t，预测加入的噪声
        """
        return self.forward(x_t, timesteps)
