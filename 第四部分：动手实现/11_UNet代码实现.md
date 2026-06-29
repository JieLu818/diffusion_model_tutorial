# U-Net 代码实现

> **一句话总结**：本章逐段讲解 U-Net 的 PyTorch 实现代码，包括时间嵌入、残差块、注意力模块和整体架构。

## 代码概览

完整的 U-Net 实现在 `code/model.py` 中，约 200 行代码。我们把它拆成几个模块：

## 1. 正弦时间嵌入

```python
class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间嵌入（和 Transformer 位置编码相同）
    
    输入：t [B] — 时间步
    输出：emb [B, dim] — 时间嵌入向量
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        
        # 频率分母：10000^(2i/d)
        # 用 exp(log) 形式更稳定
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )
        
        # t [B, 1] × freqs [1, half_dim] = angles [B, half_dim]
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        
        # 拼接 sin 和 cos
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        
        return emb
```

**关键点**：
- 不需要学习参数，是**固定的**编码
- 输出维度是 `dim`，拼接了 sin 和 cos 各 `dim/2` 维
- 数值稳定：用 `exp(log())` 而不是直接算 `10000^(2i/d)`

## 2. 残差块

```python
class ResidualBlock(nn.Module):
    """
    带时间嵌入的残差块
    
    结构：Conv → SiLU → Conv + 跳跃连接
          中间加了时间嵌入的调节
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        
        # 第一次卷积
        self.norm1 = nn.GroupNorm(32, in_ch)  # 32 组归一化
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # 时间嵌入 → 通道数映射
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        
        # 第二次卷积
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # 跳跃连接（如果通道数变化）
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        # 第一次处理
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 注入时间信息
        time_scale = self.time_mlp(t_emb)
        time_scale = time_scale[:, :, None, None]  # [B, C, 1, 1]
        h = h + time_scale
        
        # 第二次处理
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # 残差连接
        return h + self.skip(x)
```

**关键点**：
- 用 `GroupNorm(32, ...)` 而不是 BatchNorm（小批次更稳定）
- 时间嵌入通过 `time_mlp` 映射到 `out_ch` 维度，然后**加到卷积中间特征上**
- 输入输出通道数不同时，跳跃连接用 1×1 卷积对齐

> **大白话**：残差块的作用是在不改变特征图大小的前提下，对特征进行加工。时间嵌入告诉它"现在该用什么力度去噪"。

## 3. 自注意力模块

```python
class AttentionBlock(nn.Module):
    """多头自注意力"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)  # QKV 投影
        self.proj = nn.Conv2d(channels, channels, 1)     # 输出投影
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        # QKV 一起算更高效
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 多头变形
        q = q.view(B, self.num_heads, C // self.num_heads, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, -1)
        
        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return x + self.proj(out)
```

**关键点**：
- 用 `nn.Conv2d(channels, channels * 3, 1)` 同时算 Q、K、V
- 多头：把通道分成 `num_heads` 组，每组独立做注意力
- 残差连接：`x + self.proj(out)`

## 4. 完整的 U-Net

```python
class UNet(nn.Module):
    """
    扩散模型 U-Net
    
    输入：x [B, C, H, W]，t [B]
    输出：预测的噪声 [B, C, H, W]
    """
    def __init__(self, config):
        super().__init__()
        base_ch = config.base_channels      # 128
        time_dim = config.time_emb_dim     # 256
        
        # 时间嵌入网络
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # 输入卷积
        self.input_conv = nn.Conv2d(1, base_ch, 3, padding=1)
        
        # 编码器（3 个阶段）
        self.encoder_blocks = nn.ModuleList()
        # 解码器（3 个阶段，对称结构）
        self.decoder_blocks = nn.ModuleList()
        # 中间层
        self.mid_block1 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid_attn = AttentionBlock(base_ch * 4)
        self.mid_block2 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)
        # 输出层
        self.output_conv = nn.Conv2d(base_ch, 1, 3, padding=1)
```

完整的 forward 过程：

```python
def forward(self, x, t):
    # 1. 时间嵌入
    t_emb = self.time_embed(t)
    
    # 2. 输入卷积 + 编码器（保存跳跃连接）
    h = self.input_conv(x)
    skips = []
    for encoder_blocks, downsample in zip(...):
        for block in encoder_blocks:
            h = block(h, t_emb)
        skips.append(h)
        h = downsample(h)
    
    # 3. 中间层
    h = self.mid_block1(h, t_emb)
    h = self.mid_attn(h)
    h = self.mid_block2(h, t_emb)
    
    # 4. 解码器（拼接跳跃连接）
    for decoder_blocks, upsample in zip(...):
        skip = skips.pop()
        h = torch.cat([h, skip], dim=1)  # 融合跳跃连接
        for block in decoder_blocks:
            h = block(h, t_emb)
        h = upsample(h)
    
    # 5. 输出层
    h = F.silu(self.output_norm(h))
    return self.output_conv(h)
```

## 验证参数量

```python
from model import UNet
from config import DiffusionConfig

model = UNet(DiffusionConfig())
params = sum(p.numel() for p in model.parameters())
print(f"参数量: {params:,}")
# → 参数量: 约 3,600,000（360 万参数）
```

> **大白话**：360 万参数对于一个扩散模型来说很小（DDPM 原版有 3550 万参数）。因为 MNIST 只有 28×28，网络可以更轻量。

## 要点回顾

1. U-Net 由 **5 个主要模块**组成：时间嵌入 → 编码器 → 中间层（+注意力） → 解码器 → 输出
2. **时间嵌入**在每个残差块中注入，告诉网络当前时间步
3. **跳跃连接**在解码器中拼接编码器的特征，保住细节
4. 使用 **GroupNorm** 而不是 BatchNorm（更稳定）
5. 总共约 **360 万参数**，在 CPU 上也能快速训练

---

**继续阅读**：[[12_训练循环与采样流程]]
