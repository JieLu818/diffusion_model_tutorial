"""
超参数配置文件
所有关键参数集中管理，方便实验时调参对比
"""

class DiffusionConfig:
    """扩散模型的默认超参数"""

    # ========== 数据相关 ==========
    image_size     = 28          # MNIST 图像尺寸 28×28
    image_channels = 1           # 灰度图

    # ========== 扩散过程 ==========
    T = 1000                     # 总步数（从干净到纯噪声分 1000 步走完）
    beta_start    = 1e-4         # 起始噪声率（很小，第一步几乎看不出来噪声）
    beta_end      = 0.02         # 终止噪声率（最后一步接近纯噪声）
    beta_schedule = "linear"     # 调度方式：linear / cosine

    # ========== 模型结构 ==========
    base_channels  = 128         # U-Net 基础通道数（每层翻倍：128→256→512）
    time_emb_dim   = 256         # 时间嵌入的维度
    num_res_blocks = 2           # 每个分辨率层有几个残差块

    # ========== 训练 ==========
    batch_size     = 128         # 批次大小
    epochs         = 50          # 训练轮数
    lr             = 1e-3        # 学习率
    num_workers    = 0           # 数据加载线程（Mac 上设为 0 更稳定）

    # ========== 采样 ==========
    sample_batch_size = 16       # 一次生成多少张图
    sample_steps      = 1000     # 生成时的步数（<=T，可小于训练步数加速）
