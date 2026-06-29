# diffusion_model_tutorial
Starter of diffusion model.


##文件结构

```

diffusion/
   ├── 00_导读.md
   ├── 第一部分：直觉与概念/
   │   ├── 01_从一碗墨水说起.md         ← 核心直觉类比 + 全流程鸟瞰 Mermaid 图
   │   ├── 02_前向过程_加噪.md           ← Forward process + 加噪示意图
   │   └── 03_反向过程_去噪.md           ← Reverse / denoising + 去噪示意图
   ├── 第二部分：数学框架/
   │   ├── 04_数学预备_高斯分布与马尔可夫链.md   ← 必备复习 + 重参数化直觉
   │   ├── 05_变分下界_ELBO.md                    ← 从「为什么不是直接似然」引出 ELBO
   │   └── 06_核心推导_从ELBO到MSE.md            ← 主要推导，每步配大白话
   ├── 第三部分：模型架构/
   │   ├── 07_U_Net架构.md               ← Mermaid 结构图 + 各组件说明
   │   └── 08_时间嵌入与注意力机制.md     ← Positional encoding + attention
   ├── 第四部分：动手实现/
   │   ├── 09_项目结构与噪声调度器.md     ← 项目骨架 + beta schedule + 调度器图示
   │   ├── 10_前向加噪与数据集准备.md     ← Forward process 实现 + Dataset 类
   │   ├── 11_UNet代码实现.md             ← U-Net PyTorch 代码 + 逐段讲解
   │   └── 12_训练循环与采样流程.md       ← Training loop + sampling algorithm
   ├── 第五部分：实验/
   │   ├── 13_MNIST上跑第一个扩散模型.md  ← 端到端训练日志 + 损失曲线图
   │   ├── 14_生成效果可视化.md           ← 逐渐生成的 GIF / 多步快照
   │   └── 15_超参数探索.md               ← T步数、beta schedule、学习率对比
   ├── 第六部分：进阶/
   │   ├── 16_条件扩散模型.md             ← Class-conditioned diffusion
   │   ├── 17_DDIM加速采样.md             ← DDIM 原理 + 代码差异
   │   └── 18_总结与阅读地图.md           ← 重要论文、资源、下一站建议
   ├── code/
   │   ├── requirements.txt
   │   ├── scheduler.py              ← BetaSchedule + 前向/后向系数计算
   │   ├── model.py                  ← U-Net + time embedding + attention
   │   ├── diffusion.py              ← Diffusion pipeline (train_step + sample)
   │   ├── train_mnist.py            ← MNIST 训练主脚本
   │   ├── sample_and_visualize.py   ← 采样 + 可视化生成结果
   │   └── config.py                 ← 超参数配置
   └── scripts/
       ├── gen_forward_diagram.py    ← 生成前向加噪过程配图
       └── gen_loss_curve.py         ← 从训练日志绘制损失曲线

```




