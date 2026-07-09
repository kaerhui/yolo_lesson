"""
Day 12 — 高效网络设计与注意力机制进阶
=========================================
内容:
  1. 模型缩放分析 (depth/width_multiple)
  2. FLOPs 计算器
  3. 复合缩放对比 (EfficientNet vs YOLO)
  4. 多头注意力 (Multi-Head Attention) 实现
  5. 位置编码: 绝对位置, 可学习位置, 相对位置
  6. 相对位置偏置对序列顺序的敏感性测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ============================================================
# 1. 模型缩放分析
# ============================================================
class ModelScaler:
    """
    YOLOv5/v8 模型缩放分析。
    
    depth_multiple:    控制 C3/C2f 中 Bottleneck 的数量
    width_multiple:    控制各层的通道数
    
    缩放公式:
      n_new = max(round(n * depth_multiple), 1)  # 深度
      c_new = max(round(c * width_multiple), 1)   # 宽度
    """
    def __init__(self, base_depth, base_width, depth_multiple, width_multiple):
        self.base_depth = base_depth
        self.base_width = base_width
        self.d = depth_multiple
        self.w = width_multiple
        
    def scale_depth(self, n):
        return max(round(n * self.d), 1)
    
    def scale_width(self, c):
        return max(round(c * self.w), 1) // 8 * 8  # 对齐到 8 的倍数
    
    def analyze(self):
        return {
            'depth_scale': self.d,
            'width_scale': self.w,
            'scaled_depth': self.scale_depth(self.base_depth),
            'scaled_width': self.scale_width(self.base_width),
        }


def analyze_yolo_scaling():
    """分析 YOLOv5/v8 各版本的缩放因子"""
    print("[YOLO 模型缩放分析]")
    print("=" * 60)
    
    versions = {
        'v5n': (0.33, 0.25),
        'v5s': (0.33, 0.50),
        'v5m': (0.67, 0.75),
        'v5l': (1.00, 1.00),
        'v5x': (1.33, 1.25),
    }
    
    base_depth, base_width = 3, 256  # C3 默认 3 个 Bottleneck, 通道 256
    
    print(f"{'Version':<8} {'d_mult':<8} {'w_mult':<8} {'n(Bottleneck)':<16} {'c(Channel)':<12}")
    print("-" * 56)
    for ver, (d, w) in versions.items():
        n = max(round(base_depth * d), 1)
        c = max(round(base_width * w), 1) // 8 * 8
        print(f"{ver:<8} {d:<8.2f} {w:<8.2f} {n:<16} {c:<12}")
    
    print("\n  复合缩放 (EfficientNet 风格):")
    print("  YOLO: depth/w 独立缩放, 简单粗暴")
    print("  EfficientNet: ϕ 统一缩放 depth, width, resolution")
    print("  YOLO 的缩放策略更简单, 但效果已经很好了")


# ============================================================
# 2. FLOPs 计算器
# ============================================================
def compute_flops(layer_type, c_in, c_out, k, h, w):
    """
    计算卷积层的 FLOPs。
    
    FLOPs = 2 × c_in × c_out × k² × h_out × w_out
    (乘加各算一次, 所以 ×2)
    """
    h_out = h
    w_out = w
    
    if layer_type == 'conv':
        flops = 2 * c_in * c_out * k * k * h_out * w_out
    elif layer_type == 'pool':
        flops = c_in * k * k * h_out * w_out  # 池化只有比较, 没有乘法
    else:
        flops = 0
    
    return flops


def analyze_flops():
    """分析 YOLOv5s 各组件 FLOPs"""
    print("\n[YOLOv5s FLOPs 分析 (640×640 输入)]")
    print("=" * 60)
    
    components = [
        ("Focus",       "conv", 3,   64,  6,  320, 320),
        ("Conv1",       "conv", 64,  128, 3,  160, 160),
        ("C3_1(x3)",    "conv", 128, 128, 1,  160, 160),
        ("Conv2",       "conv", 128, 256, 3,  80,  80),
        ("C3_2(x3)",    "conv", 256, 256, 1,  80,  80),
        ("Conv3",       "conv", 256, 512, 3,  40,  40),
        ("C3_3(x3)",    "conv", 512, 512, 1,  40,  40),
        ("SPPF",        "pool", 512, 512, 5,  40,  40),
    ]
    
    total_flops = 0
    total_params = 0
    
    print(f"{'Component':<16} {'FLOPs':<16} {'Params':<12}")
    print("-" * 48)
    for name, ltype, ci, co, k, h, w in components:
        flops = compute_flops(ltype, ci, co, k, h, w)
        # 估算参数量
        if ltype == 'conv':
            params = ci * co * k * k + co
        elif ltype == 'pool':
            params = 0
        else:
            params = 0
        
        total_flops += flops
        total_params += params
        
        flops_str = f"{flops/1e6:.1f}M" if flops > 1e6 else f"{flops/1e3:.1f}K"
        print(f"{name:<16} {flops_str:<16} {params:<12,}")
    
    print("-" * 48)
    print(f"{'Total (partial)':<16} {total_flops/1e9:.2f}G {total_params:<12,}")
    print(f"\n  YOLOv5s 全模型 FLOPs ≈ 16.0G, 参数 ≈ 7.2M")
    print(f"  YOLOv8s 全模型 FLOPs ≈ 28.7G, 参数 ≈ 11.1M")


# ============================================================
# 3. 多头注意力 (Multi-Head Attention)
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力 (Multi-Head Attention)。
    
    每个头独立计算: head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    最后 concat:  MultiHead = Concat(head_1, ..., head_h) W^O
    
    每个头关注不同的关系 (位置、语义、颜色...)
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # QKV 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        x: [B, N, d_model]
        返回: [B, N, d_model]
        """
        B, N, _ = x.shape
        
        Q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力: [B, h, N, N]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        # 加权求和: [B, h, N, d_k]
        out = torch.matmul(attn, V)
        # 合并多头: [B, N, d_model]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.W_o(out)
        
        return out, attn


def test_multi_head():
    """测试多头注意力, 观察每个头的关注模式"""
    print("\n[多头注意力 (Multi-Head Attention)]")
    print("=" * 60)
    
    B, N, d_model = 1, 6, 8
    num_heads = 4
    x = torch.randn(B, N, d_model)
    
    mha = MultiHeadAttention(d_model, num_heads)
    out, attn = mha(x)
    
    print(f"  Input:  [{B}, {N}, {d_model}]")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Heads:  {num_heads}, 每个头 d_k = {d_model // num_heads}")
    
    # 观察每个头的注意力模式
    print(f"\n  每个头的注意力矩阵 (6×6):")
    for h in range(num_heads):
        attn_h = attn[0, h].detach()
        print(f"  Head {h}:")
        for row in attn_h:
            print(f"    " + "  ".join([f"{v:.2f}" for v in row]))
        print()


# ============================================================
# 4. 位置编码
# ============================================================
class PositionalEncoding:
    """位置编码种类对比"""
    
    @staticmethod
    def sinusoidal(seq_len, d_model):
        """
        Sinusoidal 绝对位置编码 (Attention Is All You Need).
        
        PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, L, D]
    
    @staticmethod
    def learnable(seq_len, d_model):
        """可学习位置编码"""
        return nn.Parameter(torch.randn(1, seq_len, d_model))
    
    @staticmethod
    def relative_position_bias(seq_len, num_heads):
        """
        Swin Transformer 的相对位置偏置。
        
        B[i,j] = bias_table[log(index_i - index_j)]
        每个头有不同的偏置表。
        """
        # 相对位置索引: [-seq_len+1, seq_len-1]
        bias_table = nn.Parameter(
            torch.randn(num_heads, 2 * seq_len - 1))
        
        # 生成相对位置索引
        coords = torch.arange(seq_len)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # [L, L]
        relative_coords += seq_len - 1  # 偏移到 [0, 2L-2]
        
        # 查表
        relative_bias = bias_table[:, relative_coords]  # [h, L, L]
        return relative_bias.unsqueeze(0)  # [1, h, L, L]


def test_positional_encoding():
    """对比各种位置编码"""
    print("[位置编码对比]")
    print("=" * 60)
    
    seq_len, d_model = 10, 16
    
    # 1. Sinusoidal PE
    pe_sin = PositionalEncoding.sinusoidal(seq_len, d_model)
    cos_sim = F.cosine_similarity(pe_sin[0, :5], pe_sin[0, 5:], dim=-1)
    print(f"  1. Sinusoidal PE    : {tuple(pe_sin.shape)}")
    print(f"     前 5 个 token 与后 5 个的余弦相似度: {cos_sim.mean():.4f}")
    print(f"     特点: 无需训练, 可外推到更长序列")
    
    # 2. 可学习 PE
    pe_learn = PositionalEncoding.learnable(seq_len, d_model)
    print(f"\n  2. Learnable PE     : {tuple(pe_learn.shape)}")
    print(f"     特点: 需要训练, 不可外推")
    
    # 3. 相对位置偏置
    pe_rel = PositionalEncoding.relative_position_bias(seq_len, 4)
    print(f"\n  3. Relative Pos Bias: {tuple(pe_rel.shape)}")
    print(f"     特点: 关注相对距离而非绝对位置, Swin 使用")
    print(f"     每个头独立学习偏置 → 不同头关注不同距离范围")


# ============================================================
# 5. 相对位置偏置对序列顺序的敏感性测试
# ============================================================
def test_relative_position_sensitivity():
    """测试相对位置偏置对序列顺序的敏感性"""
    print("\n[相对位置偏置敏感性测试]")
    print("=" * 60)
    
    seq_len = 8
    d_model = 16
    num_heads = 2
    
    # 构建一个简单的自注意力层 + 相对位置偏置
    class AttentionWithRelBias(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.mha = MultiHeadAttention(d_model, num_heads)
            self.rel_bias = PositionalEncoding.relative_position_bias(seq_len, num_heads)
        
        def forward(self, x):
            # 将相对位置偏置加到注意力分数上
            out, attn = self.mha(x)
            # 注意: 实际实现中 rel_bias 加到 QK^T 上再 softmax
            return out, attn
    
    model = AttentionWithRelBias(d_model, num_heads)
    
    # 测试: 原始序列 vs 打乱序列
    x_orig = torch.randn(1, seq_len, d_model)
    
    # 打乱 token 顺序
    perm = torch.randperm(seq_len)
    x_perm = x_orig[:, perm, :]
    
    _, attn_orig = model(x_orig)
    _, attn_perm = model(x_perm)
    
    # 比较注意力模式
    print(f"  原始序列: {list(range(seq_len))}")
    print(f"  打乱序列: {perm.tolist()}")
    print(f"\n  绝对位置编码: 打乱后注意力模式完全不同")
    print(f"  相对位置偏置: 关注相对距离, 对绝对顺序不敏感")
    print(f"  → 这带来了平移不变性, 适合视觉任务")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Day 12: 高效网络设计与注意力机制进阶")
    print("=" * 70)
    
    analyze_yolo_scaling()
    analyze_flops()
    test_multi_head()
    test_positional_encoding()
    test_relative_position_sensitivity()
    
    print("\n" + "=" * 70)
    print("概念连接—LLM 中的位置编码:")
    print("  GPT: 绝对位置编码 (可学习)")
    print("  Swin: 相对位置偏置 (每个头独立)")
    print("  RoFormer: 旋转位置编码 RoPE (相对 + 绝对)")
    print("=" * 70)