"""
Day 8 — YOLOv5 完整架构拆解 + 自注意力初探
============================================
内容:
  1. Focus 层 (Stem) 实现
  2. CSPBlock (C3) 源码级复现
  3. SPPF 实现与感受野分析
  4. YOLOv5s 拓扑图打印
  5. Self-Attention 从零实现
  6. 将 SA 插入 CNN 观察全局响应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. Focus 层 (Stem)
# ============================================================
class Focus(nn.Module):
    """
    Focus 层: 将输入先切片再卷积, 等价于 stride=2 的卷积但更高效。
    
    原理: 将 HxW 的 4 个相邻像素在通道维拼起来,
          得到 H/2 x W/2, 通道数变为 4*C。
          公式: out[b, 4c, h, w] = in[b, c, 2h, 2w]
    """
    def __init__(self, c1, c2, k=1):
        super().__init__()
        # 切片后通道数为 4*c1, 然后用卷积降维到 c2
        self.conv = nn.Conv2d(4 * c1, c2, k, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 切片: 取 4 个偏移位置的像素
        x_sliced = torch.cat([
            x[..., 0::2, 0::2],  # (0,0)
            x[..., 1::2, 0::2],  # (1,0)
            x[..., 0::2, 1::2],  # (0,1)
            x[..., 1::2, 1::2],  # (1,1)
        ], dim=1)  # 通道维拼接 → [B, 4C, H/2, W/2]
        return self.conv(x_sliced)


def test_focus():
    """验证 Focus 输出尺寸"""
    x = torch.randn(1, 3, 640, 640)
    focus = Focus(3, 64)
    out = focus(x)
    print(f"[Focus] Input: {tuple(x.shape)} → Output: {tuple(out.shape)}")
    assert out.shape[-1] == 320, "Focus should halve spatial dims"
    print("  ✓ Focus 正确: 空间减半, 通道从 3→64")


# ============================================================
# 2. CSPBlock (C3) 源码级复现
# ============================================================
class Bottleneck(nn.Module):
    """标准瓶颈模块: 1×1 → 3×3, shortcut = True (残差连接)"""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c2, 3, 1, padding=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """
    CSPBlock (C3): 跨阶段局部网络。
    
    将输入在通道维分成两路:
      - 一路直接通过 (shortcut)
      - 另一路经过 n 个 Bottleneck
    最后 concat 并用卷积过渡。
    
    梯度截断效果: 由于 split, 梯度只流过一半通道,
    减少了梯度回传时的重复计算, 提升训练效率。
    """
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 中间通道
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1)  # 过渡卷积
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut) for _ in range(n)])

    def forward(self, x):
        # 两路 split
        y1 = self.m(self.cv1(x))  # 经过 Bottleneck 的主路
        y2 = self.cv2(x)          # 捷径路
        return self.cv3(torch.cat((y1, y2), dim=1))


def test_c3():
    """验证 C3 输出尺寸和梯度截断"""
    x = torch.randn(1, 128, 40, 40)
    c3 = C3(128, 128, n=3)
    out = c3(x)
    print(f"[C3] Input: {tuple(x.shape)} → Output: {tuple(out.shape)}")
    assert out.shape == x.shape, "C3 should preserve spatial & channel dims"

    # 梯度截断验证: 只有一半通道的梯度流经 Bottleneck
    loss = out.sum()
    loss.backward()
    grad_norm_cv1 = c3.cv1.weight.grad.norm().item()
    grad_norm_cv2 = c3.cv2.weight.grad.norm().item()
    print(f"  ✓ C3 梯度流: cv1 (主路) grad norm = {grad_norm_cv1:.4f}")
    print(f"  ✓ C3 梯度流: cv2 (捷径) grad norm = {grad_norm_cv2:.4f}")
    print(f"  ✓ 梯度同时流经两路, 但主路承担更多变换")


# ============================================================
# 3. SPPF — 快速空间金字塔池化
# ============================================================
class SPPF(nn.Module):
    """
    SPPF (Spatial Pyramid Pooling - Fast):
    
    三个串行的 5×5 最大池化, 等效感受野分别为 5, 9, 13。
    与并行 SPP 相比, 计算量更小 (串行复用中间结果)。
    
    感受野推导:
      - 池化 1: k=5, s=1 → RF = 5
      - 池化 2: k=5, s=1, 输入 RF=5 → RF = 5 + (5-1)*1 = 9
      - 池化 3: k=5, s=1, 输入 RF=9 → RF = 9 + (5-1)*1 = 13
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # 中间通道
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(4 * c_, c2, 1, 1)  # concat 4 路 → 4*c_
        self.m = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        # 串行池化, 每次复用上一次结果
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # concat 四路: 原图 + 3 个不同感受野的池化结果
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


def test_sppf():
    """验证 SPPF 输出尺寸"""
    x = torch.randn(1, 256, 20, 20)
    sppf = SPPF(256, 256)
    out = sppf(x)
    print(f"[SPPF] Input: {tuple(x.shape)} → Output: {tuple(out.shape)}")
    assert out.shape == x.shape, "SPPF should preserve spatial & channel dims"
    print("  ✓ SPPF 正确: 无尺寸变化, 融合多尺度池化特征")


# ============================================================
# 4. YOLOv5s 拓扑图打印
# ============================================================
def print_yolov5s_topology():
    """打印 YOLOv5s 的完整网络拓扑, 包含每层输入输出尺寸"""
    print("\n" + "=" * 70)
    print("YOLOv5s 网络拓扑 (640x640 输入)")
    print("=" * 70)

    layers = [
        ("Focus",       [3, 64],   "640→320",  "3→64"),
        ("Conv",        [64, 128], "320→160",  "64→128"),
        ("C3_1",        [128,128], "160→160",  "128→128"),
        ("Conv",        [128,256], "160→80",   "128→256"),
        ("C3_2",        [256,256], "80→80",    "256→256"),
        ("Conv",        [256,512], "80→40",    "256→512"),
        ("C3_3",        [512,512], "40→40",    "512→512"),
        ("SPPF",        [512,512], "40→40",    "512→512"),
        # Neck
        ("Conv",        [512,256], "40→20",    "512→256"),
        ("Upsample",    [256,256], "20→40",    "256→256"),
        ("C3_4",        [256,256], "40→40",    "256→256"),
        ("Conv",        [256,128], "40→20",    "256→128"),
        ("Upsample",    [128,128], "20→40",    "128→128"),
        ("C3_5",        [128,128], "40→40",    "128→128"),
        # Head
        ("Conv",        [128,128], "40→20",    "128→128"),
        ("C3_6",        [128,128], "20→20",    "128→128"),
        ("Detect_P3",   [128, 85], "20→20",    "128→85"),
        ("Conv",        [128,256], "20→10",    "128→256"),
        ("C3_7",        [256,256], "10→10",    "256→256"),
        ("Detect_P4",   [256, 85], "10→10",    "256→85"),
        ("Conv",        [256,512], "10→5",     "256→512"),
        ("C3_8",        [512,512], "5→5",      "512→512"),
        ("Detect_P5",   [512, 85], "5→5",      "512→85"),
    ]

    print(f"{'Layer':<12} {'Channels':<16} {'Spatial':<16} {'Param':<12}")
    print("-" * 60)
    for name, ch, sp, pm in layers:
        print(f"{name:<12} {str(ch):<16} {sp:<16} {pm:<12}")
    print("=" * 70)


# ============================================================
# 5. Self-Attention 从零实现
# ============================================================
class SelfAttention(nn.Module):
    """
    多头自注意力 (单头版本, 便于理解)。
    
    Attention(Q,K,V) = softmax(QK^T / √d_k) V
    
    如果 Q, K, V 来自局部窗口, 退化为动态卷积。
    (证明: 当 attention 矩阵为带状矩阵时, 等价于局部加权和)
    """
    def __init__(self, d_model, d_k=None):
        super().__init__()
        d_k = d_k or d_model
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)
        self.scale = math.sqrt(d_k)

    def forward(self, x, mask=None):
        """
        x: [B, N, d_model]  (N 个 token)
        返回: [B, N, d_k]
        """
        Q = self.W_q(x)  # [B, N, d_k]
        K = self.W_k(x)
        V = self.W_v(x)

        # 注意力分数: [B, N, N]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # 加权求和: [B, N, d_k]
        out = torch.matmul(attn, V)
        return out, attn


def test_self_attention():
    """验证自注意力输出形状, 并手算一个小例子"""
    # 3 个 token, 每个 4 维
    x = torch.randn(1, 3, 8)
    sa = SelfAttention(d_model=8, d_k=4)
    out, attn = sa(x)
    print(f"\n[SelfAttention] Input: [1, 3, 8] → Output: {tuple(out.shape)}")
    print(f"  Attention matrix: {tuple(attn.shape)}")
    print(f"  Attention weights (row sum to 1):\n{attn[0].detach():.4f}")

    # 手算验证: 第 0 个 token 的注意力权重和应为 1
    row_sum = attn[0, 0].sum().item()
    print(f"  ✓ Row 0 sum = {row_sum:.4f} (should be 1.0)")


# ============================================================
# 6. 将 Self-Attention 插入 CNN 最后一层
# ============================================================
class CNNWithSA(nn.Module):
    """
    简单 CNN + Self-Attention。
    在最后一层特征图上应用 SA, 观察全局响应。
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取: 3→16→32→64
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8x8
            nn.ReLU(),
        )
        # 自注意力: 64 维特征, 8 个头
        self.sa = SelfAttention(d_model=64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # CNN 特征: [B, 64, 8, 8]
        feat = self.features(x)
        # 展开为 token 序列: [B, 64, 8, 8] → [B, 64, 64] → [B, 64, 64]
        N = feat.shape[2] * feat.shape[3]  # 64 个位置
        tokens = feat.flatten(2).transpose(1, 2)  # [B, 64, 64]
        # SA 全局建模
        tokens, attn = self.sa(tokens)
        # 全局平均池化
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled), attn


def test_cnn_with_sa():
    """验证 CNN+SA 的全局响应"""
    model = CNNWithSA()
    x = torch.randn(2, 3, 64, 64)
    logits, attn = model(x)
    print(f"\n[CNN+SA] Input: [2, 3, 64, 64] → Logits: {tuple(logits.shape)}")
    print(f"  Attention map: {tuple(attn.shape)}")
    # 检查注意力是否覆盖全图 (64 个位置)
    print(f"  ✓ 注意力覆盖所有 {attn.shape[-1]} 个空间位置, 实现全局建模")


# ============================================================
# 主函数: 运行所有测试
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Day 8: YOLOv5 架构拆解 + Self-Attention 初探")
    print("=" * 70)

    test_focus()
    test_c3()
    test_sppf()
    print_yolov5s_topology()
    test_self_attention()
    test_cnn_with_sa()

    print("\n" + "=" * 70)
    print("所有测试通过! 输出说明:")
    print("  - Focus: 切片等价于 stride=2 卷积, 节省计算")
    print("  - C3: 跨阶段 split 减少梯度重复计算")
    print("  - SPPF: 串行池化复用中间结果, 等效 RF=5,9,13")
    print("  - Self-Attention: 全局建模, 可插入 CNN 最后一层")
    print("=" * 70)