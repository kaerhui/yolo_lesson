"""
Day 4 - 感受野 (Receptive Field) 理论与分析
=============================================
实现感受野的逐层计算，分析 CNN 各层的理论感受野，
并分析 YOLO 检测头对应的感受野范围。

Author: YOLO Lesson Week1
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


# ============================================================
# 1. 感受野逐层计算器
# ============================================================

class ReceptiveFieldCalculator:
    """
    逐层计算 CNN 的理论感受野。

    核心公式:
        r_out = r_in + (k_eff - 1) * j_in
        j_out = j_in * s
        start_out = start_in + (k_eff - 1) / 2 * j_in  (以中心为参考)

    其中:
        r: 感受野大小
        j: 相邻像素在输入图上的距离 (jump/stride)
        k_eff: 有效卷积核大小 (考虑空洞率)
        s: 步长
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, name: str, kernel_size: int, stride: int = 1, dilation: int = 1):
        """
        添加一个层到计算序列。

        Args:
            name: 层名称
            kernel_size: 卷积核大小
            stride: 步长
            dilation: 空洞率
        """
        effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
        self.layers.append({
            'name': name,
            'k': kernel_size,
            's': stride,
            'd': dilation,
            'k_eff': effective_kernel,
        })

    def compute(self, input_size: int = None):
        """
        从前往后迭代计算每层的感受野。

        Args:
            input_size: 输入特征图大小 (可选)

        Returns:
            list of dict: 每层的感受野信息
        """
        results = []
        r = 1  # 初始感受野
        j = 1  # 初始 jump
        start = 0.5  # 初始中心偏移 (假设第一层像素中心在 0.5)

        if input_size is not None:
            feature_map_size = input_size

        for layer in self.layers:
            k_eff = layer['k_eff']
            s = layer['s']

            # 更新感受野: r_out = r_in + (k_eff - 1) * j_in
            r_out = r + (k_eff - 1) * j
            # 更新 jump: j_out = j_in * s
            j_out = j * s

            if input_size is not None:
                # 特征图大小: out = floor((in - k_eff) / s) + 1
                # 简化: 使用有效卷积核
                feature_map_size = (feature_map_size - k_eff) // s + 1

            results.append({
                'layer': layer['name'],
                'kernel': layer['k'],
                'stride': s,
                'dilation': layer['d'],
                'effective_kernel': k_eff,
                'receptive_field': r_out,
                'jump': j_out,
                'feature_map_size': feature_map_size if input_size is not None else 'N/A',
            })

            r = r_out
            j = j_out

        return results

    def print_table(self, input_size: int = None):
        """打印感受野计算表格。"""
        results = self.compute(input_size)

        print(f"\n{'Layer':<20} {'Kernel':<8} {'Stride':<8} {'Dilation':<10} {'Eff.Kernel':<12} {'RF':<10} {'Jump':<8}"
              + (f" {'FM Size':<10}" if input_size else ""))
        print("=" * (90 + (12 if input_size else 0)))

        for r in results:
            fm = str(r['feature_map_size']) if input_size else ''
            print(f"{r['layer']:<20} {r['kernel']:<8} {r['stride']:<8} {r['dilation']:<10} "
                  f"{r['effective_kernel']:<12} {r['receptive_field']:<10} {r['jump']:<8}"
                  + (f" {fm:<10}" if input_size else ""))

        return results


# ============================================================
# 2. 构建示例 CNN 并分析
# ============================================================

def analyze_sample_cnn():
    """分析一个示例 CNN 的感受野。"""
    calc = ReceptiveFieldCalculator()

    # 模拟一个简单的 CNN 结构
    calc.add_layer('Conv1', kernel_size=3, stride=1)
    calc.add_layer('Pool1', kernel_size=2, stride=2)  # 池化层
    calc.add_layer('Conv2', kernel_size=3, stride=1)
    calc.add_layer('Conv3', kernel_size=3, stride=2)
    calc.add_layer('Conv4', kernel_size=3, stride=1, dilation=2)  # 空洞卷积

    print("=" * 60)
    print("示例 CNN 感受野分析 (输入: 224x224)")
    print("=" * 60)
    results = calc.print_table(input_size=224)

    return results


# ============================================================
# 3. YOLO 检测头感受野分析
# ============================================================

def analyze_yolo_receptive_field():
    """
    分析 YOLOv5s/YOLOv8n 三个检测头的感受野。

    YOLO 的 Neck 部分使用 FPN+PAN 结构，三个检测头分别位于
    P3/8 (小目标), P4/16 (中目标), P5/32 (大目标) 特征图上。

    这里模拟从输入到各检测头的路径。
    """
    print("\n" + "=" * 60)
    print("YOLO 检测头感受野分析")
    print("=" * 60)

    # YOLO 的 Backbone + Neck 简化结构
    # 实际 YOLO 使用 CSPDarknet + FPN+PAN，这里用简化模型
    # 各阶段的特征图下采样倍数: 2, 4, 8, 16, 32

    # 检测头 1: P3/8 (小目标检测)
    calc_p3 = ReceptiveFieldCalculator()
    # Backbone 前几层
    calc_p3.add_layer('Focus/Conv', kernel_size=6, stride=2)  # 下采样到 1/2
    calc_p3.add_layer('Conv1', kernel_size=3, stride=2)       # 下采样到 1/4
    calc_p3.add_layer('CSP1', kernel_size=3, stride=1)
    calc_p3.add_layer('Conv2', kernel_size=3, stride=2)       # 下采样到 1/8
    calc_p3.add_layer('CSP2', kernel_size=3, stride=1)
    # 加上 Neck 的上采样和融合 (简化)
    calc_p3.add_layer('Neck_Conv', kernel_size=3, stride=1)

    print("\n--- P3/8 检测头 (小目标) ---")
    p3_results = calc_p3.compute()
    for r in p3_results:
        print(f"  {r['layer']:<15} RF={r['receptive_field']:<6} Jump={r['jump']}")

    # 检测头 2: P4/16 (中目标检测)
    calc_p4 = ReceptiveFieldCalculator()
    calc_p4.add_layer('Focus/Conv', kernel_size=6, stride=2)
    calc_p4.add_layer('Conv1', kernel_size=3, stride=2)
    calc_p4.add_layer('CSP1', kernel_size=3, stride=1)
    calc_p4.add_layer('Conv2', kernel_size=3, stride=2)
    calc_p4.add_layer('CSP2', kernel_size=3, stride=1)
    calc_p4.add_layer('Conv3', kernel_size=3, stride=2)       # 下采样到 1/16
    calc_p4.add_layer('CSP3', kernel_size=3, stride=1)
    calc_p4.add_layer('Neck_Conv', kernel_size=3, stride=1)

    print("\n--- P4/16 检测头 (中目标) ---")
    p4_results = calc_p4.compute()
    for r in p4_results:
        print(f"  {r['layer']:<15} RF={r['receptive_field']:<6} Jump={r['jump']}")

    # 检测头 3: P5/32 (大目标检测)
    calc_p5 = ReceptiveFieldCalculator()
    calc_p5.add_layer('Focus/Conv', kernel_size=6, stride=2)
    calc_p5.add_layer('Conv1', kernel_size=3, stride=2)
    calc_p5.add_layer('CSP1', kernel_size=3, stride=1)
    calc_p5.add_layer('Conv2', kernel_size=3, stride=2)
    calc_p5.add_layer('CSP2', kernel_size=3, stride=1)
    calc_p5.add_layer('Conv3', kernel_size=3, stride=2)
    calc_p5.add_layer('CSP3', kernel_size=3, stride=1)
    calc_p5.add_layer('Conv4', kernel_size=3, stride=2)       # 下采样到 1/32
    calc_p5.add_layer('CSP4', kernel_size=3, stride=1)
    calc_p5.add_layer('Neck_Conv', kernel_size=3, stride=1)

    print("\n--- P5/32 检测头 (大目标) ---")
    p5_results = calc_p5.compute()
    for r in p5_results:
        print(f"  {r['layer']:<15} RF={r['receptive_field']:<6} Jump={r['jump']}")

    summary = [
        ("P3/8 (小目标)", p3_results[-1]['receptive_field'], p3_results[-1]['jump'], "小", "小"),
        ("P4/16 (中目标)", p4_results[-1]['receptive_field'], p4_results[-1]['jump'], "中", "中"),
        ("P5/32 (大目标)", p5_results[-1]['receptive_field'], p5_results[-1]['jump'], "大", "大"),
    ]

    print("\n" + "=" * 60)
    print(f"{'检测头':<18} {'感受野':<10} {'Jump':<8} {'特征图':<8} {'负责目标':<8}")
    print("=" * 60)
    for name, rf, jump, fm_size, target in summary:
        print(f"{name:<18} {rf:<10} {jump:<8} {fm_size:<8} {target:<8}")
    print("=" * 60)
    print("\n结论: 小特征图 (P5/32) 感受野大 → 负责大目标")
    print("      大特征图 (P3/8) 感受野小 → 负责小目标")
    print("      这就是 FPN 多尺度特征融合的核心动机。")

    return summary


# ============================================================
# 4. 有效感受野 (ERF) 可视化
# ============================================================

def visualize_effective_receptive_field():
    """
    通过 PyTorch 构建一个小 CNN，可视化有效感受野。
    方法: 在输出层中心设置梯度为 1，反向传播到输入层，观察梯度分布。
    """
    import matplotlib.pyplot as plt

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1, 8, 3, padding=1)),
                ('relu1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(2)),
                ('conv2', nn.Conv2d(8, 16, 3, padding=1)),
                ('relu2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(2)),
                ('conv3', nn.Conv2d(16, 1, 3, padding=1)),
            ]))

        def forward(self, x):
            return self.features(x)

    model = SimpleCNN()
    model.eval()

    # 创建一个较大的输入，使输出层至少有一个像素
    input_size = 64
    x = torch.randn(1, 1, input_size, input_size, requires_grad=True)
    out = model(x)

    # 在输出中心设置梯度为 1
    model.zero_grad()
    out[0, 0, 0, 0].backward()

    # 获取输入梯度（即有效感受野的近似）
    grad = x.grad[0, 0].abs().detach().numpy()

    # 归一化
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-7)

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(grad, cmap='hot', interpolation='bilinear')
    plt.colorbar(label='Gradient Magnitude')
    plt.title('Effective Receptive Field (ERF)')
    plt.xlabel('Input Width')
    plt.ylabel('Input Height')

    plt.subplot(1, 2, 2)
    # 取中心行的一维剖面
    center_row = grad[input_size // 2, :]
    plt.plot(center_row, 'b-', linewidth=2)
    plt.xlabel('Input Pixel')
    plt.ylabel('Gradient Magnitude')
    plt.title('ERF Cross-section (Center Row)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('effective_receptive_field.png', dpi=150)
    plt.show()

    print(f"\n有效感受野已保存至 effective_receptive_field.png")
    print(f"理论感受野: {3 + (2-1)*1 + (2-1)*1 + (3-1)*2 + (2-1)*2 + (3-1)*4}")  # 简化计算

    return grad


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Day 4 - 感受野 (Receptive Field) 理论与分析")
    print("=" * 80)

    # 1. 示例 CNN 感受野分析
    analyze_sample_cnn()

    # 2. YOLO 检测头感受野分析
    analyze_yolo_receptive_field()

    # 3. 有效感受野可视化
    print("\n" + "=" * 60)
    print("有效感受野 (ERF) 可视化")
    print("=" * 60)
    print("正在通过反向传播梯度法可视化有效感受野...")
    grad = visualize_effective_receptive_field()

    print("\n" + "=" * 60)
    print("关键概念总结")
    print("=" * 60)
    print("""
理论感受野 vs 有效感受野:
    - 理论感受野: 通过公式计算的"最大"可感知区域
    - 有效感受野: 实际起作用的区域，中心呈高斯分布
    - 有效感受野通常只有理论感受野的 1/3 ~ 1/2

感受野与检测任务的关系:
    - 大目标需要大感受野 → 深层的下采样特征图 (P5/32)
    - 小目标需要高分辨率特征图 → 浅层的特征图 (P3/8)
    - 多尺度特征融合 (FPN/PAN) 就是为了同时兼顾大小目标

空洞卷积的作用:
    - 在不增加参数量的情况下指数级扩大感受野
    - 有效核尺寸: k_eff = k + (k-1)*(d-1)
    """)