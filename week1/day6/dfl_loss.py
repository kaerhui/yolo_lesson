"""
Day 6 - 深度剖析 DFL (Distribution Focal Loss)
===============================================
从零实现 Distribution Focal Loss，理解 YOLOv8 回归分支的核心创新。

DFL 将框的每一条边建模为离散概率分布，而非传统 Dirac Delta 单点估计。

Author: YOLO Lesson Week1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ============================================================
# 1. DFL Loss 从零实现
# ============================================================

class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL)

    核心思想:
    - 将框的每一条边（左、上、右、下）建模为 reg_max+1 个离散值上的概率分布
    - 目标坐标的连续值被映射到附近的两个离散点上
    - 损失函数鼓励网络在目标值附近的两个整数上产生高概率

    公式:
        L_DFL = -((yi+1 - y) * log(S_i) + (y - yi) * log(S_{i+1}))

    其中 y 是连续目标值, yi = floor(y), yi+1 = ceil(y),
    S_i 是第 i 个离散值的概率 (softmax 后)。

    最终坐标恢复: y_hat = Σ(i * S_i)  (概率加权求和)
    """

    def __init__(self, reg_max: int = 16):
        """
        Args:
            reg_max: 离散区间的最大值 (通常为 16, 表示 0~15 共 16 个离散值)
        """
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测的 logits, shape (N, 4 * self.reg_max)
                  每个框有 4 条边，每条边对应 reg_max 个 logits
            target: 目标连续值, shape (N, 4)
                    值为 [l, t, r, b] 四条边距离

        Returns:
            loss: DFL 损失值, scalar
        """
        N = pred.shape[0]
        # 重塑为 (N, 4, reg_max)
        pred = pred.view(N, 4, self.reg_max)

        # 将目标值裁剪到 [0, reg_max - 1] 范围内
        target = torch.clamp(target, 0, self.reg_max - 1)

        # 找到目标值对应两个离散点的索引
        # floor 和 ceil
        target_left = torch.floor(target).long()   # yi
        target_right = torch.ceil(target).long()   # yi+1

        # 处理边界情况: 当 target 正好是整数时
        target_right = torch.clamp(target_right, 0, self.reg_max - 1)

        # 计算权重: yi+1 - y 和 y - yi
        weight_left = (target_right.float() - target)  # yi+1 - y
        weight_right = (target - target_left.float())  # y - yi

        # 对每条边、每个样本计算交叉熵
        # pred: (N, 4, reg_max) → 对 reg_max 维做 softmax
        pred_softmax = F.softmax(pred, dim=-1)  # (N, 4, reg_max)

        # 取目标索引对应的概率
        # 需要扩展到 (N, 4, 1) 然后取对应值
        prob_left = torch.gather(pred_softmax, 2, target_left.unsqueeze(-1)).squeeze(-1)  # (N, 4)
        prob_right = torch.gather(pred_softmax, 2, target_right.unsqueeze(-1)).squeeze(-1)  # (N, 4)

        # DFL Loss: -((yi+1 - y) * log(S_i) + (y - yi) * log(S_{i+1}))
        # 加上小 epsilon 避免 log(0)
        eps = 1e-7
        loss = - (weight_left * torch.log(prob_left + eps) +
                  weight_right * torch.log(prob_right + eps))

        # 对 4 条边取平均，再对 batch 取平均
        loss = loss.mean()
        return loss


# ============================================================
# 2. 从分布恢复坐标
# ============================================================

def distribution_to_bbox(pred_dist: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
    """
    从离散概率分布恢复边界框坐标。

    Args:
        pred_dist: 预测分布, shape (N, 4, reg_max) 或 (N, 4 * reg_max)
        reg_max: 离散区间最大值

    Returns:
        bbox: 恢复的坐标, shape (N, 4), 值为 [l, t, r, b]
    """
    if pred_dist.dim() == 2:
        N = pred_dist.shape[0]
        pred_dist = pred_dist.view(N, 4, reg_max)

    # softmax 将 logits 转为概率
    probs = F.softmax(pred_dist, dim=-1)  # (N, 4, reg_max)

    # 创建离散值索引: [0, 1, 2, ..., reg_max-1]
    values = torch.arange(reg_max, device=pred_dist.device, dtype=torch.float32)
    values = values.view(1, 1, reg_max)  # (1, 1, reg_max)

    # 概率加权求和: y_hat = Σ(i * S_i)
    bbox = (probs * values).sum(dim=-1)  # (N, 4)
    return bbox


# ============================================================
# 3. 与 Ultralytics 源码对照实现
# ============================================================

class DFLLossUltralyticsStyle(nn.Module):
    """
    与 Ultralytics 源码风格一致的 DFL 实现。
    参考: ultralytics/utils/loss.py 中的 DistributionFocalLoss
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        """
        Args:
            pred_dist: (N, 4*reg_max) 或 (N, 4, reg_max)
            target: (N, 4) 连续值
        """
        if pred_dist.dim() == 2:
            pred_dist = pred_dist.view(-1, 4, self.reg_max)

        # 将 target 转为 one-hot 风格的分布
        # 目标值在 [0, reg_max-1] 范围内
        target = target.clamp(0, self.reg_max - 1)

        # 获取左右索引
        target_left = target.long()
        target_right = target_left + 1
        target_right = target_right.clamp(0, self.reg_max - 1)

        # 权重
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left

        # 交叉熵 (使用 cross_entropy = log_softmax + nll_loss)
        # pred_dist: (N*4, reg_max), target: (N*4,)
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max),
            target_left.view(-1),
            reduction='none'
        ).view(-1, 4)
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max),
            target_right.view(-1),
            reduction='none'
        ).view(-1, 4)

        loss = (weight_left * loss_left + weight_right * loss_right).mean()
        return loss


# ============================================================
# 4. 测试与验证
# ============================================================

def test_dfl_against_ultralytics():
    """
    测试我们的 DFL 实现，模拟与 Ultralytics 结果对比。
    """
    print("=" * 60)
    print("测试 DFL 实现")
    print("=" * 60)

    reg_max = 16
    N = 8  # batch size
    torch.manual_seed(42)

    # 随机生成预测和目标
    pred = torch.randn(N, 4 * reg_max)
    target = torch.rand(N, 4) * (reg_max - 1)  # 目标值在 [0, 15] 范围内

    # 我们的实现
    dfl_loss = DistributionFocalLoss(reg_max=reg_max)
    loss1 = dfl_loss(pred, target)

    # Ultralytics 风格实现
    dfl_loss_ultra = DFLLossUltralyticsStyle(reg_max=reg_max)
    loss2 = dfl_loss_ultra(pred, target)

    print(f"我们的实现:      {loss1.item():.6f}")
    print(f"Ultralytics 风格: {loss2.item():.6f}")
    print(f"差异:            {abs(loss1.item() - loss2.item()):.6e}")

    # 测试坐标恢复
    bbox = distribution_to_bbox(pred, reg_max)
    print(f"\n预测分布形状:   {pred.shape}")
    print(f"恢复坐标形状:   {bbox.shape}")
    print(f"恢复坐标范围:   [{bbox.min().item():.3f}, {bbox.max().item():.3f}]")

    return loss1, loss2


# ============================================================
# 5. 可视化概率分布
# ============================================================

def visualize_distribution():
    """
    可视化一个真实预测输出的四条边的概率分布。
    """
    print("\n" + "=" * 60)
    print("可视化 DFL 概率分布")
    print("=" * 60)

    reg_max = 16
    torch.manual_seed(123)

    # 模拟一个实际预测: 假设真实目标值在位置 7.3, 5.8, 10.2, 3.0
    true_values = torch.tensor([[7.3, 5.8, 10.2, 3.0]])

    # 生成有偏好的预测 logits (在真实值附近有较高值)
    pred = torch.randn(1, 4, reg_max)
    for i, val in enumerate(true_values[0]):
        val_int = int(val)
        # 在真实值附近增加 logits
        pred[0, i, max(0, val_int-1):min(reg_max, val_int+2)] += 3.0
        pred[0, i, val_int] += 5.0  # 在真实值处峰值更高

    pred = pred.view(1, 4 * reg_max)

    # 计算概率分布
    pred_dist = pred.view(1, 4, reg_max)
    probs = F.softmax(pred_dist, dim=-1).squeeze(0).detach().numpy()

    # 恢复坐标
    bbox = distribution_to_bbox(pred, reg_max)
    print(f"真实值: l={true_values[0, 0].item():.1f}, t={true_values[0, 1].item():.1f}, "
          f"r={true_values[0, 2].item():.1f}, b={true_values[0, 3].item():.1f}")
    print(f"恢复值: l={bbox[0, 0].item():.3f}, t={bbox[0, 1].item():.3f}, "
          f"r={bbox[0, 2].item():.3f}, b={bbox[0, 3].item():.3f}")

    # 绘制
    edge_names = ['Left (l)', 'Top (t)', 'Right (r)', 'Bottom (b)']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        values = np.arange(reg_max)
        ax.bar(values, probs[i], color='steelblue', alpha=0.7, width=0.8)
        ax.axvline(x=true_values[0, i].item(), color='red', linestyle='--',
                   linewidth=2, label=f'True={true_values[0, i].item():.1f}')
        ax.axvline(x=bbox[0, i].item(), color='green', linestyle='-',
                   linewidth=2, label=f'Pred={bbox[0, i].item():.2f}')
        ax.set_xlabel('Discrete Value')
        ax.set_ylabel('Probability')
        ax.set_title(f'{edge_names[i]} Distribution')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('DFL: 四条边的概率分布 (红色=真值, 绿色=预测)', fontsize=14)
    plt.tight_layout()
    plt.savefig('dfl_distribution.png', dpi=150)
    plt.show()
    print("概率分布图已保存至 dfl_distribution.png")


# ============================================================
# 6. DFL 与传统回归的对比
# ============================================================

def compare_dfl_vs_l1():
    """
    对比 DFL 和传统 L1 Loss 在回归任务中的表现。
    """
    print("\n" + "=" * 60)
    print("DFL vs L1 Loss 回归对比")
    print("=" * 60)

    reg_max = 16
    torch.manual_seed(42)

    # 模拟数据: 100 个样本，4 条边
    N = 100
    true_targets = torch.rand(N, 4) * 12 + 2  # 目标值在 [2, 14] 范围内

    # 初始化预测 (随机)
    pred_logits = torch.randn(N, 4 * reg_max, requires_grad=True)

    # DFL 优化
    dfl_loss_fn = DistributionFocalLoss(reg_max=reg_max)
    optimizer_dfl = torch.optim.SGD([pred_logits], lr=0.1)

    # 恢复坐标
    def get_coords(logits):
        return distribution_to_bbox(logits, reg_max)

    dfl_errors = []
    for step in range(200):
        optimizer_dfl.zero_grad()
        loss = dfl_loss_fn(pred_logits, true_targets)
        loss.backward()
        optimizer_dfl.step()

        if step % 40 == 0:
            pred_coords = get_coords(pred_logits)
            mae = F.l1_loss(pred_coords, true_targets).item()
            dfl_errors.append(mae)
            print(f"  DFL Step {step:3d}: loss={loss.item():.6f}, MAE={mae:.4f}")

    # 传统 L1 优化 (直接优化坐标)
    pred_coords_direct = torch.rand(N, 4, requires_grad=True)
    optimizer_l1 = torch.optim.SGD([pred_coords_direct], lr=0.01)

    l1_errors = []
    for step in range(200):
        optimizer_l1.zero_grad()
        loss = F.l1_loss(pred_coords_direct, true_targets)
        loss.backward()
        optimizer_l1.step()

        if step % 40 == 0:
            mae = F.l1_loss(pred_coords_direct, true_targets).item()
            l1_errors.append(mae)
            print(f"  L1  Step {step:3d}: loss={loss.item():.6f}, MAE={mae:.4f}")

    # 总结
    print(f"\n最终 DFL MAE: {dfl_errors[-1]:.4f}")
    print(f"最终 L1  MAE: {l1_errors[-1]:.4f}")
    print("DFL 优势: 分布建模提供了更精细的梯度信息，收敛更稳定")

    return dfl_errors, l1_errors


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Day 6 - 深度剖析 DFL (Distribution Focal Loss)")
    print("=" * 80)

    # 1. 测试 DFL 实现
    test_dfl_against_ultralytics()

    # 2. 可视化概率分布
    visualize_distribution()

    # 3. DFL vs L1 对比
    compare_dfl_vs_l1()

    # 4. 总结
    print("\n" + "=" * 60)
    print("DFL 核心要点总结")
    print("=" * 60)
    print("""
1. 传统回归的问题:
   - 直接预测坐标偏移量，假设坐标为 Dirac Delta 单点分布
   - 梯度信息有限，对模糊边界不鲁棒

2. DFL 的创新:
   - 将每条边建模为 reg_max 个离散值的概率分布
   - 损失函数: 交叉熵 + 线性插值
   - 坐标恢复: 概率加权求和 (软性 argmax)

3. DFL 的优势:
   - 提供更丰富的梯度信号
   - 对边界模糊、遮挡等不确定场景更鲁棒
   - 分布天然支持"不确定性估计"

4. 在 YOLOv8 中的位置:
   - 回归分支输出: 4 * reg_max 个 logits
   - 配合 CIoU Loss 共同优化回归任务
   - 源码: ultralytics/utils/loss.py → v8DetectionLoss
    """)