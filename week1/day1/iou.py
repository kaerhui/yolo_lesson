"""
Day 1 - IoU 及其变体的数学推导与实现
========================================
实现 IoU, GIoU, DIoU, CIoU 四种度量及其损失函数，
并通过梯度下降模拟回归过程。

Author: YOLO Lesson Week1
"""

import numpy as np
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from typing import Tuple


# ============================================================
# 1. IoU (Intersection over Union)
# ============================================================

def iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算两个框的 IoU。

    Args:
        box1, box2: shape (..., 4), 格式为 [x1, y1, x2, y2] (左上角, 右下角)
    Returns:
        iou: shape (...), IoU 值
    """
    # 交集区域的左上角和右下角
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    # 交集面积 (clamp 保证无重叠时为 0)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 并集面积
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = area1 + area2 - inter_area

    # 防止除零
    iou_val = inter_area / (union_area + 1e-7)
    return iou_val


def iou_loss(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """IoU Loss: L = 1 - IoU"""
    return 1 - iou(box1, box2)


# ============================================================
# 2. GIoU (Generalized IoU)
# ============================================================

def giou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算 GIoU。

    GIoU = IoU - |C \ (A ∪ B)| / |C|
    其中 C 是包含 A 和 B 的最小外接闭包框。
    """
    iou_val = iou(box1, box2)

    # 最小外接闭包框 C
    C_x1 = torch.min(box1[..., 0], box2[..., 0])
    C_y1 = torch.min(box1[..., 1], box2[..., 1])
    C_x2 = torch.max(box1[..., 2], box2[..., 2])
    C_y2 = torch.max(box1[..., 3], box2[..., 3])

    C_area = (C_x2 - C_x1) * (C_y2 - C_y1)

    # 并集面积
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = area1 + area2 - (torch.clamp(C_x2 - C_x1, min=0) * torch.clamp(C_y2 - C_y1, min=0)) + (  # 用 inter 简化
        # 直接计算 union
    )
    # 更正: 用已有的 iou 计算 GIoU
    # GIoU 的惩罚项: (C_area - union_area) / C_area
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    union_area = area1 + area2 - inter_area

    giou_val = iou_val - (C_area - union_area) / (C_area + 1e-7)
    return giou_val


def giou_loss(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """GIoU Loss: L = 1 - GIoU"""
    return 1 - giou(box1, box2)


# ============================================================
# 3. DIoU (Distance-IoU)
# ============================================================

def diou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算 DIoU。

    DIoU = IoU - ρ²(b, b_gt) / c²
    其中 ρ 是中心点欧氏距离, c 是最小外接框对角线长度。
    """
    iou_val = iou(box1, box2)

    # 中心点坐标
    center_x1 = (box1[..., 0] + box1[..., 2]) / 2
    center_y1 = (box1[..., 1] + box1[..., 3]) / 2
    center_x2 = (box2[..., 0] + box2[..., 2]) / 2
    center_y2 = (box2[..., 1] + box2[..., 3]) / 2

    # 中心点欧氏距离的平方
    rho_squared = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

    # 最小外接框对角线长度的平方
    C_x1 = torch.min(box1[..., 0], box2[..., 0])
    C_y1 = torch.min(box1[..., 1], box2[..., 1])
    C_x2 = torch.max(box1[..., 2], box2[..., 2])
    C_y2 = torch.max(box1[..., 3], box2[..., 3])
    c_squared = (C_x2 - C_x1) ** 2 + (C_y2 - C_y1) ** 2

    diou_val = iou_val - rho_squared / (c_squared + 1e-7)
    return diou_val


def diou_loss(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """DIoU Loss: L = 1 - DIoU"""
    return 1 - diou(box1, box2)


# ============================================================
# 4. CIoU (Complete IoU)
# ============================================================

def ciou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算 CIoU。

    CIoU = IoU - (ρ²(b,b_gt)/c² + αv)

    其中:
    v = (4/π²) * (arctan(w_gt/h_gt) - arctan(w/h))²
    α = v / ((1 - IoU) + v)
    """
    iou_val = iou(box1, box2)

    # DIoU 的惩罚项
    center_x1 = (box1[..., 0] + box1[..., 2]) / 2
    center_y1 = (box1[..., 1] + box1[..., 3]) / 2
    center_x2 = (box2[..., 0] + box2[..., 2]) / 2
    center_y2 = (box2[..., 1] + box2[..., 3]) / 2
    rho_squared = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

    C_x1 = torch.min(box1[..., 0], box2[..., 0])
    C_y1 = torch.min(box1[..., 1], box2[..., 1])
    C_x2 = torch.max(box1[..., 2], box2[..., 2])
    C_y2 = torch.max(box1[..., 3], box2[..., 3])
    c_squared = (C_x2 - C_x1) ** 2 + (C_y2 - C_y1) ** 2

    # 长宽比一致性惩罚项
    w1 = box1[..., 2] - box1[..., 0]
    h1 = box1[..., 3] - box1[..., 1]
    w2 = box2[..., 2] - box2[..., 0]
    h2 = box2[..., 3] - box2[..., 1]

    # v = (4/π²) * (arctan(w2/h2) - arctan(w1/h1))²
    arctan_diff = torch.atan(w2 / (h2 + 1e-7)) - torch.atan(w1 / (h1 + 1e-7))
    v = (4 / (math.pi ** 2)) * (arctan_diff ** 2)

    # α = v / ((1 - IoU) + v)
    alpha = v / ((1 - iou_val) + v + 1e-7)

    ciou_val = iou_val - rho_squared / (c_squared + 1e-7) - alpha * v
    return ciou_val


def ciou_loss(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """CIoU Loss: L = 1 - CIoU"""
    return 1 - ciou(box1, box2)


# ============================================================
# 5. 对比分析：典型位置关系
# ============================================================

def create_test_cases():
    """
    创建五组典型位置关系的框对。
    格式: [x1, y1, x2, y2]
    """
    cases = [
        ("完全分离", np.array([0, 0, 2, 2]), np.array([4, 4, 6, 6])),
        ("部分重叠", np.array([0, 0, 3, 3]), np.array([2, 2, 5, 5])),
        ("完全包含", np.array([0, 0, 5, 5]), np.array([1, 1, 4, 4])),
        ("中心重合但大小不同", np.array([0, 0, 4, 4]), np.array([1, 1, 3, 3])),
        ("边缘接触", np.array([0, 0, 2, 2]), np.array([2, 0, 4, 2])),
    ]
    return cases


def compare_all_iou(cases):
    """在所有测试用例上计算四种 IoU 值并打印对比表格。"""
    print("=" * 80)
    print(f"{'位置关系':<20} {'IoU':<12} {'GIoU':<12} {'DIoU':<12} {'CIoU':<12}")
    print("=" * 80)

    for name, box1_np, box2_np in cases:
        box1 = torch.tensor(box1_np, dtype=torch.float32)
        box2 = torch.tensor(box2_np, dtype=torch.float32)

        iou_val = iou(box1, box2).item()
        giou_val = giou(box1, box2).item()
        diou_val = diou(box1, box2).item()
        ciou_val = ciou(box1, box2).item()

        print(f"{name:<20} {iou_val:<12.6f} {giou_val:<12.6f} {diou_val:<12.6f} {ciou_val:<12.6f}")
    print("=" * 80)


# ============================================================
# 6. 梯度下降模拟回归过程
# ============================================================

def simulate_regression(target_box, init_box, lr=0.05, steps=100, loss_fn="ciou"):
    """
    使用梯度下降模拟预测框回归过程。

    Args:
        target_box: 真值框 [x1, y1, x2, y2]
        init_box: 初始预测框 [x1, y1, x2, y2]
        lr: 学习率
        steps: 迭代步数
        loss_fn: 使用的损失函数 ("iou", "giou", "diou", "ciou")
    """
    # 选择损失函数
    loss_funcs = {
        "iou": iou_loss,
        "giou": giou_loss,
        "diou": diou_loss,
        "ciou": ciou_loss,
    }
    loss_func = loss_funcs[loss_fn]

    # 将框包装成可训练的张量
    pred = torch.tensor(init_box, dtype=torch.float32, requires_grad=True)
    target = torch.tensor(target_box, dtype=torch.float32)

    history = []
    loss_values = []

    for step in range(steps):
        loss = loss_func(pred, target)
        loss.backward()

        with torch.no_grad():
            pred -= lr * pred.grad
            pred.grad.zero_()

        history.append(pred.detach().clone().numpy())
        loss_values.append(loss.item())

        if step % 20 == 0:
            print(f"Step {step:3d}: loss={loss.item():.6f}, box={pred.detach().numpy()}")

    return np.array(history), loss_values


def plot_regression(history, loss_values, target_box, loss_name):
    """可视化回归过程和损失曲线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 框的回归轨迹
    ax1.add_patch(plt.Rectangle(
        (target_box[0], target_box[1]),
        target_box[2] - target_box[0],
        target_box[3] - target_box[1],
        fill=False, edgecolor='green', linewidth=2, label='Target'
    ))
    # 初始框
    init = history[0]
    ax1.add_patch(plt.Rectangle(
        (init[0], init[1]), init[2] - init[0], init[3] - init[1],
        fill=False, edgecolor='red', linewidth=1.5, linestyle='--', label='Initial'
    ))
    # 最终框
    final = history[-1]
    ax1.add_patch(plt.Rectangle(
        (final[0], final[1]), final[2] - final[0], final[3] - final[1],
        fill=False, edgecolor='blue', linewidth=2, label='Final'
    ))
    # 中间轨迹
    for i, box in enumerate(history[1:-1]):
        alpha = 0.3 + 0.5 * (i / len(history))
        ax1.add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor='orange', linewidth=0.5, alpha=alpha
        ))

    ax1.set_xlim(-1, 6)
    ax1.set_ylim(-1, 6)
    ax1.set_aspect('equal')
    ax1.set_title(f'{loss_name} Regression Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: 损失曲线
    ax2.plot(loss_values, 'b-', linewidth=1.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel(f'{loss_name} Loss')
    ax2.set_title('Loss Curve')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'regression_{loss_name.lower()}.png', dpi=150)
    plt.show()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Day 1 - IoU 及其变体的数学推导与实现")
    print("=" * 80)

    # 1. 对比四种 IoU
    print("\n1. 典型位置关系的 IoU 对比")
    print("-" * 40)
    cases = create_test_cases()
    compare_all_iou(cases)

    # 2. 梯度下降回归模拟
    print("\n2. CIoU Loss 梯度下降回归模拟")
    print("-" * 40)
    target = np.array([2.0, 2.0, 5.0, 5.0])
    init = np.array([0.0, 0.0, 3.0, 3.0])

    print("Target box:", target)
    print("Initial box:", init)
    print()

    history, loss_values = simulate_regression(
        target, init, lr=0.1, steps=100, loss_fn="ciou"
    )

    print("\n回归完成！")
    print(f"初始损失: {loss_values[0]:.6f}")
    print(f"最终损失: {loss_values[-1]:.6f}")
    print(f"最终预测框: {history[-1]}")
    print(f"目标真值框: {target}")