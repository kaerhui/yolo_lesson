"""
Day 3 - 非极大值抑制 (NMS) 算法族
====================================
实现 Greedy NMS、Soft-NMS、DIoU-NMS 三种算法，
并在密集框场景下对比它们的抑制效果。

Author: YOLO Lesson Week1
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import List, Tuple, Callable


# ============================================================
# 辅助函数: IoU 计算
# ============================================================

def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个框的 IoU。
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-7)


def diou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个框的 DIoU。
    DIoU = IoU - ρ²(b, b_gt) / c²
    """
    iou_val = iou(box1, box2)

    # 中心点
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    rho_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # 最小外接框对角线
    C_x1 = min(box1[0], box2[0])
    C_y1 = min(box1[1], box2[1])
    C_x2 = max(box1[2], box2[2])
    C_y2 = max(box1[3], box2[3])
    c_sq = (C_x2 - C_x1) ** 2 + (C_y2 - C_y1) ** 2

    return iou_val - rho_sq / (c_sq + 1e-7)


# ============================================================
# 1. 传统 Greedy NMS
# ============================================================

def nms(dets: np.ndarray, scores: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    传统 Greedy NMS。

    Args:
        dets: (N, 4) 检测框, [x1, y1, x2, y2]
        scores: (N,) 置信度
        thresh: IoU 阈值

    Returns:
        keep: 保留框的索引数组
    """
    # 按置信度降序排序
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        # 取最高分框
        i = order[0]
        keep.append(i)

        # 计算其余框与最高分框的 IoU
        ious = np.array([iou(dets[i], dets[j]) for j in order[1:]])

        # 保留 IoU 小于阈值的框
        inds = np.where(ious <= thresh)[0]
        order = order[inds + 1]  # +1 因为 order[0] 被移除

    return np.array(keep)


# ============================================================
# 2. Soft-NMS
# ============================================================

def soft_nms(
    dets: np.ndarray,
    scores: np.ndarray,
    thresh: float = 0.5,
    sigma: float = 0.5,
    method: str = 'gaussian'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft-NMS: 不是暴力删除高 IoU 框，而是衰减其得分。

    Args:
        dets: (N, 4) 检测框
        scores: (N,) 置信度
        thresh: 最终得分阈值 (低于此值的框被移除)
        sigma: 高斯衰减的 sigma 参数
        method: 'linear' 或 'gaussian'

    Returns:
        keep: 保留框的索引
        new_scores: 更新后的得分
    """
    N = len(dets)
    # 复制得分，避免修改原数组
    new_scores = scores.copy()
    indices = np.arange(N)

    # 按原始置信度排序
    order = np.argsort(new_scores)[::-1]
    dets = dets[order]
    new_scores = new_scores[order]
    indices = indices[order]

    for i in range(N):
        # 当前最高分框
        max_score_idx = np.argmax(new_scores[i:])
        max_idx = i + max_score_idx

        # 交换当前位置与最高分位置
        if max_score_idx != 0:
            dets[[i, max_idx]] = dets[[max_idx, i]]
            new_scores[[i, max_idx]] = new_scores[[max_idx, i]]
            indices[[i, max_idx]] = indices[[max_idx, i]]

        # 跳过已衰减到 0 的框
        if new_scores[i] == 0:
            continue

        # 计算后续框与当前框的 IoU 并衰减得分
        for j in range(i + 1, N):
            if new_scores[j] == 0:
                continue

            iou_val = iou(dets[i], dets[j])

            if method == 'linear':
                # 线性衰减: s_i = s_i * (1 - IoU)
                if iou_val >= thresh:
                    new_scores[j] *= (1 - iou_val)
            elif method == 'gaussian':
                # 高斯衰减: s_i = s_i * exp(-IoU² / σ)
                new_scores[j] *= np.exp(-(iou_val ** 2) / sigma)

    # 筛选得分大于阈值的框
    final_mask = new_scores > thresh
    keep = indices[final_mask]
    final_scores = new_scores[final_mask]

    # 按得分降序排列
    sort_idx = np.argsort(final_scores)[::-1]
    return keep[sort_idx], final_scores[sort_idx]


# ============================================================
# 3. DIoU-NMS
# ============================================================

def diou_nms(dets: np.ndarray, scores: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    DIoU-NMS: 使用 DIoU 替代 IoU 进行抑制判定。
    对于中心点距离较远的框，即使 IoU 很大也可保留。

    Args:
        dets: (N, 4) 检测框
        scores: (N,) 置信度
        thresh: DIoU 阈值

    Returns:
        keep: 保留框的索引数组
    """
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        dious = np.array([diou(dets[i], dets[j]) for j in order[1:]])

        inds = np.where(dious <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


# ============================================================
# 4. 可视化工具
# ============================================================

def plot_boxes(
    dets: np.ndarray,
    scores: np.ndarray,
    keep: np.ndarray,
    title: str = "NMS Result",
    colors: List[str] = None
):
    """可视化 NMS 前后的框。"""
    if colors is None:
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    keep_set = set(keep)

    # 左图: 所有检测框
    for i in range(len(dets)):
        box = dets[i]
        color = 'green' if i in keep_set else 'red'
        alpha = 1.0 if i in keep_set else 0.3
        ax1.add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor=color, linewidth=1.5, alpha=alpha
        ))
        ax1.text(box[0], box[1] - 2, f'{scores[i]:.2f}',
                 fontsize=8, color=color, alpha=alpha)

    ax1.set_title(f'All Detections ({len(dets)} boxes)')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # 右图: 保留的框
    for i, idx in enumerate(keep):
        box = dets[idx]
        color = colors[i % len(colors)]
        ax2.add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor=color, linewidth=2, label=f'Score: {scores[idx]:.2f}'
        ))
        ax2.text(box[0], box[1] - 2, f'{scores[idx]:.2f}', fontsize=9, color=color)

    ax2.set_title(f'{title} - Kept {len(keep)} boxes')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    ax2.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()


# ============================================================
# 5. 构造密集场景与对比
# ============================================================

def create_crowded_scene():
    """
    构造一个密集目标场景（模拟人群或零件堆叠）。
    包含多个高度重叠的框。
    """
    dets = np.array([
        # 两个非常接近的目标（密集人群场景）
        [20, 20, 50, 50],   # 目标 A
        [25, 22, 55, 52],   # 目标 B (与 A 高度重叠)
        [22, 45, 52, 75],   # 目标 C (与 A/B 部分重叠)
        [60, 20, 90, 50],   # 目标 D (远离 A/B/C)
        [65, 25, 95, 55],   # 目标 E (与 D 高度重叠)
        # 一些分散的低置信度假阳性
        [5, 5, 25, 25],
        [75, 60, 95, 80],
        [10, 60, 30, 80],
    ], dtype=float)

    scores = np.array([0.95, 0.90, 0.85, 0.92, 0.88, 0.40, 0.35, 0.30])
    return dets, scores


def create_center_aligned_scene():
    """
    构造中心重合但大小不同的场景（测试 DIoU-NMS 的优势）。
    """
    dets = np.array([
        [20, 20, 80, 80],   # 大框 (中心: 50, 50)
        [30, 30, 70, 70],   # 中框 (中心: 50, 50)
        [40, 40, 60, 60],   # 小框 (中心: 50, 50)
        [10, 50, 30, 70],   # 瘦长框 (中心: 20, 60)
        [70, 50, 90, 70],   # 瘦长框 (中心: 80, 60)
    ], dtype=float)

    scores = np.array([0.95, 0.90, 0.85, 0.80, 0.75])
    return dets, scores


def compare_nms_methods(dets: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5):
    """对比三种 NMS 方法的抑制结果。"""
    print(f"\n检测框数量: {len(dets)}")
    print(f"IoU/DIoU 阈值: {iou_thresh}")
    print("=" * 60)

    # 1. Greedy NMS
    keep_nms = nms(dets, scores, thresh=iou_thresh)
    print(f"\n1. Greedy NMS: 保留 {len(keep_nms)} 个框")
    for idx in keep_nms:
        print(f"   - 框 {idx}: score={scores[idx]:.3f}, box={dets[idx]}")

    # 2. Soft-NMS (Gaussian)
    keep_soft, scores_soft = soft_nms(dets, scores, thresh=0.3, sigma=0.5, method='gaussian')
    print(f"\n2. Soft-NMS (Gaussian): 保留 {len(keep_soft)} 个框")
    for idx, score in zip(keep_soft, scores_soft):
        i = np.where((dets == dets[idx]).all(axis=1))[0][0]
        print(f"   - 框 {i}: score={score:.3f} (原始: {scores[i]:.3f}), box={dets[idx]}")

    # 3. DIoU-NMS
    keep_diou = diou_nms(dets, scores, thresh=iou_thresh)
    print(f"\n3. DIoU-NMS: 保留 {len(keep_diou)} 个框")
    for idx in keep_diou:
        print(f"   - 框 {idx}: score={scores[idx]:.3f}, box={dets[idx]}")

    # 可视化
    plot_boxes(dets, scores, keep_nms, title="Greedy NMS")
    plot_boxes(dets, scores, keep_soft, title="Soft-NMS Gaussian")
    plot_boxes(dets, scores, keep_diou, title="DIoU-NMS")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Day 3 - 非极大值抑制 (NMS) 算法族")
    print("=" * 80)

    # 场景 1: 密集目标场景
    print("\n" + "=" * 60)
    print("场景 1: 密集目标（模拟人群/零件堆叠）")
    print("=" * 60)
    dets1, scores1 = create_crowded_scene()
    compare_nms_methods(dets1, scores1, iou_thresh=0.5)

    # 场景 2: 中心重合但大小不同
    print("\n" + "=" * 60)
    print("场景 2: 中心重合但大小不同")
    print("=" * 60)
    dets2, scores2 = create_center_aligned_scene()
    compare_nms_methods(dets2, scores2, iou_thresh=0.5)

    # 分析总结
    print("\n" + "=" * 60)
    print("三种 NMS 算法对比总结")
    print("=" * 60)
    print("""
Greedy NMS:
    - 简单快速地移除高 IoU 的重复框
    - 缺陷：密集场景中会误删真实的紧邻目标
    - 适用：目标稀疏、重叠度低的情况

Soft-NMS:
    - 通过衰减而非删除来保留更多候选框
    - 高斯衰减比线性衰减更平滑
    - 适用：需要保留更多候选框做二次处理的场景

DIoU-NMS:
    - 考虑中心点距离，对中心点远的框更宽容
    - 在拥挤场景中表现优于传统 NMS
    - 适用：人群、密集小目标检测

工业缺陷检测建议：
    如果两个缺陷非常靠近 (高 IoU 但中心点不同)：
    → DIoU-NMS 最佳，因为它能区分中心点不同的密集目标
    → Soft-NMS 次之，通过得分衰减而非直接删除
    → Greedy NMS 最差，高 IoU 重叠会被直接删除
    """)