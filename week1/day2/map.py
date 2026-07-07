"""
Day 2 - mAP 计算原理与评价指标
=================================
从零实现 VOC 和 COCO 两种 mAP 计算流程，
并与 pycocotools 结果进行验证。

Author: YOLO Lesson Week1
"""

import numpy as np
import json
from collections import defaultdict
from typing import List, Dict, Tuple


# ============================================================
# 1. 构建模拟检测结果和真值
# ============================================================

def create_sample_data():
    """
    构建简单的模拟检测结果。

    Returns:
        detections: list of dict, 每个检测结果包含:
            - image_id: 图片ID
            - category_id: 类别ID
            - bbox: [x, y, w, h] (COCO 格式)
            - score: 置信度
        annotations: list of dict, 每个真值包含:
            - image_id: 图片ID
            - category_id: 类别ID
            - bbox: [x, y, w, h]
    """
    detections = [
        # 图片0: 两个目标
        {"image_id": 0, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.95},
        {"image_id": 0, "category_id": 1, "bbox": [12, 12, 45, 48], "score": 0.85},
        {"image_id": 0, "category_id": 1, "bbox": [100, 100, 30, 30], "score": 0.75},  # FP
        {"image_id": 0, "category_id": 2, "bbox": [70, 70, 40, 40], "score": 0.90},
        # 图片1: 一个目标
        {"image_id": 1, "category_id": 1, "bbox": [20, 20, 60, 60], "score": 0.92},
        {"image_id": 1, "category_id": 1, "bbox": [22, 22, 55, 55], "score": 0.80},
        # 图片2: 一个目标
        {"image_id": 2, "category_id": 2, "bbox": [30, 30, 80, 80], "score": 0.88},
    ]

    annotations = [
        # 图片0
        {"image_id": 0, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "id": 1},
        {"image_id": 0, "category_id": 1, "bbox": [60, 60, 40, 40], "area": 1600, "id": 2},
        {"image_id": 0, "category_id": 2, "bbox": [70, 70, 40, 40], "area": 1600, "id": 3},
        # 图片1
        {"image_id": 1, "category_id": 1, "bbox": [20, 20, 60, 60], "area": 3600, "id": 4},
        # 图片2
        {"image_id": 2, "category_id": 2, "bbox": [30, 30, 80, 80], "area": 6400, "id": 5},
    ]

    return detections, annotations


# ============================================================
# 2. IoU 计算 (COCO 格式: [x, y, w, h])
# ============================================================

def iou_coco(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个 COCO 格式框 [x, y, w, h] 的 IoU。
    """
    # 转换为 [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter

    return inter / (union + 1e-7)


# ============================================================
# 3. 单个类别、单个 IoU 阈值下的 TP/FP 计算
# ============================================================

def compute_tp_fp(
    detections: List[Dict],
    annotations: List[Dict],
    category_id: int,
    iou_thresh: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    计算单个类别、单个 IoU 阈值下的 TP/FP。

    Args:
        detections: 检测结果列表
        annotations: 真值列表
        category_id: 当前计算的类别ID
        iou_thresh: IoU 阈值

    Returns:
        tp: bool 数组，标记每个检测是否为 TP
        fp: bool 数组，标记每个检测是否为 FP
        num_gt: 该类别的真值总数
    """
    # 筛选当前类别的检测框，按置信度降序排列
    cls_dets = [d for d in detections if d["category_id"] == category_id]
    cls_dets = sorted(cls_dets, key=lambda x: x["score"], reverse=True)

    # 筛选当前类别的真值框
    cls_gts = [g for g in annotations if g["category_id"] == category_id]
    num_gt = len(cls_gts)

    # 标记每个真值框是否已被匹配
    gt_matched = [False] * num_gt

    tp = np.zeros(len(cls_dets), dtype=bool)
    fp = np.zeros(len(cls_dets), dtype=bool)

    for i, det in enumerate(cls_dets):
        det_bbox = np.array(det["bbox"])
        best_iou = 0
        best_gt_idx = -1

        # 寻找与该检测框 IoU 最大的未匹配真值框
        for j, gt in enumerate(cls_gts):
            if gt_matched[j]:
                continue
            # 确保同一图片
            if det["image_id"] != gt["image_id"]:
                continue
            iou_val = iou_coco(det_bbox, np.array(gt["bbox"]))
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = j

        # 判断 TP 还是 FP
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp[i] = True
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = True

    return tp, fp, num_gt


# ============================================================
# 4. Precision-Recall 曲线与 AP 计算
# ============================================================

def compute_precision_recall(tp: np.ndarray, fp: np.ndarray, num_gt: int):
    """
    给定 TP/FP 数组, 计算不同置信度阈值下的 Precision 和 Recall。

    Returns:
        precisions, recalls: 按置信度降序排列的精确率和召回率数组
    """
    # 累计 TP 和 FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Precision = TP / (TP + FP)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

    # Recall = TP / num_gt
    recalls = tp_cumsum / (num_gt + 1e-7)

    return precisions, recalls


def compute_ap_voc11(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    """
    VOC 2007 标准: 11 点插值 AP。
    在 Recall 的 [0.0, 0.1, ..., 1.0] 这 11 个点上取该点右侧的最大 Precision 值，求平均。
    """
    precisions, recalls = compute_precision_recall(tp, fp, num_gt)

    ap = 0.0
    for recall_thresh in np.linspace(0, 1, 11):
        # 取该 recall 阈值右侧的最大 precision
        mask = recalls >= recall_thresh
        if mask.any():
            p = np.max(precisions[mask])
        else:
            p = 0
        ap += p

    return ap / 11


def compute_ap_coco(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    """
    COCO 标准: 101 点插值 AP。
    在 Recall 的 0.00 到 1.00 等间距取 101 个点，计算每个点右侧最大 Precision，求平均。
    """
    precisions, recalls = compute_precision_recall(tp, fp, num_gt)

    ap = 0.0
    for recall_thresh in np.linspace(0, 1, 101):
        mask = recalls >= recall_thresh
        if mask.any():
            p = np.max(precisions[mask])
        else:
            p = 0
        ap += p

    return ap / 101


def compute_ap_all_iou_thresholds(
    detections: List[Dict],
    annotations: List[Dict],
    category_id: int
) -> Tuple[float, float, float]:
    """
    计算 IoU 从 0.5 到 0.95 (步长 0.05) 的所有 AP，并返回 mAP@[0.5:0.95]。
    同时返回 mAP@0.5 和 mAP@0.75。

    Returns:
        ap_50: IoU=0.5 时的 AP
        ap_75: IoU=0.75 时的 AP
        ap_all: mAP@[0.5:0.95] (10 个 IoU 阈值的平均 AP)
    """
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps = []

    for iou_thresh in iou_thresholds:
        tp, fp, num_gt = compute_tp_fp(detections, annotations, category_id, iou_thresh)
        if num_gt == 0:
            ap = 0.0
        else:
            ap = compute_ap_coco(tp, fp, num_gt)
        aps.append(ap)

    # 提取 AP@0.5 和 AP@0.75
    ap_50 = aps[0]  # IoU=0.5
    ap_75 = aps[5]  # IoU=0.75
    ap_all = np.mean(aps)

    return ap_50, ap_75, ap_all


# ============================================================
# 5. 主评估流程
# ============================================================

def evaluate(
    detections: List[Dict],
    annotations: List[Dict],
    categories: List[int] = None
) -> Dict:
    """
    完整评估流程: 计算所有类别的 AP 和 mAP。

    Args:
        detections: 检测结果列表
        annotations: 真值列表
        categories: 类别ID列表, None 则自动检测

    Returns:
        results dict
    """
    if categories is None:
        categories = sorted(set(
            [d["category_id"] for d in detections] +
            [g["category_id"] for g in annotations]
        ))

    results = {}
    all_aps_50 = []
    all_aps_all = []

    print(f"{'Category':<12} {'AP@0.5':<12} {'AP@0.75':<12} {'AP@[0.5:0.95]':<15}")
    print("=" * 55)

    for cat_id in categories:
        ap_50, ap_75, ap_all = compute_ap_all_iou_thresholds(detections, annotations, cat_id)
        results[cat_id] = {"AP@0.5": ap_50, "AP@0.75": ap_75, "AP@[0.5:0.95]": ap_all}
        all_aps_50.append(ap_50)
        all_aps_all.append(ap_all)

        print(f"Category {cat_id:<5} {ap_50:<12.4f} {ap_75:<12.4f} {ap_all:<15.4f}")

    # mAP
    results["mAP@0.5"] = np.mean(all_aps_50)
    results["mAP@[0.5:0.95]"] = np.mean(all_aps_all)

    print("=" * 55)
    print(f"{'mAP':<12} {results['mAP@0.5']:<12.4f} {'':<12} {results['mAP@[0.5:0.95]']:<15.4f}")

    return results


# ============================================================
# 6. PR 曲线绘制
# ============================================================

def plot_pr_curve(
    tp: np.ndarray, fp: np.ndarray, num_gt: int,
    title: str = "Precision-Recall Curve"
):
    """绘制 PR 曲线。"""
    try:
        import matplotlib.pyplot as plt

        precisions, recalls = compute_precision_recall(tp, fp, num_gt)
        ap = compute_ap_coco(tp, fp, num_gt)

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, 'b-', linewidth=2, label=f'AP={ap:.4f}')
        plt.fill_between(recalls, precisions, alpha=0.2, color='blue')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        plt.savefig('pr_curve.png', dpi=150)
        plt.show()
        print(f"PR curve saved to pr_curve.png, AP={ap:.4f}")
    except ImportError:
        print("matplotlib not available, skipping plot.")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Day 2 - mAP 计算原理与评价指标")
    print("=" * 80)

    # 1. 创建模拟数据
    detections, annotations = create_sample_data()
    print(f"\n模拟数据: {len(detections)} 个检测框, {len(annotations)} 个真值框")

    # 2. 手动实现评估
    print("\n1. 手动实现 mAP 评估")
    print("-" * 40)
    results = evaluate(detections, annotations)

    # 3. PR 曲线 (类别1)
    print("\n2. PR 曲线 (类别1)")
    print("-" * 40)
    tp, fp, num_gt = compute_tp_fp(detections, annotations, category_id=1, iou_thresh=0.5)
    print(f"类别1: TP={tp.sum()}, FP={fp.sum()}, GT={num_gt}")
    # plot_pr_curve(tp, fp, num_gt, title="Class 1 PR Curve (IoU=0.5)")

    # 4. 详细 TP/FP 分析
    print("\n3. 类别1 详细匹配结果 (按置信度降序)")
    print("-" * 40)
    cls_dets = sorted(
        [d for d in detections if d["category_id"] == 1],
        key=lambda x: x["score"], reverse=True
    )
    print(f"{'Score':<8} {'Is TP':<8}")
    for i, det in enumerate(cls_dets):
        print(f"{det['score']:<8.3f} {str(tp[i]):<8}")

    # 5. 与 pycocotools 验证 (可选)
    print("\n4. 与 pycocotools 对比验证")
    print("-" * 40)
    print("如需验证，请安装 pycocotools 并运行以下代码：")
    print("""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # 构建 COCO 格式的标注和结果
    coco_gt = COCO()
    coco_gt.dataset = {...}  # 你的真值数据
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(detection_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    """)