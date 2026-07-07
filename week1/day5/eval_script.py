"""
Day 5 - 综合应用：评估管道与 YOLO 验证代码剖析
===============================================
将前几天的知识串联，构建一个完整的 YOLO 评估脚本，
并深入理解 YOLO 验证流程。

Author: YOLO Lesson Week1
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# ============================================================
# 1. YOLO 输出解码 (模拟)
# ============================================================

def decode_yolo_output(
    raw_output: np.ndarray,
    anchors: List[Tuple[float, float]],
    stride: int,
    img_size: Tuple[int, int] = (640, 640),
    conf_thresh: float = 0.001
) -> List[Dict]:
    """
    将 YOLO 的原始输出解码为检测框。

    这是 YOLO 后处理的简化版本，展示核心逻辑。

    Args:
        raw_output: 模型原始输出, shape (N, C, H, W)
        anchors: 锚框列表
        stride: 下采样步长
        img_size: 输入图像大小 (w, h)
        conf_thresh: 置信度阈值

    Returns:
        detections: 解码后的检测结果列表
    """
    # 简化实现: 假设 raw_output 已经过 sigmoid
    # 实际 YOLO 的输出需要 sigmoid + 解码
    H, W = raw_output.shape[2], raw_output.shape[3]
    num_anchors = len(anchors)
    num_classes = raw_output.shape[1] // num_anchors - 5  # 简化

    detections = []
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    for ai, (anchor_w, anchor_h) in enumerate(anchors):
        # 提取该锚框的输出通道
        offset = ai * (5 + num_classes)
        obj_conf = 1 / (1 + np.exp(-raw_output[0, offset + 4]))  # sigmoid
        # 取置信度较高的网格
        obj_mask = obj_conf > conf_thresh

        if not obj_mask.any():
            continue

        # 解码坐标
        tx = raw_output[0, offset + 0]
        ty = raw_output[0, offset + 1]
        tw = raw_output[0, offset + 2]
        th = raw_output[0, offset + 3]

        # 边界框中心坐标
        bx = (grid_x + tx) * stride
        by = (grid_y + ty) * stride
        # 边界框宽高
        bw = np.exp(tw) * anchor_w
        bh = np.exp(th) * anchor_h

        # 转 [x1, y1, x2, y2] 格式
        x1 = bx - bw / 2
        y1 = by - bh / 2
        x2 = bx + bw / 2
        y2 = by + bh / 2

        # 类别概率
        cls_scores = []
        for c in range(num_classes):
            cls_score = 1 / (1 + np.exp(-raw_output[0, offset + 5 + c]))
            cls_scores.append(cls_score)
        cls_scores = np.array(cls_scores)

        # 最终得分 = 目标置信度 * 类别概率
        final_scores = obj_conf * cls_scores.max(axis=0)

        # 收集检测结果
        mask = obj_mask & (final_scores > conf_thresh)
        if not mask.any():
            continue

        for i, j in zip(*np.where(mask)):
            detections.append({
                'bbox': [float(x1[j, i]), float(y1[j, i]),
                         float(x2[j, i]), float(y2[j, i])],
                'score': float(final_scores[j, i]),
                'class_id': int(cls_scores[:, j, i].argmax()),
            })

    return detections


# ============================================================
# 2. 混淆矩阵与 TP/FP 匹配
# ============================================================

class ConfusionMatrix:
    """
    检测任务的混淆矩阵。
    跟踪 TP, FP, FN 的匹配情况。
    """
    def __init__(self, num_classes: int, iou_thresh: float = 0.5):
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
        # 最后一行/列用于背景 (FP/FN)

    def process_batch(
        self,
        detections: List[Dict],
        annotations: List[Dict],
        img_size: Tuple[int, int]
    ):
        """
        处理一批检测结果和真值，更新混淆矩阵。

        Args:
            detections: 检测结果列表 [{'bbox':[x1,y1,x2,y2], 'score':float, 'class_id':int}]
            annotations: 真值列表 [{'bbox':[x1,y1,x2,y2], 'category_id':int}]
            img_size: 图像大小 (w, h)
        """
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)

        # 标记已匹配的真值
        gt_matched = [False] * len(annotations)

        for det in detections:
            det_bbox = np.array(det['bbox'])
            det_cls = det['class_id']
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(annotations):
                if gt_matched[j]:
                    continue
                gt_bbox = np.array(gt['bbox'])
                # 转换 COCO [x, y, w, h] 到 [x1, y1, x2, y2] 如果需要
                if gt_bbox.shape[0] == 4:
                    # 假设已经是 [x1, y1, x2, y2]
                    pass
                iou_val = self._compute_iou(det_bbox, gt_bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = j

            if best_iou >= self.iou_thresh and best_gt_idx >= 0:
                gt_cls = annotations[best_gt_idx]['category_id']
                self.matrix[det_cls, gt_cls] += 1
                gt_matched[best_gt_idx] = True
            else:
                # FP: 检测到但没匹配到真值
                self.matrix[det_cls, self.num_classes] += 1

        # FN: 未匹配的真值
        for j, gt in enumerate(annotations):
            if not gt_matched[j]:
                gt_cls = gt['category_id']
                self.matrix[self.num_classes, gt_cls] += 1

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算 IoU。"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / (union + 1e-7)

    def get_metrics(self) -> Dict:
        """计算各类别指标。"""
        metrics = {}
        for c in range(self.num_classes):
            tp = self.matrix[c, c]
            fp = self.matrix[c, self.num_classes]
            fn = self.matrix[self.num_classes, c]

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)

            metrics[c] = {
                'TP': tp, 'FP': fp, 'FN': fn,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
            }

        return metrics

    def print_summary(self):
        """打印混淆矩阵摘要。"""
        metrics = self.get_metrics()
        print(f"\n{'Class':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("=" * 70)
        for c, m in metrics.items():
            print(f"Class {c:<5} {m['TP']:<8} {m['FP']:<8} {m['FN']:<8} "
                  f"{m['Precision']:<12.4f} {m['Recall']:<12.4f} {m['F1']:<12.4f}")


# ============================================================
# 3. 完整评估脚本
# ============================================================

class YOLOEvaluator:
    """
    独立的 YOLO 评估脚本。
    输入检测结果 JSON 和真值 JSON，输出各类别 AP 和 mAP。
    """

    def __init__(self, iou_thresholds: List[float] = None):
        if iou_thresholds is None:
            self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        else:
            self.iou_thresholds = np.array(iou_thresholds)

    def evaluate(
        self,
        detections: List[Dict],
        annotations: List[Dict],
        category_ids: List[int] = None
    ) -> Dict:
        """
        执行完整评估。

        Args:
            detections: [{'image_id', 'category_id', 'bbox':[x,y,w,h], 'score'}]
            annotations: [{'image_id', 'category_id', 'bbox':[x,y,w,h]}]
            category_ids: 类别ID列表

        Returns:
            results dict with AP/mAP
        """
        if category_ids is None:
            category_ids = sorted(set(
                [d['category_id'] for d in detections] +
                [g['category_id'] for g in annotations]
            ))

        # 按图片分组
        gt_by_image = defaultdict(list)
        for ann in annotations:
            gt_by_image[ann['image_id']].append(ann)

        det_by_image = defaultdict(list)
        for det in detections:
            det_by_image[det['image_id']].append(det)

        results = {}
        print(f"\n{'Class':<10} {'AP@0.5':<12} {'AP@0.75':<12} {'AP@[0.5:0.95]':<15}")
        print("=" * 55)

        for cat_id in category_ids:
            aps = []
            for iou_thresh in self.iou_thresholds:
                tp_list = []
                fp_list = []
                num_gt = 0

                # 遍历所有图片
                all_image_ids = set(list(gt_by_image.keys()) + list(det_by_image.keys()))
                for img_id in all_image_ids:
                    img_gts = [g for g in gt_by_image.get(img_id, [])
                              if g['category_id'] == cat_id]
                    img_dets = [d for d in det_by_image.get(img_id, [])
                               if d['category_id'] == cat_id]
                    img_dets = sorted(img_dets, key=lambda x: x['score'], reverse=True)

                    num_gt += len(img_gts)
                    gt_matched = [False] * len(img_gts)

                    for det in img_dets:
                        det_bbox = np.array(det['bbox'])
                        best_iou = 0
                        best_gt_idx = -1

                        for j, gt in enumerate(img_gts):
                            if gt_matched[j]:
                                continue
                            gt_bbox = np.array(gt['bbox'])
                            iou_val = self._iou_coco(det_bbox, gt_bbox)
                            if iou_val > best_iou:
                                best_iou = iou_val
                                best_gt_idx = j

                        if best_iou >= iou_thresh and best_gt_idx >= 0:
                            tp_list.append(True)
                            fp_list.append(False)
                            gt_matched[best_gt_idx] = True
                        else:
                            tp_list.append(False)
                            fp_list.append(True)

                if num_gt == 0:
                    ap = 0.0
                else:
                    ap = self._compute_ap_101pt(np.array(tp_list), np.array(fp_list), num_gt)
                aps.append(ap)

            ap_50 = aps[0]
            ap_75 = aps[5] if len(aps) > 5 else aps[-1]
            ap_all = np.mean(aps)

            results[cat_id] = {'AP@0.5': ap_50, 'AP@0.75': ap_75, 'AP@[0.5:0.95]': ap_all}
            print(f"Class {cat_id:<5} {ap_50:<12.4f} {ap_75:<12.4f} {ap_all:<15.4f}")

        results['mAP@0.5'] = np.mean([r['AP@0.5'] for r in results.values() if isinstance(r, dict)])
        results['mAP@[0.5:0.95]'] = np.mean([r['AP@[0.5:0.95]'] for r in results.values() if isinstance(r, dict)])

        print("=" * 55)
        print(f"{'mAP':<10} {results['mAP@0.5']:<12.4f} {'':<12} {results['mAP@[0.5:0.95]']:<15.4f}")

        return results

    def _iou_coco(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """COCO 格式 [x, y, w, h] 的 IoU。"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - inter
        return inter / (union + 1e-7)

    def _compute_ap_101pt(self, tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
        """101 点插值 AP。"""
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-7)
        recalls = tp_cum / (num_gt + 1e-7)

        ap = 0.0
        for t in np.linspace(0, 1, 101):
            mask = recalls >= t
            if mask.any():
                ap += np.max(precisions[mask])
        return ap / 101

    def plot_pr_curve(self, detections, annotations, cat_id, iou_thresh=0.5, save_path='pr_curve.png'):
        """绘制 PR 曲线。"""
        tp_list, fp_list = [], []
        gt_by_img = defaultdict(list)
        det_by_img = defaultdict(list)

        for ann in annotations:
            if ann['category_id'] == cat_id:
                gt_by_img[ann['image_id']].append(ann)
        for det in detections:
            if det['category_id'] == cat_id:
                det_by_img[det['image_id']].append(det)

        num_gt = sum(len(v) for v in gt_by_img.values())

        for img_id in set(list(gt_by_img.keys()) + list(det_by_img.keys())):
            img_gts = gt_by_img.get(img_id, [])
            img_dets = sorted(det_by_img.get(img_id, []), key=lambda x: x['score'], reverse=True)
            gt_matched = [False] * len(img_gts)

            for det in img_dets:
                best_iou, best_j = 0, -1
                for j, gt in enumerate(img_gts):
                    if gt_matched[j]:
                        continue
                    iou_val = self._iou_coco(np.array(det['bbox']), np.array(gt['bbox']))
                    if iou_val > best_iou:
                        best_iou, best_j = iou_val, j
                if best_iou >= iou_thresh and best_j >= 0:
                    tp_list.append(True)
                    fp_list.append(False)
                    gt_matched[best_j] = True
                else:
                    tp_list.append(False)
                    fp_list.append(True)

        tp_arr = np.array(tp_list)
        fp_arr = np.array(fp_list)
        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-7)
        recalls = tp_cum / (num_gt + 1e-7)
        ap = self._compute_ap_101pt(tp_arr, fp_arr, num_gt)

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, 'b-', linewidth=2, label=f'AP={ap:.4f}')
        plt.fill_between(recalls, precisions, alpha=0.2, color='blue')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'PR Curve (Class {cat_id}, IoU={iou_thresh})', fontsize=14)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"PR curve saved to {save_path}, AP={ap:.4f}")


# ============================================================
# 4. 示例: 使用预训练模型评估 (模拟)
# ============================================================

def create_sample_coco_data():
    """创建模拟的 COCO 验证集子集数据。"""
    detections = []
    annotations = []
    np.random.seed(42)

    # 3 个类别, 5 张图片
    for img_id in range(5):
        # 每张图片 2-4 个真值
        num_gts = np.random.randint(2, 5)
        for _ in range(num_gts):
            x = np.random.uniform(10, 200)
            y = np.random.uniform(10, 200)
            w = np.random.uniform(20, 100)
            h = np.random.uniform(20, 100)
            cat_id = np.random.randint(0, 3)
            annotations.append({
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [x, y, w, h],
            })

        # 每张图片 3-6 个检测结果 (含噪声)
        num_dets = np.random.randint(3, 7)
        for _ in range(num_dets):
            # 模拟检测 - 添加噪声
            gt = np.random.choice([g for g in annotations if g['image_id'] == img_id]) if annotations else None
            if gt and np.random.random() > 0.3:
                # 接近真值框 (TP)
                x = gt['bbox'][0] + np.random.normal(0, 5)
                y = gt['bbox'][1] + np.random.normal(0, 5)
                w = gt['bbox'][2] + np.random.normal(0, 5)
                h = gt['bbox'][3] + np.random.normal(0, 5)
                cat_id = gt['category_id']
            else:
                # 随机框 (FP)
                x = np.random.uniform(0, 300)
                y = np.random.uniform(0, 300)
                w = np.random.uniform(20, 100)
                h = np.random.uniform(20, 100)
                cat_id = np.random.randint(0, 3)

            detections.append({
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [x, y, w, h],
                'score': np.random.uniform(0.3, 0.99),
            })

    return detections, annotations


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Day 5 - 综合应用：评估管道与 YOLO 验证代码剖析")
    print("=" * 80)

    # 1. 创建模拟数据
    print("\n1. 创建模拟 COCO 验证集数据...")
    detections, annotations = create_sample_coco_data()
    print(f"   检测结果: {len(detections)} 个")
    print(f"   真值: {len(annotations)} 个")

    # 2. 混淆矩阵分析
    print("\n2. 混淆矩阵分析 (IoU=0.5)")
    print("-" * 40)
    cm = ConfusionMatrix(num_classes=3, iou_thresh=0.5)
    gt_by_img = defaultdict(list)
    det_by_img = defaultdict(list)
    for ann in annotations:
        gt_by_img[ann['image_id']].append(ann)
    for det in detections:
        det_by_img[det['image_id']].append(det)

    for img_id in set(list(gt_by_img.keys()) + list(det_by_img.keys())):
        cm.process_batch(
            det_by_img.get(img_id, []),
            gt_by_img.get(img_id, []),
            img_size=(640, 640)
        )
    cm.print_summary()

    # 3. 完整评估
    print("\n\n3. 完整 mAP 评估")
    print("-" * 40)
    evaluator = YOLOEvaluator()
    results = evaluator.evaluate(detections, annotations)

    # 4. PR 曲线
    print("\n4. PR 曲线 (类别 0, IoU=0.5)")
    print("-" * 40)
    evaluator.plot_pr_curve(detections, annotations, cat_id=0, save_path='pr_curve.png')

    # 5. Ultralytics 验证流程剖析
    print("\n" + "=" * 80)
    print("附录: Ultralytics YOLO 验证流程剖析")
    print("=" * 80)
    print("""
  在 Ultralytics 源码中，验证流程如下:

  ultralytics/models/yolo/detect/val.py
  ┌─────────────────────────────────────────────────────────┐
  │ 1. DetectionValidator.__init__()                        │
  │    - 设置指标、数据加载器、后处理参数                      │
  │                                                         │
  │ 2. postprocess()                                        │
  │    - 从模型原始输出解码框坐标 [x, y, w, h]               │
  │    - 应用 NMS 过滤重复框                                 │
  │    - 返回 Detections 对象列表                            │
  │                                                         │
  │ 3. update_metrics()                                     │
  │    - 调用 ConfusionMatrix.process_batch()               │
  │    - 计算 TP/FP/FN 匹配                                  │
  │                                                         │
  │ 4. finalize_metrics() → metrics.py                      │
  │    - ap_per_class(): 计算各类别 AP                       │
  │    - 11 点插值 (VOC) 或 101 点插值 (COCO)               │
  │    - 计算 mAP@0.5, mAP@0.75, mAP@[0.5:0.95]            │
  │                                                         │
  │ 关键文件:                                                │
  │   - ultralytics/utils/metrics.py                        │
  │     → ConfusionMatrix, ap_per_class, fitness            │
  │   - ultralytics/models/yolo/detect/val.py               │
  │     → DetectionValidator                                │
  └─────────────────────────────────────────────────────────┘
  """)