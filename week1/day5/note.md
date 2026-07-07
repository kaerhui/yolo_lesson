# Day 5 学习笔记：综合应用 — 评估管道与 YOLO 验证代码剖析

> Ultralytics 源码分析 + 评估脚本构建

---

## 一、Ultralytics YOLO 验证流程全景

### 核心文件

| 文件 | 作用 |
|------|------|
| `ultralytics/models/yolo/detect/val.py` | 检测验证器 |
| `ultralytics/utils/metrics.py` | 评估指标计算 |
| `ultralytics/utils/loss.py` | 损失函数（训练用） |
| `ultralytics/utils/tal.py` | 任务对齐分配器 |

### 验证流程

```
DetectionValidator.__init__()
    ↓
    设置指标、数据加载器、后处理参数
    ↓
postprocess()
    ├── 从模型原始输出解码框坐标
    ├── 应用 NMS 过滤重复框
    └── 返回 Detections 对象列表
    ↓
update_metrics()
    ├── 调用 ConfusionMatrix.process_batch()
    ├── 计算 TP/FP/FN 匹配
    └── 更新每个类别的统计
    ↓
finalize_metrics()
    ├── ap_per_class(): 计算各类别 AP
    ├── 11 点插值 (VOC) 或 101 点插值 (COCO)
    └── 计算 mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
```

---

## 二、关键源码解析

### 1. postprocess — 输出解码

```python
# 简化版: YOLO 后处理
def postprocess(self, preds, img, orig_imgs):
    # preds: 模型原始输出
    # 1. 解码框坐标 (从网格偏移量 → 绝对坐标)
    # 2. 应用 NMS
    # 3. 返回 Detections 对象
    preds = ops.non_max_suppression(
        preds,
        self.args.conf,      # 置信度阈值
        self.args.iou,       # IoU 阈值 (NMS)
        self.args.classes,   # 类别过滤
        agnostic=self.args.agnostic_nms,
        max_det=self.args.max_det,
    )
    return preds
```

### 2. ConfusionMatrix — 混淆矩阵

```python
# ultralytics/utils/metrics.py
class ConfusionMatrix:
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
        # 最后一行/列: 背景 (FP/FN)

    def process_batch(self, detections, labels):
        # 1. 按置信度排序检测结果
        # 2. 对每个检测:
        #    - 找到与 GT 最大 IoU 的匹配
        #    - IoU >= 阈值 → TP (matrix[det_cls, gt_cls]++)
        #    - 否则 → FP (matrix[det_cls, nc]++)
        # 3. 未匹配的 GT → FN (matrix[nc, gt_cls]++)
```

### 3. ap_per_class — AP 计算

```python
# ultralytics/utils/metrics.py
def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """
    计算每个类别的 AP (VOC 11 点或 COCO 101 点插值)

    Args:
        tp: (N,) 每个检测是否为 TP
        conf: (N,) 置信度
        pred_cls: (N,) 预测类别
        target_cls: (M,) 真实类别
    Returns:
        p, r, ap, f1, unique_classes
    """
    # 1. 按类别分组
    # 2. 对每个类别:
    #    - 按置信度降序排列
    #    - 计算累积 TP/FP → Precision, Recall
    #    - 计算 AP (VOC 11 点或 COCO 101 点)
    # 3. 返回各类别指标
```

---

## 三、完整评估脚本结构

### 输入输出

```
输入:
  - detection_results.json: [{'image_id', 'category_id', 'bbox', 'score'}]
  - ground_truth.json:      [{'image_id', 'category_id', 'bbox'}]

输出:
  - 各类别 AP@0.5, AP@0.75, AP@[0.5:0.95]
  - 数据集 mAP
  - PR 曲线图
```

### 核心步骤

```python
class YOLOEvaluator:
    def evaluate(self, detections, annotations):
        # 1. 按图片分组 GT 和检测
        # 2. 对每个类别:
        #    - 遍历所有图片
        #    - 进行 TP/FP 匹配
        #    - 计算 AP
        # 3. 汇总 mAP
        # 4. 绘制 PR 曲线
```

---

## 四、TP/FP 匹配的关键细节

### 匹配规则

1. 按置信度降序排列检测框
2. 对每个检测框，找到与它 IoU 最大的**未匹配** GT 框
3. 如果 IoU ≥ 阈值 → TP，标记 GT 已匹配
4. 否则 → FP
5. 未匹配的 GT → FN

### 注意事项

- **每个 GT 只能匹配一次**: 一一对应
- **同一图片内匹配**: 不同图片的检测和 GT 不交叉
- **类别匹配**: 检测和 GT 的类别 ID 必须一致

---

## 五、与 Ultralytics 官方结果对比验证

### 验证步骤

1. 用预训练模型在 COCO 验证集子集上运行推理
2. 保存检测结果为 JSON 格式
3. 用你的评估脚本计算 mAP
4. 用 Ultralytics 的 `val.py` 计算 mAP
5. 对比结果，分析误差

### 常见误差来源

- NMS 参数不同 (conf_thresh, iou_thresh)
- IoU 计算精度 (float32 vs float64)
- AP 插值方法 (VOC 11 点 vs COCO 101 点)
- 最大检测数限制 (max_det=300 vs 100)

---

## 六、参考资料

- **源码**:
  - `ultralytics/utils/metrics.py`: [GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py)
  - `ultralytics/models/yolo/detect/val.py`: [GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py)
- **工具**: matplotlib, json, numpy
- **本日代码**: [eval_script.py](eval_script.py) — 完整评估脚本 + PR 曲线绘制