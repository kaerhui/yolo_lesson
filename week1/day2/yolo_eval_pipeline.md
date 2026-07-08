# YOLO 评估流水线：mAP 计算原理与调用链分析

> 基于 Ultralytics YOLOv8 源码分析（ultralytics-main）
>
> 文件: `ultralytics/engine/validator.py`, `ultralytics/models/yolo/detect/val.py`, `ultralytics/utils/metrics.py`

---

## 一、完整调用流水线总览

```
yolo val model=yolo26n.pt data=coco8.yaml
  │
  └─→ DetectionValidator.__call__()
       │  ultralytics/models/yolo/detect/val.py
       │  ultralytics/engine/validator.py#L146-260
       │
       ├─1── init_metrics(model)             ← 初始化评估参数
       │    │  detect/val.py#L78-124
       │    │  设置 iouv (IoU阈值), 创建 DetMetrics 对象
       │    │
       │    └─ DetMetrics.__init__()         ← 创建 Metric 统计容器
       │         metrics.py#L1130-1137
       │         stats = {tp:[], conf:[], pred_cls:[], target_cls:[], target_img:[]}
       │
       ├─2── 遍历 dataloader:
       │    │
       │    ├─ preprocess(batch)              ← 图像预处理
       │    │    detect/val.py#L48-57
       │    │    img /= 255, 转移到 GPU
       │    │
       │    ├─ model(batch["img"])            ← 模型推理
       │    │    engine/validator.py#L238
       │    │
       │    ├─ postprocess(preds)             ← NMS 后处理
       │    │    detect/val.py#L133-147
       │    │    nms.non_max_suppression()
       │    │
       │    └─ update_metrics(preds, batch)   ← 逐图像评估
       │         detect/val.py#L162-219
       │         │
       │         └─ 对每张图:
       │              ├─ _prepare_batch()     ← GT 预处理 (xywh→xyxy)
       │              ├─ _prepare_pred()      ← 预测后处理
       │              │
       │              └─ _process_batch()     ← TP/FP 匹配 ← 核心!
       │                   │  detect/val.py#L302-311
       │                   │
       │                   ├─ box_iou()       ← 计算 N×M IoU 矩阵
       │                   │    metrics.py#L82-91
       │                   │    return inter / union
       │                   │
       │                   └─ match_predictions() ← 一一匹配
       │                        engine/validator.py#L296-345
       │                        对 10 个 IoU 阈值分别匹配
       │                        return (N_preds, 10) TP 矩阵
       │
       ├─3── gather_stats()                  ← 分布式汇总
       │
       └─4── get_stats() → process()         ← 最终计算 mAP
            │  metrics.py#L1140-1170
            │
            └─ ap_per_class()                ← 核心 AP 计算
                 │  metrics.py#L795-883
                 │
                 ├─ 按类别分组
                 ├─ 按置信度排序
                 ├─ 累积 TP/FP → PR 曲线
                 └─ compute_ap()              ← 101 点插值
                      metrics.py#L763-790
                      return ap, mpre, mrec
```

---

## 二、逐层源码分析

### 第 1 层：用户入口 — `yolo val`

```bash
yolo val model=yolo26n.pt data=coco8.yaml
```

CLI 入口分发到 `DetectionValidator`。

### 第 2 层：DetectionValidator.__call__()

**文件**: [ultralytics/engine/validator.py#L146-L260](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/engine/validator.py#L146-L260)

```python
@smart_inference_mode()
def __call__(self, trainer=None, model=None):
    self.init_metrics(unwrap_model(model))

    for batch_i, batch in enumerate(bar):
        # Step 1: 预处理
        batch = self.preprocess(batch)

        # Step 2: 推理
        preds = model(batch["img"], augment=augment)

        # Step 3: 后处理 (NMS)
        preds = self.postprocess(preds)

        # Step 4: 逐图像更新指标
        self.update_metrics(preds, batch)

    # Step 5: 汇总统计 → 计算 mAP
    stats = self.get_stats()
    self.finalize_metrics()
    self.print_results()
    return stats
```

**流程**：遍历 dataloader → 推理 + NMS → 逐图匹配 → 汇总计算 mAP。

### 第 3 层：init_metrics() — 初始化评估参数

**文件**: [ultralytics/models/yolo/detect/val.py#L78-L124](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/models/yolo/detect/val.py#L78-L124)

```python
class DetectionValidator(BaseValidator):
    def __init__(self, ...):
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU 阈值 [0.5, 0.55, ..., 0.95]
        self.niou = self.iouv.numel()               # 10
        self.metrics = DetMetrics()                  # 指标统计器

    def init_metrics(self, model):
        self.is_coco = "coco" in val  # 是否为 COCO 数据集
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = model.names
        self.metrics.clear_stats()
        self.confusion_matrix = ConfusionMatrix(...)
```

**关键**：`self.iouv` 对应笔记中的 `IoU 阈值从 0.5 到 0.95（步长 0.05）共 10 个值`。

### 第 4 层：update_metrics() — 逐图像匹配

**文件**: [ultralytics/models/yolo/detect/val.py#L162-L219](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/models/yolo/detect/val.py#L162-L219)

```python
def update_metrics(self, preds, batch):
    for si, pred in enumerate(preds):             # 遍历每张图
        self.seen += 1
        pbatch = self._prepare_batch(si, batch)    # GT 预处理
        predn = self._prepare_pred(pred)           # 预测后处理

        # ── 核心：TP/FP 匹配 ──
        self.metrics.update_stats({
            **self._process_batch(predn, pbatch),  # TP 矩阵
            "target_cls": cls,                     # GT 类别
            "target_img": np.unique(cls),          # 图像含哪些类
            "conf": predn["conf"].cpu().numpy(),   # 置信度
            "pred_cls": predn["cls"].cpu().numpy(),# 预测类别
            "im_name": Path(pbatch["im_file"]).name,
        })
```

对应笔记中的：**对每个类别，所有预测框按置信度降序排列，遍历每个预测框做匹配**。

#### 4.1 _prepare_batch() — GT 预处理

**文件**: [detect/val.py#L130-L150](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/models/yolo/detect/val.py#L130-L150)

```python
def _prepare_batch(self, si, batch):
    bbox = ops.xywh2xyxy(bbox) * imgsz[[1, 0, 1, 0]]  # xywh → xyxy, 缩放回原图尺寸
    return {"cls": cls, "bboxes": bbox, ...}
```

#### 4.2 _process_batch() — 核心匹配

**文件**: [detect/val.py#L302-L311](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/models/yolo/detect/val.py#L302-L311)

```python
def _process_batch(self, preds, batch):
    if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
        return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}

    iou = box_iou(batch["bboxes"], preds["bboxes"])  # N_gt × N_pred IoU 矩阵
    return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}
```

**步骤**：
1. 调用 `box_iou()` 计算 GT 与预测框的完整 IoU 矩阵
2. 调用 `match_predictions()` 做一一匹配
3. 返回 TP 矩阵 shape `(N_preds, 10)` — 每个框在 10 个 IoU 阈值下是否为 TP

### 第 5 层：match_predictions() — 一一匹配算法

**文件**: [ultralytics/engine/validator.py#L296-L345](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/engine/validator.py#L296-L345)

```python
def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
    # Dx10 矩阵，D = 检测数，10 = IoU 阈值数
    correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)

    # 类别过滤：类别不匹配的 IoU 置零
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class

    for i, threshold in enumerate(self.iouv.cpu().tolist()):
        if use_scipy:
            # 匈牙利算法（精确匹配）
            cost_matrix = iou * (iou >= threshold)
            labels_idx, detections_idx = linear_sum_assignment(-cost_matrix)
            valid = cost_matrix[labels_idx, detections_idx] > 0
            correct[detections_idx[valid], i] = True
        else:
            # 贪心匹配（默认）— 对应笔记中的一一匹配规则
            matches = np.nonzero(iou >= threshold)        # IoU ≥ 阈值
            matches = np.array(matches).T
            if matches.shape[0]:
                # 按 IoU 降序排列
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                # 每个 GT 只能匹配一次 (unique GT)
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # 每个检测只能匹配一次 (unique det)
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
```

**关键算法逻辑**（对应笔记中的 TP/FP 定义）：

```
对每个 IoU 阈值:
  1. 找到所有 IoU ≥ 阈值的 (GT, 检测) 对
  2. 按 IoU 降序排列
  3. 每个 GT 只能被匹配一次
  4. 每个检测只能匹配一个 GT
  5. 匹配上的 → TP, 未匹配的 → FP
  6. 未匹配的 GT → FN
```

### 第 6 层：DetMetrics.update_stats() — 统计收集

**文件**: [ultralytics/utils/metrics.py#L1140-L1147](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L1140-L1147)

```python
def update_stats(self, stat):
    for k in self.stats.keys():
        self.stats[k].append(stat[k])
```

统计数据收集：所有图像的 `tp`、`conf`、`pred_cls`、`target_cls`、`target_img` 累积到列表。

### 第 7 层：DetMetrics.process() — 汇总触发 AP 计算

**文件**: [ultralytics/utils/metrics.py#L1150-L1170](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L1150-L1170)

```python
def process(self, ...):
    stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}  # 拼接所有图像

    results = ap_per_class(
        stats["tp"],
        stats["conf"],
        stats["pred_cls"],
        stats["target_cls"],
        plot=plot, save_dir=save_dir, names=self.names,
    )[2:]  # 返回 [p, r, ap, f1, unique_classes]

    self.box.update(results)
    return stats
```

### 第 8 层：ap_per_class() — 按类别计算 AP

**文件**: [ultralytics/utils/metrics.py#L795-L883](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L795-L883)

```python
def ap_per_class(tp, conf, pred_cls, target_cls, ...):
    # 1. 按置信度降序排列（对应笔记：所有预测框按置信度降序）
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 2. 找到唯一类别（对应笔记：对某个类别）
    unique_classes, nt = np.unique(target_cls, return_counts=True)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c                # 当前类别的预测
        n_l = nt[ci]                     # 当前类别的 GT 数
        n_p = i.sum()                    # 当前类别的预测数

        # 3. 累积 TP/FP（对应笔记：遍历每个预测框）
        fpc = (1 - tp[i]).cumsum(0)      # FP 累积
        tpc = tp[i].cumsum(0)            # TP 累积

        # 4. 计算 PR 曲线
        recall = tpc / (n_l + eps)       # Recall = TP / (TP + FN)
        precision = tpc / (tpc + fpc)    # Precision = TP / (TP + FP)

        # 5. 对每个 IoU 阈值计算 AP
        for j in range(tp.shape[1]):     # 10 个 IoU 阈值
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
```

### 第 9 层：compute_ap() — 101 点插值

**文件**: [ultralytics/utils/metrics.py#L763-L790](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L763-L790)

```python
def compute_ap(recall, precision):
    # 1. 添加哨兵值（首尾扩展，保证插值边界完整）
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    # 2. 取右侧最大 Precision（插值核心！对应笔记：取该点右侧的最大 Precision）
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 3. 101 点插值积分（对应笔记：COCO 标准 101 点插值 AP）
    x = np.linspace(0, 1, 101)           # 101 个 Recall 点
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # 插值后求曲线下面积

    return ap, mpre, mrec
```

**等价实现**（与笔记中伪代码一致）：

```python
# 笔记中的 101 点实现：
for t in np.linspace(0, 1, 101):       # 101 个召回率阈值
    mask = recalls >= t                  # 取右侧
    p = np.max(precisions[mask]) if mask.any() else 0  # 最大 Precision
    ap += p / 101                        # 平均
```

YOLO 的 `np.maximum.accumulate(np.flip(mpre))` + `np.interp` 是实现同样逻辑的**向量化高效写法**。

---

## 三、关键数据结构

### 3.1 统计收集器

```python
# DetMetrics.stats
self.stats = {
    "tp": [],         # list of (N_preds_i, 10) bool 数组, 每张图一个
    "conf": [],       # list of (N_preds_i,) float 数组, 置信度
    "pred_cls": [],   # list of (N_preds_i,) int 数组, 预测类别
    "target_cls": [], # list of (M_i,) int 数组, GT 类别
    "target_img": [], # list of (unique_classes_i,) int 数组
}
```

最终 `np.concatenate` 后：
- `tp`: `(total_preds, 10)` — 所有预测框 × 10 个 IoU 阈值
- `conf`: `(total_preds,)` — 所有预测框置信度
- `pred_cls`: `(total_preds,)` — 所有预测框类别

### 3.2 Metric 结果存储

```python
class Metric:
    self.all_ap: shape (nc, 10)      # 每个类别 × 10 个 IoU 阈值的 AP
    self.ap_class_index: shape (nc,)  # 类别索引映射
    self.p: shape (nc,)              # 每个类别的最优 Precision
    self.r: shape (nc,)              # 每个类别的最优 Recall
```

---

## 四、笔记概念 → 源码映射

| 笔记概念 | YOLO 源码位置 | 实现 |
|---------|-------------|------|
| **IoU 阈值 0.5~0.95 × 10** | [detect/val.py#L58](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/models/yolo/detect/val.py#L58) | `torch.linspace(0.5, 0.95, 10)` |
| **按置信度降序排列** | [metrics.py#L799](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L799) | `i = np.argsort(-conf)` |
| **TP/FP 定义：每个 GT 只能匹配一次** | [validator.py#L296-345](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/engine/validator.py#L296-L345) | `match_predictions()` 贪心/匈牙利匹配 |
| **PR 曲线** | [metrics.py#L810-812](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L810-L812) | `recall=tpc/n_l, precision=tpc/(tpc+fpc)` |
| **COCO 101 点插值** | [metrics.py#L784-787](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L784-L787) | `np.linspace(0,1,101)` + `np.interp` + `np.trapz` |
| **取右侧最大 Precision** | [metrics.py#L782](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L782) | `np.flip(np.maximum.accumulate(np.flip(mpre)))` |
| **AP → mAP 平均** | [metrics.py#L985-1003](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L985-L1003) | `self.all_ap.mean(1)` → `map50`, `map` |
| **AP_small/medium/large** | [metrics.py#L1070-1100](file:///home/nvidia/wh/lesson/yolo_lesson/week1/ultralytics-main/ultralytics-main/ultralytics/utils/metrics.py#L1070-L1100) | `update_image_metrics()` 按面积分组 |
| **AR (Average Recall)** | 间接计算 | 从累积 recall 取最大值 |

---

## 五、数据流向全图

```
模型输出 (batch, 84, 8400)
  │
  └─→ postprocess (NMS)
       │
       └─→ [{bboxes: (N,4), conf: (N,), cls: (N,)}, ...]    ← 每张图
            │
            └─→ _process_batch(predn, pbatch)
                 │
                 ├─ box_iou(gt_boxes, pred_boxes)     → (M, N) IoU 矩阵
                 │
                 └─ match_predictions(cls, gt_cls, iou) → (N, 10) TP 矩阵
                      │                                  ↑每个框在10个IoU阈值下是否为TP
                      └─ 存入 stats{"tp": [], "conf": [], "pred_cls": [], ...}
                           │
                           └─ (所有图像汇总后) process()
                                │
                                └─ ap_per_class(tp, conf, pred_cls, target_cls)
                                     │
                                     ├─ 按类别分组, 按置信度排序
                                     ├─ cumsum(TP), cumsum(FP) → PR 曲线
                                     └─ compute_ap(recall, precision) → AP
                                          │
                                          └─ 101 点插值 → 最终 mAP
```

---

## 六、不同任务 / 模型的差异

| 模型 | Validator 类 | IoU 计算 | 额外匹配维度 |
|------|-------------|---------|------------|
| **YOLOv8-det** | `DetectionValidator` | `box_iou` (bbox) | 无 |
| **YOLOv8-seg** | `SegmentationValidator` | `box_iou` + `mask_iou` | 掩码 TP (tp_m) |
| **YOLOv8-pose** | `PoseValidator` | `box_iou` + `kpt_iou(OKS)` | 关键点 TP (tp_kpts) |
| **YOLOv8-obb** | `OBBValidator` | `batch_probiou` (旋转框) | 无 |

---

## 七、关键源码映射表

| 功能 | 文件 | 行号 |
|------|------|------|
| `DetectionValidator` 定义 | `ultralytics/models/yolo/detect/val.py` | L21-330 |
| `BaseValidator.__call__()` 主循环 | `ultralytics/engine/validator.py` | L146-260 |
| `DetectionValidator.init_metrics()` | `detect/val.py` | L78-124 |
| `DetectionValidator.update_metrics()` | `detect/val.py` | L162-219 |
| `DetectionValidator._process_batch()` | `detect/val.py` | L302-311 |
| `_prepare_batch()` GT 预处理 | `detect/val.py` | L130-150 |
| `postprocess()` NMS 后处理 | `detect/val.py` | L133-147 |
| `match_predictions()` 一一匹配 | `engine/validator.py` | L296-345 |
| `box_iou()` IoU 矩阵 | `utils/metrics.py` | L82-91 |
| `compute_ap()` 101 点插值 | `utils/metrics.py` | L763-790 |
| `ap_per_class()` 逐类 AP | `utils/metrics.py` | L795-883 |
| `DetMetrics.update_stats()` 统计收集 | `utils/metrics.py` | L1140-1147 |
| `DetMetrics.process()` mAP 汇总 | `utils/metrics.py` | L1150-1170 |
| `Metric` 指标存储 | `utils/metrics.py` | L895-1007 |
| `DetectionValidator.get_stats()` | `detect/val.py` | L268 |

---

## 八、一句话总结

> **YOLOv8 评估时，每张图通过 `box_iou` + `match_predictions` 得到 `(N_preds, 10)` 的 TP 矩阵，汇总所有图像后由 `ap_per_class` 按类别分别计算 PR 曲线，最后经 `compute_ap` 的 101 点插值得到每个类别的 AP，取平均即为 mAP。整个过程对应笔记中 TP/FP 一一匹配 → PR 曲线 → 插值 AP → mAP 的完整链路。**