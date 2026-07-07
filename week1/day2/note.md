# Day 2 学习笔记：mAP 计算原理与评价指标

> COCO 官方评估文档 + pycocotools 源码解析 + 博客教程

---

## 一、COCO 官方评估指标详解

### 1. 核心指标

COCO 提供 12 种评价指标，最重要的如下：

**AP (Average Precision)** — 主要指标
- **AP@[0.5:0.95]**: IoU 阈值从 0.5 到 0.95（步长 0.05）共 10 个值下的 AP 平均值
  - 这是 COCO 竞赛的**最终排名指标**
- **AP@0.5**: IoU 阈值 = 0.5 时的 AP（对应 VOC 标准）
- **AP@0.75**: IoU 阈值 = 0.75 时的 AP（更严格的评价标准）

**AP Across Scales** — 按目标尺度分类
- **AP_small**: 面积 < 32² 的小目标
- **AP_medium**: 32² < 面积 < 96² 的中目标
- **AP_large**: 面积 > 96² 的大目标

**AR (Average Recall)**
- AR@max=1: 每张图最多 1 个检测时的最大召回
- AR@max=10: 每张图最多 10 个检测时的最大召回
- AR@max=100: 每张图最多 100 个检测时的最大召回

### 2. COCO 数据集中目标分布

- 小目标 (small): 约 41% — 面积 < 32×32 = 1024 像素
- 中目标 (medium): 约 34% — 32² < 面积 < 96²
- 大目标 (large): 约 24% — 面积 > 96²

---

## 二、mAP 计算流程详解

### 1. TP/FP/FN 定义

给定一张图片，对某个类别：
- 所有预测框按置信度**降序**排列
- 遍历每个预测框：
  - 若与某个**未匹配**的真值框 IoU ≥ 阈值 → **TP**
  - 否则 → **FP**
- 未匹配的真值框 → **FN**

**关键**: 每个真值框只能匹配一次（一一匹配）

### 2. Precision-Recall 曲线

```
Precision = TP / (TP + FP)  → 预测框中有多少是正确的
Recall    = TP / (TP + FN)  → 真实目标中有多少被检测到了
```

按置信度阈值从高到低变化，得到一系列 (Recall, Precision) 点。

### 3. VOC 2007 标准 — 11 点插值 AP

在 Recall 的 [0.0, 0.1, ..., 1.0] 这 11 个点上：
- 取该点**右侧**的最大 Precision 值
- 对这 11 个 Precision 求平均

```python
def compute_ap_voc11(tp, fp, num_gt):
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / num_gt

    ap = 0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        p = np.max(precisions[mask]) if mask.any() else 0
        ap += p
    return ap / 11
```

### 4. COCO 标准 — 101 点插值 AP

在 Recall 的 0.00 到 1.00 等间距取 101 个点：
- 计算每个点右侧最大 Precision
- 求平均

```python
def compute_ap_coco(tp, fp, num_gt):
    # 同上，但 np.linspace(0, 1, 101)
    ...
```

---

## 三、pycocotools 源码解析

### COCOeval 类核心流程

```
COCOeval.evaluate()    → 逐图像评估
    ↓
COCOeval.accumulate() → 汇总结果
    ↓
COCOeval.summarize()  → 输出指标
```

### 关键参数

```python
class Params:
    iouThrs = np.linspace(.5, 0.95, 10)  # 10 个 IoU 阈值
    recThrs = np.linspace(0, 1, 101)      # 101 个召回率阈值
    areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
    maxDets = [1, 10, 100]                 # 最大检测数
    iouType = 'bbox'                       # 'bbox', 'segm', 'keypoints'
```

### evaluateImg 单图像评估

```
1. 过滤忽略的 GT 和超出面积范围的 GT
2. 按置信度排序检测结果
3. 对每个 IoU 阈值，计算匹配矩阵:
   - 对每个检测，寻找最佳匹配 GT
   - 记录 dtMatches (检测匹配的 GT ID) 和 gtMatches (GT 匹配的检测 ID)
```

### accumulate 汇总

```
1. 遍历所有类别、面积范围、最大检测数
2. 汇总所有图像的 dtScores
3. 计算 TP (dtm 且非 ignore) 和 FP (非 dtm 且非 ignore)
4. 计算累积 TP 和 FP → 得到 Precision 和 Recall
5. 对每个 recall 阈值取最大 precision → 得到 AP
```

---

## 四、关键概念常见问题

**Q: mAP@0.5 和 mAP@[0.5:0.95] 有什么区别？**
- mAP@0.5: 只在 IoU=0.5 时计算 AP（宽松标准）
- mAP@[0.5:0.95]: 在 IoU=0.5~0.95 的 10 个阈值上分别计算 AP 再平均（严格标准）
- 后者更能反映定位精度

**Q: 为什么 COCO 用 101 点插值而 VOC 用 11 点？**
- 101 点更精细，能更准确反映 PR 曲线形状
- 11 点插值是历史遗留标准，计算简单但不够精确

**Q: AP 和 mAP 的关系？**
- AP: 单个类别在某个 IoU 阈值下的 Average Precision
- mAP: 所有类别的 AP 求平均 = mean AP

---

## 五、参考资料

- **COCO 官方评估文档**: [https://cocodataset.org/#detection-eval](https://cocodataset.org/#detection-eval)
- **pycocotools 源码**: [cocoeval.py](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- **博客**: [深入解析COCO数据集评估工具pycocotools/cocoeval.py](https://blog.csdn.net/gitblog_00085/article/details/148487156)
- **论文**: *The PASCAL Visual Object Classes (VOC) Challenge* (Everingham et al., IJCV 2010)
- **本日代码**: [map.py](map.py) — 从零实现 TP/FP 匹配、VOC 11 点 AP、COCO 101 点 AP