# Day 7 学习笔记：周复盘、论文精读与知识体系构建

> 复盘总结 + 知识体系串联

---

## 一、本周知识体系关系图

```
第一周: 推理评估与基础组件
│
├── 训练阶段 (Loss)
│   ├── CIoU Loss ──── 框回归损失 (Day 1)
│   └── DFL Loss ────── 框分布损失 (Day 6)
│
├── 推理阶段 (Inference)
│   ├── 输出解码 ────── 模型输出 → 框坐标 (Day 5)
│   ├── NMS ────────── 去除重复框 (Day 3)
│   │   ├── Greedy NMS
│   │   ├── Soft-NMS
│   │   └── DIoU-NMS
│   └── 评估指标 ────── 衡量检测效果 (Day 2)
│       ├── TP/FP/FN 匹配
│       ├── AP@0.5, AP@0.75, mAP@[0.5:0.95]
│       └── PR 曲线
│
├── 理论基础
│   └── 感受野 ──────── 理解检测头设计 (Day 4)
│
└── 应用工具
    └── 评估脚本 ────── 完整评估管道 (Day 5)
```

---

## 二、两篇核心论文精读笔记

### 论文 1: *Distance-IoU Loss* (Zheng et al., AAAI 2020)

**核心创新点**:
1. DIoU: 引入中心点距离归一化惩罚项 ρ²/c²
2. CIoU: 在 DIoU 基础上增加长宽比一致性惩罚 αv
3. DIoU-NMS: 将 DIoU 作为 NMS 抑制判定标准

**关键公式**:
```
DIoU  = IoU - ρ²(b, b_gt) / c²
CIoU  = IoU - (ρ²(b,b_gt)/c² + αv)
v     = (4/π²) * (arctan(w_gt/h_gt) - arctan(w/h))²
α     = v / ((1 - IoU) + v)
```

**解决的问题**:
- IoU: 无重叠时梯度消失
- GIoU: 收敛慢，水平和垂直方向效果差，包含时退化为IoU
- DIoU: 中心点重合但长宽比不同时退化为IoU

**对自身场景的启发**:
- 工业缺陷检测中，不同缺陷可能有不同长宽比
- CIoU 的长宽比惩罚可能有助于区分相似缺陷

---

### 论文 2: *Generalized Focal Loss* (Li et al., NeurIPS 2020)

**核心创新点**:
1. QFL: 将分类分数与 IoU 定位质量联合建模
2. DFL: 框回归的离散概率分布建模
3. 将分类和回归统一到同一个 focal loss 框架下

**关键公式**:
```
DFL:  L = -((yi+1 - y) * log(S_i) + (y - yi) * log(S_{i+1}))
恢复:  ŷ = Σ(i * S_i)
QFL:  L = -|y - σ|^β * ((1-y)log(1-σ) + y*log(σ))
```

**解决的问题**:
- 传统回归: Dirac delta 单点分布，信息有限
- 定位质量和分类分数: 独立训练不兼容

**对自身场景的启发**:
- 分布建模思路可用于其他回归任务
- 分布形状本身提供不确定性信息

---

## 三、关键概念速查表

| 概念 | 一句话总结 | 在哪天 |
|------|-----------|--------|
| IoU | 交集/并集，衡量两框重叠度 | Day 1 |
| GIoU | IoU + 外接闭包框惩罚，解决无重叠梯度消失 | Day 1 |
| DIoU | IoU + 中心点距离惩罚，收敛更快 | Day 1 |
| CIoU | DIoU + 长宽比惩罚，更全面 | Day 1 |
| TP/FP/FN | 检测正确/假阳性/漏检 | Day 2 |
| AP | 单个类别的 Average Precision | Day 2 |
| mAP | 所有类别 AP 的平均值 | Day 2 |
| mAP@[0.5:0.95] | 10个IoU阈值下的平均AP，COCO主指标 | Day 2 |
| NMS | 按置信度选框，高IoU抑制，去除重复框 | Day 3 |
| Soft-NMS | 得分衰减代替直接置零，保留更多候选 | Day 3 |
| DIoU-NMS | 用DIoU代替IoU做抑制判定，对密集场景友好 | Day 3 |
| 理论感受野 | 像素能看到的输入区域大小，公式递推计算 | Day 4 |
| 有效感受野 | 实际起作用的区域，中心高斯分布 | Day 4 |
| DFL | 框回归的离散概率分布建模 | Day 6 |
| QFL | 分类分数和IoU定位质量的联合建模 | Day 6 |

---

## 四、代码仓库结构

```
week1/
├── day1/
│   ├── iou.py          # IoU, GIoU, DIoU, CIoU 实现 + 梯度下降模拟
│   └── note.md         # 学习笔记
├── day2/
│   ├── map.py          # mAP 从零实现 (VOC + COCO)
│   └── note.md
├── day3/
│   ├── nms.py          # NMS, Soft-NMS, DIoU-NMS + 密集场景对比
│   └── note.md
├── day4/
│   ├── receptive_field.py  # 感受野计算器 + YOLO 分析 + 有效RF可视化
│   └── note.md
├── day5/
│   ├── eval_script.py      # 完整评估脚本 + PR曲线
│   └── note.md
├── day6/
│   ├── dfl_loss.py         # DFL 从零实现 + 分布可视化
│   └── note.md
└── day7/
    ├── review.py           # 复盘总结
    └── note.md             # 本笔记
```

---

## 五、推荐阅读顺序

1. **先看 note.md 学习笔记** → 理解论文和博客的核心内容
2. **再运行 .py 代码文件** → 动手实践加深理解
3. **对照开源代码** → 看 Ultralytics 源码中如何应用

---

## 六、下周预告

```
Day 8-9:   YOLOv8 Backbone 详解 (CSPDarknet + SPPF)
Day 10-11: YOLOv8 Neck 详解 (FPN + PAN)
Day 12-13: YOLOv8 Head 详解 (Decoupled Head + Loss)
Day 14:    周复盘与模型架构对比
```

**下周将深入模型架构**，理解 YOLOv8 的每一层设计动机和代码实现。