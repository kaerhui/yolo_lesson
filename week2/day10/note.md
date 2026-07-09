# Day 10 学习笔记：检测头与解码器 — 从 Anchor 到 Query

> YOLOv5/v8 解码 + DETR Object Query 入门

---

## 一、Anchor-based 解码 (YOLOv5)

### 公式

```
b_x = (σ(t_x) + c_x) × stride
b_y = (σ(t_y) + c_y) × stride
b_w = (a_w × exp(t_w)) × stride
b_h = (a_h × exp(t_h)) × stride
```

| 符号 | 含义 | 值域 |
|------|------|------|
| t_x, t_y | 模型预测的中心偏移 | (-∞, ∞) → sigmoid → (0, 1) |
| t_w, t_h | 模型预测的宽高缩放 | (-∞, ∞) → exp → (0, ∞) |
| c_x, c_y | 网格坐标 | 整数 (0 ~ grid_size-1) |
| a_w, a_h | anchor 先验尺寸 | 聚类得到 |
| stride | 下采样倍数 | 8, 16, 32 |

### 局限

- 需要 k-means 聚类 anchor 先验
- 对训练数据分布敏感
- 不同数据集需要重新聚类

---

## 二、Anchor-free 解码 (YOLOv8)

### 公式

```
直接回归四条边到网格中心的距离:
  l, t, r, b

使用 DFL 将离散分布转换为连续值:
  ŷ = Σ(i × softmax(logits_i))

最终坐标:
  x1 = c_x - l,  y1 = c_y - t
  x2 = c_x + r,  y2 = c_y + b
```

**优势**: 无需 anchor 先验, 泛化性更好

---

## 三、DETR 的 Object Query

**论文**: *DETR: End-to-End Object Detection with Transformers* (Carion et al., ECCV 2020)
**论文链接**: [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)

### 核心概念

| 概念 | 说明 | 类似物 |
|------|------|--------|
| Object Query | 可学习嵌入, 每个代表一个潜在目标 | LLM 的 Token Embedding |
| 交叉注意力 | Query 从图像特征中提取信息 | 注意力机制 |
| 集合预测 | 一次输出所有目标的集合 | 序列生成 |
| 匈牙利匹配 | 预测与 GT 的二分匹配 | 与 NMS 不同的后处理 |

### YOLO vs DETR 思维对比

| 维度 | YOLO | DETR |
|------|------|------|
| 预测方式 | 密集预测 (每个网格预测) | 集合预测 (固定数量 query) |
| 后处理 | 需要 NMS | 不需要 NMS (匈牙利匹配) |
| 先验知识 | Anchor 先验 | 无 |
| 与 LLM 关系 | 类似"卷积" | 类似"Transformer" |

---

## 四、参考资料

- **论文**:
  - *DETR*: [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)
  - *ViT*: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- **源码**:
  - Ultralytics `ultralytics/nn/modules/head.py` (Detect)
  - DETR 官方: `facebookresearch/detr`
- **本日代码**: [detection_head.py](detection_head.py)