# Day 11 学习笔记：标签分配与损失函数中的"对齐"思想

> TAL 论文 + CLIP 对比学习 + 损失函数全景

---

## 一、TaskAlignedAssigner (TAL)

**论文**: *TOOD: Task-aligned One-stage Object Detection* (Feng et al., ICCV 2021)
**论文链接**: [https://arxiv.org/abs/2108.07755](https://arxiv.org/abs/2108.07755)

### 核心公式

```
align_metric = s^α × u^β
```

| 符号 | 含义 | 默认值 |
|------|------|--------|
| s | 分类分数 (预测类别与 GT 匹配) | — |
| u | IoU 分数 (预测框与 GT 框的 IoU) | — |
| α | 分类权重 | 1.0 |
| β | 定位权重 | 6.0 |

### 为什么 TAL 优于单纯 IoU 匹配？

| 方法 | 依据 | 问题 |
|------|------|------|
| 单纯 IoU 匹配 | 只看 IoU | 分类差的框可能被选中 |
| TAL | s^α × u^β | 同时考虑分类和定位质量 |

### 与 RLHF 的联系

```
TAL:   align_metric = s^α × u^β  → 筛选正样本
RLHF:  reward_model = human_feedback → 筛选高质量生成
```

两者都是"对齐"机制——用对齐信号筛选高质量结果。

---

## 二、YOLOv8 损失函数拆解

```
L_total = λ_box · L_CIoU + λ_cls · L_BCE + λ_dfl · L_DFL
```

| 损失项 | 公式 | 作用 |
|--------|------|------|
| L_CIoU | 1 - CIoU | 框回归 |
| L_BCE | -[y·log(p) + (1-y)·log(1-p)] | 分类 |
| L_DFL | 分布交叉熵 | 框分布建模 |

**Loss gain 配置**: YOLOv8 所有版本 (n/s/m/l/x) 使用相同的 loss gain

---

## 三、InfoNCE 对比损失

**论文**: *CLIP: Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., ICML 2021)
**论文链接**: [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

### 公式

```
L_i = -log(exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ))
```

### 温度系数 τ 的作用

| τ 值 | 效果 | 适用场景 |
|------|------|---------|
| 小 (< 0.1) | 尖锐分布, 仅关注最相似样本 | 高精度检索 |
| 中 (0.07) | 平衡正负样本 | 通用对比学习 |
| 大 (> 1) | 平滑分布, 对所有样本一视同仁 | 训练不稳定时 |

---

## 四、参考资料

- **论文**:
  - *TOOD*: [https://arxiv.org/abs/2108.07755](https://arxiv.org/abs/2108.07755)
  - *CLIP*: [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- **源码**: Ultralytics `loss.py`, `tal.py`
- **本日代码**: [tal_loss.py](tal_loss.py)