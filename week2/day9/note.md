# Day 9 学习笔记：YOLOv8 架构进化 — C2f, ELAN, 特征融合

> C2f 源码解剖 + ELAN 设计准则 + FPN/Swin Transformer 对比

---

## 一、C2f vs C3 核心差异

### 结构对比

```
C3:   split → [x1, x2] → x1 → Bottleneck×n → y1
                          x2 → (shortcut)
                          concat(y1, x2) → Conv  → out

C2f:  cv1 → [x0, x1] → x0 → Bottleneck → x1 → Bottleneck → ...
                          ↓     ↓          ↓     ↓
                          concat(x0, x1, b1_out, b2_out, ...) → cv2 → out
```

### 关键差异

| 维度 | C3 | C2f |
|------|----|-----|
| 通道分割 | 2 路 (各 50%) | n+2 路 (每步都保存) |
| 梯度路径 | 2 条 | n+2 条 |
| 信息流 | 只有最后一层 Bottleneck 输出用于 concat | 每层 Bottleneck 都保存用于 concat |
| 参数量 | 较少 | 较多 (效果好) |

---

## 二、ELAN 设计准则

**论文**: *Designing Network Design Strategies Through Gradient Path Analysis* (Wang et al., 2022)
**论文链接**: [https://arxiv.org/abs/2211.04800](https://arxiv.org/abs/2211.04800)

### 三个核心准则

1. **最短梯度路径** — 浅层特征可以直接传到输出
2. **最长梯度路径** — 深层特征经过足够多的变换
3. **扩展-混洗-合并** — 先扩展通道增加容量，再混洗增加多样性，最后合并

### C2f 如何实现 ELAN

- 最短路径: `x0` 和 `x1` 直接从 cv1 输出到 concat
- 最长路径: `b3` 经过 3 个 Bottleneck 才到 concat
- 扩展: cv1 将通道从 c1 扩展到 2c
- 合并: cv2 将所有路径融合

---

## 三、FPN → PAN 的演进

### 单向 FPN (Feature Pyramid Network)

```
P5 → 上采样 → + P4 → 上采样 → + P3
(语义信息从上往下传递)
```

**局限**: 只有语义信息从上往下，没有定位信息从下往上

### 双向 FPN (FPN + PAN)

```
P5 → 上采样 → + P4 → 上采样 → + P3
                                         ↓ 下采样
N5 ← 下采样 ← + N4 ← 下采样 ← + N3
```

**优势**: 语义信息 (自上而下) + 定位信息 (自下而上) 双向融合

---

## 四、Swin Transformer 的层次化设计

**论文**: *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (Liu et al., ICCV 2021)
**论文链接**: [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)

### Patch Merging 与 FPN 的异同

| 特性 | FPN | Patch Merging |
|------|-----|---------------|
| 作用 | 多尺度特征融合 | 下采样 |
| 操作 | 1×1 + 上采样 + 加 | 2×2 拼合 + Linear |
| 跨尺度 | 是 (不同层之间) | 否 (同一层逐步) |
| 输出分辨率 | 保持不变 | 减半 |

---

## 五、参考资料

- **论文**:
  - *ELAN*: Designing Network Design Strategies (Zhang et al., 2022)
  - *Swin Transformer*: [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)
  - *FPN*: [https://arxiv.org/abs/1612.03144](https://arxiv.org/abs/1612.03144)
- **源码**: Ultralytics `ultralytics/nn/modules/block.py` (C2f)
- **本日代码**: [c2f_fpn.py](c2f_fpn.py)