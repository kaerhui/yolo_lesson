# Day 12 学习笔记：高效网络设计与注意力机制进阶

> 模型缩放 + 多头注意力 + 位置编码

---

## 一、模型缩放策略

### YOLO 缩放公式

```
n_new = max(round(n × depth_multiple), 1)   # 深度缩放
c_new = max(round(c × width_multiple), 1)    # 宽度缩放
```

### YOLOv5 各版本缩放因子

| 版本 | depth_multiple | width_multiple | Bottleneck 数 | 通道数 |
|------|---------------|---------------|--------------|--------|
| n | 0.33 | 0.25 | 1 | 64 |
| s | 0.33 | 0.50 | 1 | 128 |
| m | 0.67 | 0.75 | 2 | 192 |
| l | 1.00 | 1.00 | 3 | 256 |
| x | 1.33 | 1.25 | 4 | 320 |

### YOLO vs EfficientNet 复合缩放

**论文**: *EfficientNet: Rethinking Model Scaling for CNNs* (Tan & Le, ICML 2019)
**论文链接**: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)

| 方法 | 缩放维度 | 方式 |
|------|---------|------|
| YOLO | depth, width | 独立缩放 |
| EfficientNet | depth, width, resolution | 统一缩放 (ϕ 系数) |

---

## 二、多头注意力 (Multi-Head Attention)

### 公式

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 每个头的作用

不同头关注不同的关系模式：

| 头 | 关注模式 | 视觉 | 语言 |
|----|---------|------|------|
| Head 0 | 局部关系 | 相邻像素 | 相邻词 |
| Head 1 | 全局关系 | 远距离依赖 | 长距离依赖 |
| Head 2 | 语义关系 | 同类物体 | 同义词 |
| Head 3 | 位置关系 | 空间布局 | 句法位置 |

---

## 三、位置编码种类

**论文**: *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021)
**论文链接**: [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

| 类型 | 实现 | 可外推 | 平移不变 |
|------|------|--------|---------|
| Sinusoidal | sin/cos 固定函数 | ✓ | ✗ |
| 可学习 | nn.Embedding | ✗ | ✗ |
| 相对位置 | 偏置表 | ✗ | ✓ |
| RoPE | 旋转矩阵 | ✓ | ✓ |

---

## 四、参考资料

- **论文**:
  - *EfficientNet*: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
  - *Swin Transformer*: [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)
  - *RoFormer*: [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- **本日代码**: [scaling_attention.py](scaling_attention.py)