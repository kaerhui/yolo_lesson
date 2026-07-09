# Day 8 学习笔记：YOLOv5 架构拆解 + 自注意力初探

> 论文原文 + 博客教程 + 源码分析

---

## 一、YOLOv5 整体架构

### 架构三要素

```
YOLOv5s = Backbone + Neck + Head

Backbone: Focus → Conv → C3 → Conv → C3 → Conv → C3 → SPPF
Neck:     FPN (自上而下) + PAN (自下而上)
Head:     Detect (3 个尺度, 每个 anchor-based)
```

### 关键设计点

| 组件 | 作用 | 创新点 |
|------|------|--------|
| Focus | 下采样 + 通道扩展 | 切片操作代替 stride-2 卷积，减少计算量 |
| C3 | 特征提取 | CSP 结构 split 梯度流，减少重复计算 |
| SPPF | 多尺度池化 | 串行 3 个 5×5 池化，串行复用中间结果 |
| FPN+PAN | 特征融合 | 双向融合兼顾语义和定位信息 |

---

## 二、CSPNet 论文核心思想

**论文**: *CSPNet: A New Backbone that can Enhance Learning Capability of CNN* (Wang et al., CVPRW 2020)
**论文链接**: [https://arxiv.org/abs/1911.11929](https://arxiv.org/abs/1911.11929)

### 核心发现

- 梯度流在深层网络中会重复通过大量相同计算
- 将特征图在通道维 split 为两路，一路正常计算，一路直接 shortcut
- 梯度只流过一半通道 → 计算量减半，精度反而提升

### 数学推导

```
传统残差:
  x → Conv → Conv → + → out
  梯度流经全部通道

CSP:
  split(x) → [x1, x2]
  x1 → Conv → Conv → y1
  y = concat(y1, x2) → Conv → out
  梯度只流经 x1 路径
```

---

## 三、自注意力基础

**论文**: *Attention Is All You Need* (Vaswani et al., NeurIPS 2017)
**论文链接**: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### 核心公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

| 符号 | 含义 | 维度 |
|------|------|------|
| Q | Query (查询) | N×d_k |
| K | Key (键) | N×d_k |
| V | Value (值) | N×d_v |
| d_k | 缩放因子，防止 softmax 梯度消失 | 标量 |

### 与 CNN 的联系

**当 Q、K、V 来自局部窗口时**:
- Attention 矩阵退化为带状矩阵 (band matrix)
- 每个位置只关注其邻域
- 等价于动态卷积 (卷积核权重由输入动态决定)

**区别**:
- CNN: 卷积核权重固定 (静态)
- SA: 注意力权重由输入动态生成 (动态)

### 手算小例子 (3 个 token)

```
假设 3 个 token, d_model=4:
x = [[1,0,1,0], [0,1,0,1], [1,1,0,0]]

QK^T (3×3) 归一化后:
[[0.8, 0.1, 0.1],    # token 0 关注自己最多
 [0.2, 0.6, 0.2],    # token 1 也关注自己
 [0.3, 0.3, 0.4]]    # token 2 较均匀

加权求和 = softmax(QK^T/√d) V
```

---

## 四、参考资料

- **论文**:
  - *CSPNet*: [https://arxiv.org/abs/1911.11929](https://arxiv.org/abs/1911.11929)
  - *Attention Is All You Need*: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **源码**: Ultralytics `models/common.py` (C3, SPPF)
- **博客**: [YOLOv5 网络结构完全解析](https://blog.csdn.net/weixin_44791964/article/details/120005990)
- **本日代码**: [yolov5_architecture.py](yolov5_architecture.py)