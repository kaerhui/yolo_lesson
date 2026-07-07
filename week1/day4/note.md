# Day 4 学习笔记：感受野 (Receptive Field) 理论与分析

> 论文原文 + 博客教程 + YOLO 检测头分析

---

## 一、理论感受野 (Theoretical RF)

### 定义

某层输出特征图上的一个像素点能"看到"的输入图像的区域大小。

### 逐层计算公式

从前往后迭代计算：

```
r_out = r_in + (k_eff - 1) * j_in
j_out = j_in * s
start_out = start_in + (k_eff - 1) / 2 * j_in
```

其中：
- `r`: 感受野大小
- `j`: 相邻像素在输入图上的距离 (jump)
- `k_eff`: 有效卷积核大小
- `s`: 步长

**有效卷积核**（考虑空洞卷积）:
```
k_eff = k + (k - 1) * (d - 1)
```
其中 `d` 是空洞率 (dilation rate)

### 代码示例

```python
def compute_rf(layers, input_size=None):
    r, j = 1, 1
    for name, k, s, d in layers:
        k_eff = k + (k - 1) * (d - 1)
        r = r + (k_eff - 1) * j
        j = j * s
        print(f"{name}: RF={r}, Jump={j}")
    return r, j
```

---

## 二、有效感受野 (Effective RF)

**论文**: *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks* (Luo et al., NeurIPS 2016)

### 关键发现

1. **理论 RF ≠ 有效 RF**
   - 理论 RF: 通过公式计算的最大感知区域
   - 有效 RF: 实际起作用的区域，中心呈**高斯分布**
   - 有效 RF 通常只有理论 RF 的 1/3 ~ 1/2

2. **中心像素权重最大**
   - 输入像素对输出神经元的影响从中心向外递减
   - 近似高斯分布

3. **可视化方法**
   - 反向传播：在输出层中心设置梯度为 1，反向传播到输入层
   - 观察输入层的梯度分布 → 即为有效感受野

### 影响有效 RF 的因素

- **网络深度**: 越深，有效 RF 越大
- **卷积核大小**: 大核扩大 RF
- **空洞卷积**: 在不增加参数的情况下扩大 RF
- **跳跃连接**: ResNet 等结构影响 RF 分布

---

## 三、空洞卷积对 RF 的影响

**空洞卷积** (Dilated/Atrous Convolution) 在不增加参数和计算量的情况下扩大感受野。

```
k_eff = k + (k - 1) * (d - 1)
```

示例: 3×3 卷积, d=2 → 有效核尺寸 = 3 + 2*1 = 5

**应用**: DeepLab 系列语义分割、RFB 模块等

---

## 四、YOLO 检测头感受野分析

### FPN 多尺度特征

YOLOv5/v8 使用 FPN+PAN 结构，三个检测头：

| 检测头 | 下采样倍数 | 特征图大小 | 感受野 | 负责目标 |
|-------|-----------|-----------|-------|---------|
| P3/8  | 8× | 大 (80×80) | 小 | 小目标 |
| P4/16 | 16× | 中 (40×40) | 中 | 中目标 |
| P5/32 | 32× | 小 (20×20) | 大 | 大目标 |

### 为什么小特征图负责大目标？

- **小特征图** (P5/32): 下采样倍数高 → 每个像素感受野大 → 能"看到"大范围 → 适合大目标
- **大特征图** (P3/8): 下采样倍数低 → 每个像素感受野小 → 保留细节 → 适合小目标
- 这就是 FPN 多尺度特征融合的**核心动机**

### 简化计算示例

以 YOLOv5s 为参考，P5/32 检测头的感受野大致计算：

```
Layer           k   s   RF     Jump
Focus/Conv      6   2   5      2
Conv1           3   2   9      4
CSP1            3   1   17     4
Conv2           3   2   25     8
CSP2            3   1   41     8
Conv3           3   2   57     16
CSP3            3   1   89     16
Conv4           3   2   121    32
CSP4            3   1   185    32
Neck_Conv       3   1   249    32
```

P5 检测头 RF ≈ 249，对于 640×640 输入，这个感受野足以覆盖大目标。

---

## 五、感受野与检测任务的关系

| 目标大小 | 需要的 RF | 使用的特征图 | 特点 |
|---------|----------|-------------|------|
| 小目标 | 小 | P3/8 (高分辨率) | 感受野小，但空间信息丰富 |
| 中目标 | 中 | P4/16 | 折中 |
| 大目标 | 大 | P5/32 (低分辨率) | 感受野大，语义信息丰富 |

**多尺度特征融合 (FPN/PAN)** 就是为了同时兼顾大小目标检测。

---

## 六、参考资料

- **论文**:
  - *Understanding the Effective Receptive Field in Deep CNNs* (Luo et al., NeurIPS 2016)
  - *Receptive Field Block Net for Accurate and Fast Object Detection* (Liu et al., ECCV 2018)
- **工具**: [pytorch-receptive-field](https://github.com/Fangyh09/pytorch-receptive-field) | [google-research/receptive_field](https://github.com/google-research/receptive_field)
- **博客**: 《Computing Receptive Fields of Convolutional Neural Networks》
- **本日代码**: [receptive_field.py](receptive_field.py) — RF 计算器 + YOLO 分析 + 有效 RF 可视化