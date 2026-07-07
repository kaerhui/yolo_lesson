# Day 1 学习笔记：IoU 及其变体的数学推导与实现

> 论文原文 + 博客教程 + 代码实现 综合解读

---

## 一、核心论文速览

### 1. GIoU — Generalized Intersection over Union

**论文**: *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression* (Rezatofighi et al., CVPR 2019)
**论文链接**: [https://giou.stanford.edu/GIoU.pdf](https://giou.stanford.edu/GIoU.pdf)

**动机**: 传统 IoU 做损失函数有两个致命缺陷：
1. 预测框与 GT 无重叠时，IoU=0，梯度为0，无法优化
2. IoU 相同时，检测效果可能差异巨大（如图），IoU 无法区分不同对齐方式

**核心创新**: 引入最小外接闭包框 C（能同时包裹预测框和真实框的最小矩形）

**公式**:
```
GIoU = IoU - |C \ (A∪B)| / |C|
L_GIoU = 1 - GIoU
```

**关键性质**:
- -1 ≤ GIoU ≤ 1（IoU 是 0~1）
- 两框无重叠时，GIoU 依然可计算、有梯度
- 两框无限远时，GIoU → -1
- GIoU 总是 ≤ IoU

**局限**: 当预测框在目标框内部时，GIoU 退化为 IoU；收敛慢，尤其水平和垂直方向

---

### 2. DIoU & CIoU — Distance-IoU 与 Complete-IoU

**论文**: *Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression* (Zheng et al., AAAI 2020)
**论文链接**: [https://arxiv.org/abs/1911.08287](https://arxiv.org/abs/1911.08287)

**核心思想**: 边界框回归有三个几何要素——重叠面积、中心点距离、长宽比

#### DIoU (Distance-IoU)
```
DIoU = IoU - ρ²(b, b_gt) / c²
L_DIoU = 1 - DIoU
```
- ρ：中心点欧氏距离
- c：最小外接框对角线长度
- **优势**: 直接最小化中心点距离，收敛更快
- **解决了**: GIoU 在水平和垂直方向收敛慢的问题

#### CIoU (Complete-IoU)
```
CIoU = IoU - (ρ²(b,b_gt)/c² + αv)

其中:
v = (4/π²) * (arctan(w_gt/h_gt) - arctan(w/h))²
α = v / ((1 - IoU) + v)
```
- **在 DIoU 基础上增加**：长宽比一致性惩罚
- **解决了**: 中心点重合但长宽比不同时，DIoU 退化为 IoU

**实验结果**: 在 YOLOv3 上提升了 5.91 mAP

---

## 二、博客/教程精华

### 博客《一文搞懂IoU发展历程》核心要点

**IoU 的本质问题**:
- 梯度消失：无重叠时 IoU=0，Loss=1，梯度=0
- 方向不明确：无重叠时无法判断调整方向
- 相同 IoU 不同几何关系：无法区分对齐方式

**IoU 家族演进路线**:
```
IoU (2016) → GIoU (CVPR 2019) → DIoU (AAAI 2020) → CIoU (AAAI 2020)
                                                        ↓
                                                  EIoU / αIoU / SIoU / WIoU ...
```

**各版本核心差异**:

| 损失函数 | 重叠面积 | 中心点距离 | 长宽比 | 无重叠梯度 |
|---------|---------|-----------|-------|-----------|
| IoU Loss | ✓ | ✗ | ✗ | ✗ (梯度消失) |
| GIoU Loss | ✓ | ✗ | ✗ | ✓ (弱) |
| DIoU Loss | ✓ | ✓ | ✗ | ✓ (强) |
| CIoU Loss | ✓ | ✓ | ✓ | ✓ (强) |

### 关键直觉

**GIoU 的"先扩大后重合"问题**:
- GIoU 倾向于先扩大预测框面积来增大与 GT 的交集
- 然后再通过 IoU 项引导最大化重叠区域
- 这导致收敛需要更多迭代

**DIoU 的直接中心点回归**:
- 直接最小化中心点距离 → 收敛更快
- 不受包围框大小影响 → 梯度更稳定

**CIoU 的长宽比惩罚**:
- 当中心点重合但长宽比不同时，DIoU 无法区分
- CIoU 通过 v 项惩罚长宽比差异
- α 是自适应权重，当 IoU 大时 α 也大

---

## 三、DIoU 论文中的模拟实验

论文通过模拟实验直观展示了各 IoU 变体的收敛性能：

- **7 种长宽比**: 1:4, 1:3, 1:2, 1:1, 2:1, 3:1, 4:1
- **5000 个中心点**: 在半径 3 范围内均匀分布
- **7 种尺度**: 0.5, 0.67, 0.75, 1, 1.33, 1.5, 2

**实验结果**:
- IoU loss: 只有与 GT 有交集的框能优化（盆地区域小）
- GIoU loss: 盆地区域增大，但水平和垂直方向仍有高错误率
- DIoU loss: 盆地区域最大，收敛最快

---

## 四、DIoU-NMS 简介

DIoU-NMS 将 DIoU 引入 NMS 抑制判定:
```
s_i = 0  if DIoU(M, b_i) >= N_t
```
- 传统 NMS 仅用 IoU → 拥挤场景会误删真实目标
- DIoU-NMS 同时考虑中心点距离 → 对拥挤场景更友好

---

## 五、参考资料

- **论文 PDF**: [GIoU](https://giou.stanford.edu/GIoU.pdf) | [DIoU/CIoU](https://arxiv.org/abs/1911.08287)
- **博客**: [一文搞懂IoU发展历程](https://developer.aliyun.com/article/1100583) | [损失函数详解——IoU、GIoU、DIoU、CIoU](https://blog.csdn.net/Cupid_kl/article/details/161663327)
- **代码**: [DIoU 官方实现](https://github.com/Zzh-tju/DIoU) | [Ultralytics loss.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py)
- **知乎**: 搜索"一文读懂GIoU, DIoU, CIoU"
- **本日代码**: [iou.py](iou.py) — 包含完整实现与梯度下降回归模拟