# Day 3 学习笔记：非极大值抑制 (NMS) 算法族

> Soft-NMS 论文原文 + DIoU-NMS 原理 + 博客教程

---

## 一、传统 Greedy NMS

### 算法流程

```
1. 输入: 检测框 B = [b1, ..., bN], 得分 S = [s1, ..., sN], IoU 阈值 Nt
2. 按得分降序排序
3. 取最高分框 M，加入保留集 D
4. 计算其余框与 M 的 IoU
5. 移除 IoU > Nt 的框
6. 重复 3-5 直到 B 为空
7. 返回 D
```

**数学表达**:
```
s_i = s_i,  if IoU(M, b_i) < Nt
s_i = 0,    if IoU(M, b_i) >= Nt
```

### 缺陷

1. **密集目标漏检**: 两个真实目标高度重叠时，得分低的那一个会被直接删除
2. **暴力抑制**: 高 IoU 框分数直接置零，没有"缓冲"

---

## 二、Soft-NMS

**论文**: *Soft-NMS — Improving Object Detection With One Line of Code* (Bodla et al., ICCV 2017)
**论文链接**: [https://arxiv.org/abs/1704.04503](https://arxiv.org/abs/1704.04503)

### 核心思想

不是暴力地将高 IoU 框得分置零，而是用函数**衰减**其得分。

**论文标题说"一行代码的改进"**，对比传统 NMS 的伪代码，只改了一行：
```
# 传统 NMS:
if IoU >= Nt:  S[j] = 0

# Soft-NMS:
S[j] = S[j] * f(IoU)   # 用乘法衰减代替直接置零
```

### 两种衰减方式

**线性衰减** (不连续):
```
s_i = s_i * (1 - IoU(M, b_i)),  if IoU >= Nt
```

**高斯衰减** (连续，推荐使用):
```
s_i = s_i * exp(-IoU(M, b_i)² / σ)
```

**高斯衰减的优势**:
- 连续函数，得分不会出现"断层"
- 无重叠时无惩罚，高重叠时强惩罚
- 实践中更常用，σ 默认 0.5

### 效果

- PASCAL VOC 2007: +1.7% AP
- MS-COCO: +1.1%~1.3% AP
- 无需重新训练，可直接替换 NMS 模块
- 计算复杂度与 NMS 相同

---

## 三、DIoU-NMS

**来源**: DIoU 论文中同时提出了 DIoU-NMS

### 核心改进

传统 NMS 仅用 IoU 判定，DIoU-NMS 使用 DIoU 替代 IoU：

```
s_i = 0  if DIoU(M, b_i) >= Nt

其中:
DIoU = IoU - ρ²(b, b_gt) / c²
```

### 为什么更好？

- 传统 NMS: 两个框 IoU 很大 → 直接抑制 → 拥挤场景漏检
- DIoU-NMS: 即使 IoU 很大，如果中心点距离远，DIoU 会较小 → 不会被抑制
- 对**拥挤场景**（人群、密集小目标）效果显著

---

## 四、算法对比总结

| 特性 | Greedy NMS | Soft-NMS | DIoU-NMS |
|------|-----------|---------|---------|
| 抑制方式 | 直接置零 | 得分衰减 | 直接置零 |
| 判定标准 | IoU | IoU | DIoU (IoU+中心点距离) |
| 密集场景 | 差（漏检） | 较好 | 最好 |
| 计算复杂度 | O(N²) | O(N²) | O(N²) |
| 额外参数 | 无 | σ (高斯) | 无 |
| 可插拔性 | 默认 | 无需重训 | 无需重训 |

---

## 五、工业缺陷检测场景建议

**问题**: 两个缺陷非常靠近时，应该用哪种 NMS？

**推荐排序**:
1. **DIoU-NMS** — 最佳
   - 能区分中心点不同的密集目标
   - 即使 IoU 很高，中心点不同就不抑制
2. **Soft-NMS** — 次之
   - 得分衰减而非直接删除
   - 保留更多候选框供后续处理
3. **Greedy NMS** — 最差
   - 高 IoU 重叠会被直接删除

---

## 六、参考资料

- **论文**:
  - *Soft-NMS — Improving Object Detection With One Line of Code*: [https://arxiv.org/abs/1704.04503](https://arxiv.org/abs/1704.04503)
  - *Distance-IoU Loss* (DIoU-NMS 章节): [https://arxiv.org/abs/1911.08287](https://arxiv.org/abs/1911.08287)
- **源码**:
  - YOLOv5/v8: `utils/general.py` 的 `non_max_suppression`
  - PyTorch: `torchvision.ops.nms`
  - Soft-NMS 官方代码: [http://bit.ly/2nJLNMu](http://bit.ly/2nJLNMu)
- **博客**: [Soft NMS 详解](https://blog.csdn.net/weixin_34114823/article/details/92950087)
- **本日代码**: [nms.py](nms.py) — 完整实现 + 密集场景对比可视化