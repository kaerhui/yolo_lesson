# Day 13 学习笔记：Transformer 编码器-解码器与 DETR 训练流程

> DETR 论文 + 源码 + The Annotated Transformer

---

## 一、标准 Transformer 架构

**论文**: *Attention Is All You Need* (Vaswani et al., NeurIPS 2017)
**论文链接**: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### 架构图

```
Encoder:                    Decoder:
输入 → Embedding → PE      输出 → Embedding → PE
       ↓                          ↓
    Self-Attention          Masked Self-Attention
       ↓                          ↓
        + Norm                     + Norm
       ↓                          ↓
       FFN                   Cross-Attention (来自 Encoder)
       ↓                          ↓
        + Norm                     + Norm
       ↓                          ↓
       Encoder Out                FFN
                                  ↓
                                   + Norm
                                  ↓
                                输出概率
```

### Masked Self-Attention

- 每个 token 只能看到自己和之前的 token
- 实现: 上三角矩阵填充 -inf

---

## 二、KV Cache

**动机**: 自回归生成时, 每步只生成一个新 token

**原理**: 缓存之前 token 的 K, V, 只需计算新 token 的注意力

**计算量**: O(L²) → O(L)

| 序列长度 | 无 Cache | 有 Cache | 加速比 |
|---------|---------|---------|--------|
| 1 | 1 | 1 | 1× |
| 2 | 4 | 3 | 1.3× |
| 4 | 16 | 7 | 2.3× |
| 8 | 64 | 15 | 4.3× |
| 16 | 256 | 31 | 8.3× |

---

## 三、DETR 完整流程

**论文**: *DETR: End-to-End Object Detection with Transformers* (Carion et al., ECCV 2020)
**论文链接**: [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)

### 匈牙利匹配 Cost

```
Cost = λ_cls × (-s) + λ_l1 × ||box - box_gt||₁ + λ_giou × (1 - GIoU)
```

### DETR 损失函数

```
L = λ_cls × L_CE + λ_l1 × L_L1 + λ_giou × L_GIoU
```

### YOLOv8 vs DETR 对比

| 维度 | YOLOv8 | DETR |
|------|--------|------|
| 架构 | CNN + FPN | CNN + Transformer |
| 预测 | 密集 (19200 候选) | 集合 (100 query) |
| 匹配 | TAL (对齐) | 匈牙利 (二分) |
| 后处理 | NMS | 无 |
| 速度 | 实时 | 慢 |
| 与 LLM 关系 | 类似 CNN | 与 Transformer 同构 |

---

## 四、参考资料

- **论文**: *DETR*: [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)
- **源码**: [facebookresearch/detr](https://github.com/facebookresearch/detr)
- **教程**: *The Annotated Transformer* (Harvard NLP)
- **本日代码**: [transformer_detr.py](transformer_detr.py)