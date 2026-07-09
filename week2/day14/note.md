# Day 14 学习笔记：三大范式整合与 VLM/LLM 技术栈入门

> 检测范式对比 + ViT + CLIP + 解码策略 + 学习路线图

---

## 一、三大检测范式对比

### 范式演进

```
密集预测 (YOLO) → 稀疏提议 (Faster R-CNN) → 集合预测 (DETR)
先验知识: 多 → 中 → 少
后处理:  NMS → NMS → 无
与 NLP 统一: 否 → 否 → 是
```

### 为何 Transformer 统一检测和 NLP?

| 检测 | NLP |
|------|-----|
| Object Query | Token Embedding |
| 集合预测 | 序列生成 |
| 匈牙利匹配 | 交叉熵 |
| 交叉注意力 | 交叉注意力 |

统一框架: **序列到序列** / **集合到序列**

---

## 二、ViT (Vision Transformer)

**论文**: *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., ICLR 2021)
**论文链接**: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

### 核心流程

```
图像 → Patch Embed → [CLS] + Patch Tokens + Position Embed → Transformer → CLS 分类
```

| 组件 | 说明 |
|------|------|
| Patch Embed | 16×16 卷积, 将图像分成固定大小的 patch |
| [CLS] Token | 可学习的分类 token, 最终用于分类 |
| Position Embed | 可学习的位置编码 |
| Transformer | 标准 Encoder |

---

## 三、CLIP (Contrastive Language-Image Pre-training)

**论文**: *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., ICML 2021)
**论文链接**: [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

### 架构

```
Image Encoder: ViT / CNN → 图像嵌入
Text Encoder:  Transformer → 文本嵌入
对比学习:      InfoNCE → 拉近配对的图文嵌入
```

### BLIP-2 的 Q-Former

**论文**: *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models* (Li et al., 2023)
**论文链接**: [https://arxiv.org/abs/2301.05497](https://arxiv.org/abs/2301.05497)

- Q-Former 在图像编码器和 LLM 之间桥接
- 可学习的 query 从图像特征中提取视觉 token
- 视觉 token 输入 LLM 进行多模态理解

---

## 四、解码策略

| 策略 | 确定性 | 多样性 | 质量 |
|------|--------|--------|------|
| Greedy | ✓ | ✗ | 一般 |
| Top-k | ✗ | 中 | 好 |
| Top-p (Nucleus) | ✗ | 高 | 最好 |

---

## 五、学习路线图

```
Week 1: 推理评估基础 ✓
Week 2: YOLO 架构深潜 + Transformer 基础 ✓
Week 3: YOLO 压缩优化 (剪枝/量化/蒸馏)
Week 4: VLM 入门 (ViT/CLIP/BLIP-2)
Week 5: LLM 基础 (GPT/指令微调)
Week 6: VLM 前沿 (LLaVA/Qwen-VL)
Week 7+: 项目实战
```

---

## 六、参考资料

- **论文**:
  - *ViT*: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
  - *CLIP*: [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
  - *BLIP-2*: [https://arxiv.org/abs/2301.05497](https://arxiv.org/abs/2301.05497)
  - *LLaVA*: [https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
- **工具**: Hugging Face `transformers`
- **本日代码**: [paradigm_vlm.py](paradigm_vlm.py)