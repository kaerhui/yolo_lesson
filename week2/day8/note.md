# Day 8 学习笔记：YOLOv5 架构拆解 + 自注意力初探

> 论文原文 + 博客教程 + 源码分析

---

## 一、YOLOv5 架构：从 YAML 配置文件到网络构建

YOLOv5 的架构完全由一个 **YAML 配置文件** (`cfg/models/v5/yolov5.yaml`) 驱动，`parse_model()` 函数 ([tasks.py#L1774](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/nn/tasks.py#L1774)) 逐行解析这个文件，把每层按 `[from, number, module, args]` 的格式组装成网络。

### 看 YAML 文件理解整体结构

打开 [yolov5.yaml](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/cfg/models/v5/yolov5.yaml) 一看就明白：

```yaml
# 参数区：告诉模型类别数和各版本缩放因子
nc: 80
scales:
  n: [0.33, 0.25, 1024]   # YOLOv5n: depth=0.33, width=0.25
  s: [0.33, 0.50, 1024]   # YOLOv5s: depth=0.33, width=0.50
  l: [1.00, 1.00, 1024]   # YOLOv5l: 全尺寸

# backbone: 从输入到特征提取
backbone:
  - [-1, 1, Conv, [64, 6, 2, 2]]     # 0-P1/2    输入→ 通道64, 6×6卷积, 步长2
  - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4    下采样到 1/4
  - [-1, 3, C3, [128]]               # 2          3个C3模块
  - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8    下采样到 1/8 (小目标检测层)
  - [-1, 6, C3, [256]]               # 4          6个C3模块
  - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16   下采样到 1/16 (中目标检测层)
  - [-1, 9, C3, [512]]               # 6          9个C3模块
  - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32   下采样到 1/32 (大目标检测层)
  - [-1, 3, C3, [1024]]              # 8          3个C3模块
  - [-1, 1, SPPF, [1024, 5]]         # 9          SPPF 多尺度池化

# head: 特征融合 + 检测头
head:
  - [-1, 1, Conv, [512, 1, 1]]       # 10   降维
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11 上采样
  - [[-1, 6], 1, Concat, [1]]        # 12   拼接P4特征
  - [-1, 3, C3, [512, False]]        # 13   融合 (P4/16-medium)
  ... (继续下采样 + 融合, 最终3个检测头 P3/P4/P5)
  - [[17, 20, 23], 1, Detect, [nc]]  # 24   Detect 检测头
```

YAML 中的 `-1` 表示"上一层输出"，正数索引表示"第 N 层的输出"。

### Backbone 逐步拆解

**从 YAML 看 backbone 的 10 层**，每一行都对应一个具体的 PyTorch 模块：

| 层 | YAML 写法 | 真实代码 | 做了什么 |
|----|----------|---------|---------|
| 0 | `Conv, [64, 6, 2, 2]` | [conv.py#L27](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/nn/modules/conv.py#L27) `Conv(3,64,k=6,s=2)` | 6×6 卷积, 通道 3→64, 空间 640→320 |
| 1 | `Conv, [128, 3, 2]` | `Conv(64,128,k=3,s=2)` | 3×3 卷积, 通道 64→128, 空间 320→160 |
| 2 | `C3, [128]` | [block.py#L327](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/nn/modules/block.py#L327) `C3(128,128,n=3)` | 3 个 Bottleneck 的 CSP 模块 |
| 3 | `Conv, [256, 3, 2]` | `Conv(128,256,k=3,s=2)` | 下采样到 80×80 (P3/8) |
| 4 | `C3, [256]` | `C3(256,256,n=6)` | 6 个 Bottleneck |
| 5 | `Conv, [512, 3, 2]` | `Conv(256,512,k=3,s=2)` | 下采样到 40×40 (P4/16) |
| 6 | `C3, [512]` | `C3(512,512,n=9)` | 9 个 Bottleneck |
| 7 | `Conv, [1024, 3, 2]` | `Conv(512,1024,k=3,s=2)` | 下采样到 20×20 (P5/32) |
| 8 | `C3, [1024]` | `C3(1024,1024,n=3)` | 3 个 Bottleneck |
| 9 | `SPPF, [1024, 5]` | [block.py#L208](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/nn/modules/block.py#L208) `SPPF(1024,1024,k=5)` | 串行 3 个 5×5 池化 |

**理解 `C3` 模块** — 两路并行，减少重复计算：

```python
# 代码位置: block.py#L327
class C3(nn.Module):
    def __init__(self, c1, c2, n=1):
        c_ = int(c2 * 0.5)           # 隐藏通道 = 一半
        self.cv1 = Conv(c1, c_, 1)   # 主路: 降维
        self.cv2 = Conv(c1, c_, 1)   # 捷径: 降维
        self.cv3 = Conv(2*c_, c2, 1) # 输出: 融合
        self.m = nn.Sequential(       # n 个 Bottleneck
            *[Bottleneck(c_, c_) for _ in range(n)])

    def forward(self, x):
        # 主路: 经过 Bottleneck 深度提取特征
        # 捷径: 直接通过, 不做复杂变换
        # concat 后再用 1×1 卷积融合
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

**CSP 的"省钱"逻辑**: 把输入通道一分为二，主路做复杂计算，捷径直接通过。梯度反向传播时只有一半通道流经 Bottleneck → 节省计算，效果反而更好。

**SPPF 的"串行复用"逻辑** — 一行代码搞懂：

```python
# 代码位置: block.py#L208
class SPPF(nn.Module):
    def forward(self, x):
        y = [self.cv1(x)]            # 原始特征
        y.append(self.m(y[-1]))      # 第一次池化, RF=5
        y.append(self.m(y[-1]))      # 第二次池化, RF=9
        y.append(self.m(y[-1]))      # 第三次池化, RF=13
        return self.cv2(torch.cat(y, 1))  # 四路 concat → 融合
```

每次池化都复用了上一次的结果，串行 3 次得到等效感受野 5, 9, 13。如果并行做 3 个不同核的池化，计算量更大，结果一样。

### Neck — 特征融合

从 YAML 的 head 部分可以看到两步操作：

1. **自上而下 (FPN)**: 高层特征上采样 → 与低层特征拼接 → 用 C3 融合
   - YAML 第 10 行: `Conv, [512, 1, 1]` — 降维到 512
   - YAML 第 11 行: `nn.Upsample` — 2× 上采样  
   - YAML 第 12 行: `Concat` — 拼接 P4 特征
   - YAML 第 13 行: `C3, [512, False]` — 融合

2. **自下而上 (PAN)**: 低层特征下采样 → 与高层特征拼接 → 融合

**双向融合的好处**: 语义信息从高层传到低层 (FPN)，定位信息从低层传到高层 (PAN)。YOLOv5 的 Detect 头从 3 个融合后的特征图 (P3/8, P4/16, P5/32) 分别做预测，分别负责小、中、大目标。

### Head — 检测头

YAML 最后一行:
```yaml
- [[17, 20, 23], 1, Detect, [nc]]  # 三个检测层
```

[head.py#L37](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/nn/modules/head.py#L37) 的 `Detect` 类接收 3 个特征图，对每个位置预测:
- 4 个回归值 (DFL 分布 → 框坐标)
- `nc` 个分类分数

### 流程图

```
输入 640×640×3
  │
  ├─ Focus (Conv 6×6, s=2)    → 320×320×64      P1/2
  ├─ Conv (3×3, s=2)          → 160×160×128     P2/4
  ├─ C3 ×3                    → 160×160×128
  ├─ Conv (3×3, s=2)          → 80×80×256       P3/8  ← 小目标
  ├─ C3 ×6                    → 80×80×256
  ├─ Conv (3×3, s=2)          → 40×40×512       P4/16 ← 中目标
  ├─ C3 ×9                    → 40×40×512
  ├─ Conv (3×3, s=2)          → 20×20×1024      P5/32 ← 大目标
  ├─ C3 ×3                    → 20×20×1024
  ├─ SPPF (k=5)               → 20×20×1024
  │
  ├─ FPN: 上采样 + 拼接融合    → 3 个尺度
  ├─ PAN: 下采样 + 拼接融合    → 3 个尺度
  │
  └─ Detect: 3 个头 → 预测框 + 类别
```

**关键**: 这些层数 (C3 ×3, ×6, ×9) 是 **YOLOv5l** 的配置。YOLOv5s 通过 `depth_multiple=0.33` 缩放到 1, 2, 3 个 Bottleneck；YOLOv5n 则更少。这就是 `parse_model()` 函数做的事 — 读取 YAML 中的 `scales` 配置，用 `depth` 和 `width` 缩放每层的参数。详见 [tasks.py#L1790-L1800](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/nn/tasks.py#L1790-L1800)。

---

## 二、CSPNet 核心思想：为什么分两路更好？

**论文**: *CSPNet: A New Backbone that can Enhance Learning Capability of CNN* (Wang et al., CVPRW 2020)
**论文链接**: [https://arxiv.org/abs/1911.11929](https://arxiv.org/abs/1911.11929)

### 先看代码，再讲道理

YOLOv5 里用的 `C3` 类就是 CSPNet 思想的实现，直接看源码 [block.py#L327](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/nn/modules/block.py#L327)：

```python
class C3(nn.Module):
    def __init__(self, c1, c2, n=1):
        c_ = int(c2 * 0.5)                    # ① 隐藏通道 = 输出的一半
        self.cv1 = Conv(c1, c_, 1, 1)         # ② 主路降维: c1 → c_
        self.cv2 = Conv(c1, c_, 1, 1)         # ③ 捷径降维: c1 → c_
        self.cv3 = Conv(2 * c_, c2, 1, 1)     # ⑥ 融合: 2*c_ → c2
        self.m = nn.Sequential(                # ④ 主路: n 个 Bottleneck 串联
            *[Bottleneck(c_, c_) for _ in range(n)])

    def forward(self, x):
        # ⑤ 两路结果拼起来，再融合
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

而 `Bottleneck` 长这样 ([block.py#L457](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/nn/modules/block.py#L457))：

```python
class Bottleneck(nn.Module):
    def __init__(self, c1, c2):
        self.cv1 = Conv(c1, c_, 1)   # 1×1 降维
        self.cv2 = Conv(c_, c2, 3)   # 3×3 提取
        self.add = True               # 有残差连接

    def forward(self, x):
        return x + self.cv2(self.cv1(x))  # 残差: x + F(x)
```

### 用一张图看懂 CSP 做了什么

```
传统残差 (Bottleneck):
  输入 128通道
    │
    ├─ 1×1 卷积 → 64通道     (计算量: 128×64×1×1×H×W)
    ├─ 3×3 卷积 → 128通道     (计算量: 64×128×3×3×H×W)
    └─ + 残差连接
    │
  输出 128通道
  → 梯度回传时，所有 128 个通道都流经完整的计算路径

CSP (C3 模块):
  输入 128通道
    │
    ├─ 分路 A (cv1) → 64通道 → 经过 n 个 Bottleneck (深度提取)
    ├─ 分路 B (cv2) → 64通道 → 直接跳过，不做复杂变换
    │
    ├─ concat(A, B) → 128通道
    └─ 1×1 卷积 (cv3) → 128通道
  → 梯度回传时，只有分路 A 的 64 个通道流经 Bottleneck
```

### 省在哪？为什么省了反而更好？

**传统残差块**的问题：梯度反向传播时，**所有输入通道**都要经过同样的计算路径。如果堆叠很多层，梯度在每一层都"重复劳动"。

**CSP 的"省"**：把输入通道一分为二，只让**一半通道**走深度提取路径（Bottleneck 串联），另一半直接抄近道。梯度就只流经一半通道。

**计算量对比**（以 128→128 为例）：

| 步骤 | 传统残差 (Bottleneck×3) | CSP (C3, n=3) |
|------|----------------------|---------------|
| 降维 | 128→64 (×1) | 两路各 128→64 (×2) |
| 深度提取 | 64→64→64→64 (3 个 Bottleneck, 所有通道都算) | 64→64→64→64 (3 个 Bottleneck, 但只有一半通道) |
| 升维 | 64→128 (×1) | 128→128 (×1) |

**关键**：CSP 的 Bottleneck 只处理 **64 个通道**（一半），而传统残差处理 **128 个通道**（全部）。所以 CSP 的计算量大约是传统残差的 **一半**。

**为什么精度反而提升？** 因为梯度路径更"干净"了。一半通道专门做深度特征提取，另一半保持原始信息，最后 concat 融合。网络学到了"专业化分工"——有的通道负责提取深度特征，有的通道负责保留原始信息。

### 结合实际 YAML 配置理解

```yaml
# yolov5.yaml 中 backbone 第 6 层:
- [-1, 9, C3, [512]]
#     ↑  ↑   ↑   ↑
#    来自上  Bottleneck  → 输出通道 512
#    一层  数量=9
```

这句话的意思是：在第 6 层，用 `C3(512, 512, n=9)` 做特征提取。内部是：
- `cv1`: 512 → 256 (一半)
- `cv2`: 512 → 256 (另一半)
- Bottleneck ×9: 256 → 256 → ... → 256 (只处理一半通道)
- `cv3`: 512 → 512 (融合)

YOLOv5l 用了 9 个 Bottleneck（因为 `depth_multiple=1.0`），YOLOv5s 只用了 `max(round(9×0.33), 1) = 3` 个。

### 总结三句话

1. **CSP 的核心**：把通道 split 成两路，一路做深度计算，一路直接抄近道
2. **省在哪**：深度计算只处理一半通道，计算量减半
3. **为什么更好**：梯度路径更干净，网络实现"专业化分工"

---

## 三、自注意力基础：从公式到代码

**论文**: *Attention Is All You Need* (Vaswani et al., NeurIPS 2017)
**论文链接**: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### 先看代码：YOLOv8 中的 Attention 实现

Ultralytics 代码库中有一个 `Attention` 类 [block.py#L1271](file:///E:/YOLO_lesson/ultralytics-main/ultralytics-main/ultralytics/nn/modules/block.py#L1271)，直接实现了自注意力机制。把这个类看懂，你就掌握了自注意力的核心。

```python
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads          # 注意力头数，比如 8
        self.head_dim = dim // num_heads     # 每头维度，比如 dim=512 → 每头 64
        self.key_dim = int(self.head_dim * attn_ratio)  # 可以压缩 key 维度省计算
        self.scale = self.key_dim ** -0.5    # 缩放因子 = 1/√d_k

        # ① 用 1×1 卷积生成 Q, K, V
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2                  # 总通道数: dim(V) + nh_kd(Q) + nh_kd(K)
        self.qkv = Conv(dim, h, 1, act=False)  # 一个卷积同时算 QKV
        self.proj = Conv(dim, dim, 1, act=False)  # 输出投影
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # 深度可分离卷积 → 位置编码

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape      # 输入: 4D 特征图 (batch, 通道, 高, 宽)
        N = H * W                  # 位置数 = 空间像素数

        qkv = self.qkv(x)          # ② 一个卷积算出 QKV 拼接
        # ③ 分割成 Q, K, V (每头独立)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        # ④ 注意力计算: (Q × 1/√d_k) × K^T
        attn = (q * self.scale).transpose(-2, -1) @ k
        # ⑤ softmax 归一化
        attn = attn.softmax(dim=-1)
        # ⑥ 加权求和: attention × V
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)           # ⑦ 输出投影
        return x
```

### 用一个超级简单的例子搞懂

假设你有一张 **4×4 的图片**，把它切成 16 个 patch（就像 16 个"位置"）。每个位置有 512 维特征。

**自注意力要回答的问题**："每个位置应该看其他哪些位置来更新自己？"

**步骤拆解**（对应上面代码里的编号）：

#### 步骤 ① - 准备 QKV

**"问、答、值"** 的关系：

| 角色 | 变量 | 含义 | 类比 |
|------|------|------|------|
| **Q** (Query) | `q` | "我在找什么？" | 你心里想的问题 |
| **K** (Key) | `k` | "我有什么？" | 网页上的标题 |
| **V** (Value) | `v` | "我给你的信息是什么？" | 网页的内容 |

#### 步骤 ②+③ — 一个卷积生成 QKV

```python
qkv = self.qkv(x)  # dim=512 → 一个 1×1 卷积，输出通道包含 QKV 三个部分
q, k, v = split(qkv)  # 切分成三份
```

为什么**一个卷积**？因为 1×1 卷积其实就是**全连接层**（每个位置独立计算）。用 `Conv(dim, h, 1)` 本质上是 `nn.Linear(dim, h)`，同时在所有空间位置上共享参数。

#### 步骤 ④ — 计算"相关性"：QK^T

```python
# 形状: (B, num_heads, key_dim, N) × (B, num_heads, N, key_dim)
# 结果: (B, num_heads, N, N)
attn = (q * self.scale).transpose(-2, -1) @ k
```

每个位置 i 的 Q 和每个位置 j 的 K 做点积（内积）。**点积越大 → 两个位置越相关**。

**缩放因子 `scale = 1/√d_k`** 的作用：如果 d_k=64，点积结果的范围大约是 [-32, 32]，softmax 之后会非常"尖锐"（几乎成了 one-hot）。缩放后范围减小到 [-4, 4]，softmax 的梯度更友好。

#### 步骤 ⑤ — softmax 归一化

```python
attn = attn.softmax(dim=-1)  # 对每行做 softmax，使每行和为 1
```

**每行**代表**一个位置对所有位置的注意力权重**，和为 1。

#### 步骤 ⑥ — 加权求和

```python
x = (v @ attn.transpose(-2, -1))  # 用注意力权重加权 V
```

**每个位置的新特征 = 所有位置 V 的加权平均**，权重就是注意力分数。

#### 步骤 ⑦ — 输出投影

```python
x = self.proj(x)  # 又一个 1×1 卷积，混合各头信息
```

### 手算冷却：3 个 token 的完整流程

```
假设输入 3 个位置, 每位置 4 维:
x = [[1,0,1,0],   # 位置 0
     [0,1,0,1],   # 位置 1
     [1,1,0,0]]   # 位置 2

假设 W_q = W_k = I (单位矩阵, 方便演示):
Q = K = x,  V = x

第 1 步: QK^T  (3×3 矩阵)
  位置0-0: 1×1+0×0+1×1+0×0 = 2
  位置0-1: 1×0+0×1+1×0+0×1 = 0
  ...
  QK^T = [[2, 0, 1],
          [0, 2, 1],
          [1, 1, 2]]

第 2 步: 除以 √d_k=√4=2
  [[1.0, 0.0, 0.5],
   [0.0, 1.0, 0.5],
   [0.5, 0.5, 1.0]]

第 3 步: softmax (每行)
  softmax([1.0, 0.0, 0.5]) = [0.39, 0.14, 0.47]
  softmax([0.0, 1.0, 0.5]) = [0.14, 0.39, 0.47]
  softmax([0.5, 0.5, 1.0]) = [0.26, 0.26, 0.48]

第 4 步: attention × V
  位置0新 = 0.39×[1,0,1,0] + 0.14×[0,1,0,1] + 0.47×[1,1,0,0]
          = [0.86, 0.61, 0.39, 0.14]
  → 位置 0 的"自己"权重最大，但吸收了位置 2 的部分信息
```

### 一个关键问题：为什么特征图不需要展平？

代码里的 `Attention` 类处理的是 **4D 特征图** (B,C,H,W)，但 Transformer 论文里处理的是 **2D 序列** (N,d)。区别在哪？

回答：**自注意力机制本身不关心数据排列方式**。`N = H×W` 把特征图展平成一维序列，QK^T 计算的是"所有位置两两之间"的相似度——不管这些位置在空间上是相邻还是相隔很远。

这就是**自注意力 vs 卷积**的核心区别之一：

| 特性 | 卷积 (Conv) | 自注意力 (Self-Attention) |
|------|-----------|------------------------|
| 感受野 | 固定大小 (3×3, 5×5) | 全局 (全图) |
| 权重 | 静态 (训练后固定) | 动态 (随输入变化) |
| 参数 | 核参数共享 | QKV 投影矩阵 |
| 计算量 | O(k²×C²×H×W) | O(N²×d) |
| 局部性 | 天生有 | 没有 (靠位置编码) |

### 多头注意力（Multi-Head Attention）

代码里的 `num_heads=8` 就是多头。为什么需要多头？

```python
# 8 个头，每个头独立计算注意力
# 头 0: 关注"颜色相似"的位置
# 头 1: 关注"空间相邻"的位置
# 头 2: 关注"语义相关"的位置
# ...
# 最后 concat 所有头的结果，用 proj 融合
```

**多头 = 多组不同的 QKV 投影矩阵**，每组学习一种"关注方式"。最终把 8 种关注方式的结果拼起来。

### 和 YOLO 的关系

YOLOv8 的某些模块（如 `PSABlock`）在特征提取中加入了 `Attention` 层，目的是在 CNN 的局部感受野基础上，补充**全局依赖**的建模能力。具体来说：

- **CNN** 擅长提取局部纹理、边缘等特征
- **Self-Attention** 擅长捕捉长距离依赖（如"远处的车和近处的车"）
- **两者结合** = 局部精度 + 全局视野

---

## 四、参考资料

- **论文**:
  - *CSPNet*: [https://arxiv.org/abs/1911.11929](https://arxiv.org/abs/1911.11929)
  - *Attention Is All You Need*: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **源码**: Ultralytics `models/common.py` (C3, SPPF)
- **博客**: [YOLOv5 网络结构完全解析](https://blog.csdn.net/weixin_44791964/article/details/120005990)
- **本日代码**: [yolov5_architecture.py](yolov5_architecture.py)