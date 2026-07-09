"""
Day 14 — 三大范式整合与 VLM/LLM 技术栈入门
===========================================
内容:
  1. 检测范式对比全景表
  2. 范式演进图
  3. 小型 ViT 实现与训练
  4. CLIP 双塔结构演示
  5. 文本生成解码策略 (greedy, top-k, top-p)
  6. 从 YOLO → VLM 的学习路线图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ============================================================
# 1. 检测范式对比全景表
# ============================================================
def print_paradigm_comparison():
    """三大检测范式对比"""
    print("[检测范式对比全景图]")
    print("=" * 80)
    
    print(f"{'维度':<18} {'YOLO (密集预测)':<30} {'Faster R-CNN (稀疏提议)':<30} {'DETR (集合预测)':<30}")
    print("-" * 108)
    print(f"{'特征提取':<18} {'Backbone + FPN':<30} {'Backbone + FPN':<30} {'Backbone + Encoder':<30}")
    print(f"{'区域提议':<18} {'网格 + Anchor':<30} {'RPN (区域提议网络)':<30} {'Object Query (可学习)':<30}")
    print(f"{'预测头':<18} {'靠 Conv 解耦头':<30} {'RoI Pooling + Head':<30} {'Transformer Decoder':<30}")
    print(f"{'后处理':<18} {'NMS':<30} {'NMS':<30} {'匈牙利匹配 (无 NMS)':<30}")
    print(f"{{'损失函数':<18}} {'BCE + CIoU + DFL':<30} {'CE + L1 + SmoothL1':<30} {'CE + L1 + GIoU':<30}")
    print(f"{'推理速度':<18} {'最快 (实时)':<30} {'慢 (区域提议瓶颈)':<30} {'较慢 (Transformer)':<30}")
    print(f"{'与 Transformer 关系':<18} {'前 Transformer 时代':<30} {'前 Transformer 时代':<30} {'后 Transformer 时代':<30}")
    
    print(f"\n  范式演进图:")
    print(f"  密集预测 (YOLO) → 稀疏提议 (Faster R-CNN) → 集合预测 (DETR)")
    print(f"  先验知识: 多 → 中 → 少")
    print(f"  后处理:  NMS → NMS → 无")
    print(f"  与 NLP 统一: 否 → 否 → 是 (序列到序列)")
    
    print(f"\n  为何 Transformer 能统一检测和 NLP?")
    print(f"  检测: 集合预测 (set of objects) → 序列 (sequence)")
    print(f"  NLP:  序列 (sequence) → 序列 (sequence)")
    print(f"  统一框架: '序列到序列' 或 '集合到序列'")


# ============================================================
# 2. 小型 ViT 实现
# ============================================================
class PatchEmbed(nn.Module):
    """图像分块 → 线性嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x


class ViTBlock(nn.Module):
    """ViT 编码器块"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MiniViT(nn.Module):
    """
    小型 Vision Transformer (ViT-Tiny 风格)。
    
    流程: 图像 → 分块 → 线性嵌入 + CLS token + 位置编码 → Transformer → 分类
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=192, depth=6, num_heads=3, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, E]
        
        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        # 取 CLS token 分类
        return self.head(x[:, 0])


def test_mini_vit():
    """测试 Mini ViT 的前向传播"""
    print("\n[小型 ViT 实现]")
    print("=" * 60)
    
    model = MiniViT(img_size=32, patch_size=4, embed_dim=192, 
                    depth=6, num_heads=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    
    print(f"  Input:  [2, 3, 32, 32]")
    print(f"  Patch 大小: 4×4, 每张图 {model.patch_embed.num_patches} 个 patch")
    print(f"  Embed dim: {192}")
    print(f"  Transformer 深度: {6} 层")
    print(f"  Output: {tuple(logits.shape)}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  ✓ ViT 前向传播正常")


# ============================================================
# 3. CLIP 双塔结构
# ============================================================
class CLIPDemo(nn.Module):
    """
    CLIP 双塔结构 (简化版)。
    
    Image Encoder: ViT (或 CNN)
    Text Encoder:  Transformer
    对比学习: 使配对的图文嵌入相似度 > 非配对
    """
    def __init__(self, embed_dim=256, vocab_size=50, text_len=10):
        super().__init__()
        # 图像编码器 (简化: 用 CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
        )
        # 文本编码器 (简化: 用 Transformer)
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_pos = nn.Parameter(torch.randn(1, text_len, embed_dim))
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        
        # 温度系数
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, tokens):
        x = self.text_embed(tokens) + self.text_pos[:, :tokens.shape[1], :]
        x = self.text_encoder(x)
        x = x.mean(dim=1)  # 平均池化
        return self.text_proj(x)

    def forward(self, images, tokens):
        # 获取嵌入
        img_emb = F.normalize(self.encode_image(images), dim=-1)
        text_emb = F.normalize(self.encode_text(tokens), dim=-1)
        
        # 相似度矩阵
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(img_emb, text_emb.T)
        
        return logits


def test_clip():
    """测试 CLIP 双塔结构"""
    print("\n[CLIP 双塔结构演示]")
    print("=" * 60)
    
    B = 4
    embed_dim = 64
    model = CLIPDemo(embed_dim=embed_dim, vocab_size=50, text_len=10)
    
    # 模拟图像和文本
    images = torch.randn(B, 3, 64, 64)
    tokens = torch.randint(1, 50, (B, 8))
    
    logits = model(images, tokens)
    
    print(f"  图像: [{B}, 3, 64, 64]")
    print(f"  文本: [{B}, 8] (token 索引)")
    print(f"  相似度矩阵: {tuple(logits.shape)}")
    print(f"  对角线 (正样本): {logits.diag().detach().numpy().round(2)}")
    print(f"  非对角线 (负样本): {logits[0, 1:].detach().numpy().round(2)}")
    print(f"  ✓ 正样本相似度 > 负样本相似度 (经过训练后)")
    
    print(f"\n  CLIP 架构:")
    print(f"    Image: CNN → {embed_dim}d") 
    print(f"    Text:  Transformer → {embed_dim}d")
    print(f"    Loss:  InfoNCE (对比损失, Day 11)")
    print(f"  → 无监督学习, 只需图文对")


# ============================================================
# 4. 文本生成解码策略
# ============================================================
def demonstrate_decoding_strategies():
    """演示贪心、top-k、top-p 解码策略"""
    print("\n[文本生成解码策略]")
    print("=" * 60)
    
    # 模拟词汇表 logits
    vocab_size = 10
    torch.manual_seed(42)
    logits = torch.randn(1, 1, vocab_size)
    
    print(f"  Logits (模拟): {logits[0, 0].detach().numpy().round(2)}")
    print(f"  Probabilities: {F.softmax(logits[0, 0], dim=-1).detach().numpy().round(3)}")
    
    # 1. Greedy
    probs = F.softmax(logits, dim=-1)
    greedy = probs.argmax(dim=-1)
    print(f"\n  1. Greedy 解码: token {greedy.item()} (概率 {probs[0, 0, greedy].item():.3f})")
    print(f"     特点: 简单, 但可能重复、缺乏多样性")
    
    # 2. Top-k 采样
    k = 3
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    print(f"\n  2. Top-{k} 采样:")
    for i in range(k):
        print(f"     Token {topk_indices[0, 0, i].item()}: {topk_probs[0, 0, i].item():.3f}")
    print(f"     特点: 只从 top-k 里采样, 避免低概率 token")
    
    # 3. Top-p (nucleus) 采样
    p = 0.9
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum <= p
    # 至少保留一个
    if not mask.any():
        mask[..., 0] = True
    nucleus_probs = sorted_probs[mask]
    nucleus_indices = sorted_indices[mask]
    print(f"\n  3. Top-p (p={p}) 采样:")
    for i in range(len(nucleus_indices)):
        print(f"     Token {nucleus_indices[i].item()}: {nucleus_probs[i].item():.3f}")
    print(f"     特点: 自适应截断, 概率分布集中时只选少数, 分散时选更多")
    
    print(f"\n  总结:")
    print(f"  Greedy:  确定性, 质量稳定但缺乏多样性")
    print(f"  Top-k:   限制候选数, 简单可控")
    print(f"  Top-p:   自适应截断, 更自然")


# ============================================================
# 5. 学习路线图
# ============================================================
def print_roadmap():
    """从 YOLO → VLM → LLM 的学习路线图"""
    print("\n[从 YOLO → VLM → LLM 学习路线图]")
    print("=" * 70)
    
    roadmap = """
    Week 1: 推理评估基础 (已学)
    Week 2: YOLO 架构深潜 + Transformer 基础 (当前)
    
    Week 3: (YOLO 压缩优化)
      ├── 剪枝、量化、蒸馏
      ├── TensorRT 部署
      └── 边缘端推理优化
    
    Week 4: (VLM 入门)
      ├── ViT 深入 + CLIP 微调
      ├── BLIP-2 / Q-Former
      └── 图文检索 + 多模态理解
    
    Week 5: (LLM 基础)
      ├── GPT 架构详解
      ├── 自回归生成 + 指令微调
      └── Hugging Face Transformers
    
    Week 6: (VLM 前沿)
      ├── LLaVA / Qwen-VL
      ├── 视觉 token 与语言 token 融合
      └── 多模态推理
    
    Week 7+: (项目实战)
      ├── 工业缺陷检测 + 多模态辅助
      ├── 自定义数据集训练
      └── 模型部署
    """
    print(roadmap)


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Day 14: 三大范式整合与 VLM/LLM 技术栈入门")
    print("=" * 70)
    
    print_paradigm_comparison()
    test_mini_vit()
    test_clip()
    demonstrate_decoding_strategies()
    print_roadmap()
    
    print("\n" + "=" * 70)
    print("第二周总结:")
    print("  Day  8: YOLOv5 架构 + Self-Attention")
    print("  Day  9: C2f/C3 + ELAN + FPN")
    print("  Day 10: Anchor-free + Object Query")
    print("  Day 11: TAL + InfoNCE")
    print("  Day 12: 模型缩放 + MHA + 位置编码")
    print("  Day 13: Mini Transformer + DETR")
    print("  Day 14: 范式对比 + ViT + CLIP + 解码策略")
    print("=" * 70)