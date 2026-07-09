"""
Day 13 — Transformer 编码器-解码器与 DETR 训练流程
===================================================
内容:
  1. 标准 Transformer Encoder-Decoder 实现
  2. Mini Transformer 序列复制任务
  3. KV Cache 理解
  4. DETR 匈牙利匹配实现
  5. DETR 损失函数
  6. YOLOv8 vs DETR 可视化对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ============================================================
# 1. 标准 Transformer Encoder-Decoder
# ============================================================
class TransformerEncoder(nn.Module):
    """Transformer 编码器层"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerDecoder(nn.Module):
    """Transformer 解码器层 (含交叉注意力)"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # Masked 自注意力
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + attn_out)
        # 交叉注意力
        attn_out, _ = self.cross_attn(x, memory, memory, attn_mask=src_mask)
        x = self.norm2(x + attn_out)
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class MiniTransformer(nn.Module):
    """
    迷你 Transformer: Encoder + Decoder
    
    用于序列到序列任务 (如复制序列)。
    """
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=256, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = self._create_pos_enc(100, d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoder(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _create_pos_enc(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * 
                       -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, src, tgt):
        B, L_src = src.shape
        B, L_tgt = tgt.shape
        
        # Embedding + 位置编码
        src_emb = self.embed(src) + self.pos_enc[:, :L_src, :].to(src.device)
        tgt_emb = self.embed(tgt) + self.pos_enc[:, :L_tgt, :].to(tgt.device)
        
        # 因果掩码 (Decoder 看不到未来 token)
        tgt_mask = torch.triu(torch.full((L_tgt, L_tgt), float('-inf')), diagonal=1)
        tgt_mask = tgt_mask.to(src.device)
        
        # Encoder
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory)
        
        # Decoder
        out = tgt_emb
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask=tgt_mask)
        
        return self.out_proj(out)


def test_sequence_copy():
    """测试 Mini Transformer 的序列复制任务"""
    print("[Mini Transformer — 序列复制任务]")
    print("=" * 60)
    
    vocab_size = 10
    seq_len = 5
    batch_size = 4
    
    # 生成随机序列
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    # 目标: 复制输入序列, 需要 <BOS> 和 <EOS>
    bos = torch.zeros(batch_size, 1, dtype=torch.long)
    eos = torch.zeros(batch_size, 1, dtype=torch.long)
    tgt_in = torch.cat([bos, src[:, :-1]], dim=1)  # teacher forcing 输入
    tgt_out = src  # 预测目标: 复制输入
    
    model = MiniTransformer(vocab_size, d_model=32, num_heads=2, d_ff=64, num_layers=2)
    logits = model(src, tgt_in)
    
    print(f"  Source: [{batch_size}, {seq_len}]  (值域 1~{vocab_size-1})")
    print(f"  Target: [{batch_size}, {seq_len}]")
    print(f"  Logits: {tuple(logits.shape)}")
    
    # 推理: 自回归生成
    model.eval()
    with torch.no_grad():
        # 从 <BOS> 开始
        generated = torch.zeros(batch_size, 1, dtype=torch.long)
        for i in range(seq_len):
            out = model(src, generated)
            next_token = out[:, -1:].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
    
    print(f"  Generated: {tuple(generated.shape)}")
    print(f"  Original:  {src[0].tolist()}")
    print(f"  Generated: {generated[0, 1:].tolist()}")
    match = (generated[:, 1:] == src).all(dim=1).float().mean().item()
    print(f"  ✓ 复制准确率: {match * 100:.0f}%" if match > 0.5 else f"  ~ 复制准确率: {match * 100:.0f}% (需要训练)")


# ============================================================
# 2. KV Cache 理解
# ============================================================
def explain_kv_cache():
    """
    KV Cache: 推理时缓存 K 和 V 矩阵, 避免重复计算。
    
    自回归生成时, 每步只生成一个新 token。
    新 token 的 Q 变化, 但之前 token 的 K, V 不变。
    → 缓存 K, V, 只需计算新 token 的注意力。
    """
    print("\n[KV Cache 理解]")
    print("=" * 60)
    
    explanation = """
    自回归推理 (逐个 token 生成):
    
    Step 1: 输入 [a] → 计算 Q₁, K₁, V₁ → 输出 a'
    Step 2: 输入 [a, b] → 计算 Q₂, K₂, V₂ → 输出 b'
    
    无 KV Cache 时:
      Step 2 需要重新计算 K₁, V₁, 以及 2 个 token 的注意力
    
    有 KV Cache 时:
      Step 1 缓存 K₁, V₁
      Step 2 只需计算新 token 的 K₂, V₂, 与缓存拼接
      注意力 = softmax(Q₂ @ [K₁, K₂]ᵀ / √d) @ [V₁, V₂]
    
    计算量: O(L²) → O(L) (L 为序列长度)
    """
    print(explanation)
    
    # 模拟演示
    L = [1, 2, 4, 8, 16]
    print(f"  计算量对比 (相对值):")
    print(f"  {'Seq Len':<10} {'No Cache':<16} {'With Cache':<16} {'Speedup':<10}")
    print(f"  {'-'*10} {'-'*16} {'-'*16} {'-'*10}")
    for l in L:
        no_cache = l ** 2
        with_cache = 2 * l - 1  # 近似
        speedup = no_cache / with_cache
        print(f"  {l:<10} {no_cache:<16} {with_cache:<16} {speedup:<10.1f}x")


# ============================================================
# 3. DETR 匈牙利匹配
# ============================================================
class HungarianMatcher:
    """
    DETR 的匈牙利匹配器。
    
    构建 cost matrix 并在预测和 GT 之间进行二分匹配。
    Cost = λ_cls × (-分类分数) + λ_l1 × L1 距离 + λ_giou × GIoU 距离
    """
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def compute_cost_matrix(self, pred_logits, pred_boxes, gt_labels, gt_boxes):
        """
        pred_logits: [N, nc+1] N 个预测, nc 个类别 + 1 个无目标
        pred_boxes:  [N, 4] 归一化坐标
        gt_labels:   [M] M 个 GT
        gt_boxes:    [M, 4]
        """
        N, nc = pred_logits.shape
        M = gt_labels.shape[0]
        device = pred_logits.device
        
        # 1. 分类 cost: 取负的类别预测分数
        cost_class = -pred_logits[:, gt_labels]  # [N, M]
        
        # 2. L1 box cost
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)  # [N, M]
        
        # 3. GIoU cost (简化: 用 IoU 代替)
        ious = self._bbox_iou(pred_boxes, gt_boxes)
        cost_giou = 1 - ious  # [N, M]
        
        # 加权求和
        C = self.cost_class * cost_class + \
            self.cost_bbox * cost_bbox + \
            self.cost_giou * cost_giou
        
        return C

    def _bbox_iou(self, b1, b2):
        """计算 IoU 矩阵 [N, M]"""
        N, M = b1.shape[0], b2.shape[0]
        b1 = b1.unsqueeze(1).expand(-1, M, -1)
        b2 = b2.unsqueeze(0).expand(N, -1, -1)
        
        inter_x1 = torch.max(b1[..., 0], b2[..., 0])
        inter_y1 = torch.max(b1[..., 1], b2[..., 1])
        inter_x2 = torch.min(b1[..., 2], b2[..., 2])
        inter_y2 = torch.min(b1[..., 3], b2[..., 3])
        
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-7)

    def match(self, pred_logits, pred_boxes, gt_labels, gt_boxes):
        """
        执行匈牙利匹配。
        简化: 用贪心算法近似 (实际 DETR 用 scipy.optimize.linear_sum_assignment)
        """
        C = self.compute_cost_matrix(pred_logits, pred_boxes, gt_labels, gt_boxes)
        N, M = C.shape
        
        # 贪心匹配: 每个 GT 选 cost 最小的预测
        matched = torch.full((M,), -1, dtype=torch.long, device=C.device)
        used = torch.zeros(N, dtype=torch.bool, device=C.device)
        
        for j in range(M):
            costs = C[:, j].clone()
            costs[used] = float('inf')
            i = costs.argmin().item()
            if not used[i]:
                matched[j] = i
                used[i] = True
        
        return matched


def test_hungarian_matching():
    """测试匈牙利匹配"""
    print("\n[DETR 匈牙利匹配]")
    print("=" * 60)
    
    N = 10  # 预测数
    M = 3   # GT 数
    nc = 5  # 类别数
    
    pred_logits = torch.randn(N, nc + 1)
    pred_boxes = torch.rand(N, 4)
    gt_labels = torch.randint(0, nc, (M,))
    gt_boxes = torch.rand(M, 4)
    
    matcher = HungarianMatcher()
    matched = matcher.match(pred_logits, pred_boxes, gt_labels, gt_boxes)
    
    print(f"  预测数: {N}, GT 数: {M}, 类别数: {nc}")
    print(f"  Cost matrix: {N}×{M}")
    print(f"  Matched: GT→预测 {matched.tolist()}")
    print(f"  (未匹配的预测 → 无目标类别, 不参与 box loss)")
    print(f"\n  DETR vs YOLOv8 损失对比:")
    print(f"  DETR  : 分类 CE + L1 box + GIoU  (匈牙利匹配后)")
    print(f"  YOLOv8: 分类 BCE + CIoU + DFL    (TAL 分配后)")


# ============================================================
# 4. YOLOv8 vs DETR 推理对比
# ============================================================
def compare_yolo_detr():
    """对比 YOLO 和 DETR 的推理流程"""
    print("\n[YOLOv8 vs DETR 推理流程对比]")
    print("=" * 60)
    
    print(f"{'阶段':<20} {'YOLOv8':<30} {'DETR':<30}")
    print("-" * 80)
    print(f"{'特征提取':<20} {'CSPDarknet Backbone':<30} {'CNN Backbone + Transformer':<30}")
    print(f"{'特征融合':<20} {'FPN + PAN':<30} {'Encoder 自注意力':<30}")
    print(f"{'预测方式':<20} {'密集预测 (网格+anchor)':<30} {'集合预测 (100 query)':<30}")
    print(f"{'匹配方式':<20} {'TAL (对齐度量)':<30} {'匈牙利匹配 (二分图)':<30}")
    print(f"{'后处理':<20} {'NMS 去除重复':<30} {'无 NMS':<30}")
    print(f"{'损失函数':<20} {'BCE + CIoU + DFL':<30} {'CE + L1 + GIoU':<30}")
    print(f"{'推理速度':<20} {'快 (实时)':<30} {'慢 (Transformer 计算)':<30}")
    print(f"{'与 LLM 关系':<20} {'类似 CNN 文本编码':<30} {'与 Transformer 同构':<30}")
    
    print(f"\n  YOLOv8 推理: 640×640 图像 → 19200 个候选 → NMS → 框")
    print(f"  DETR 推理:  640×640 图像 → 100 个 query → 匈牙利匹配 → 框")
    print(f"  → DETR 的 query 与 LLM 的 token 同构!")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Day 13: Transformer 编码器-解码器与 DETR 训练流程")
    print("=" * 70)
    
    test_sequence_copy()
    explain_kv_cache()
    test_hungarian_matching()
    compare_yolo_detr()
    
    print("\n" + "=" * 70)
    print("连接 VLM 视角:")
    print("  DETR Object Query → 图像的区域 token")
    print("  LLaVA 视觉 token → 语言 token 的映射")
    print("  两者都是: 视觉特征 → 序列化表示 → 与语言交互")
    print("=" * 70)