"""
Day 10 — 检测头与解码器: 从 Anchor 到 Query
===========================================
内容:
  1. YOLOv5 Anchor-based 解码
  2. YOLOv8 Anchor-free 解码 (l, t, r, b)
  3. 两种解码的可视化对比
  4. Object Query 实现 (DETR 风格)
  5. 观察 query 的多样性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 1. YOLOv5 Anchor-based 解码
# ============================================================
def decode_yolov5(pred, anchors, stride, img_size=640):
    """
    YOLOv5 anchor-based 解码。
    
    公式:
      b_x = (σ(t_x) + c_x) * stride
      b_y = (σ(t_y) + c_y) * stride
      b_w = (a_w * exp(t_w)) * stride
      b_h = (a_h * exp(t_h)) * stride
    
    其中:
      t_x, t_y, t_w, t_h 是模型预测的偏移量
      c_x, c_y 是网格坐标
      a_w, a_h 是 anchor 先验尺寸
      σ 是 sigmoid 函数
    """
    batch_size, _, grid_h, grid_w = pred.shape
    num_anchors = len(anchors)
    num_params = pred.shape[1] // num_anchors  # 通常 85 (4+1+80)
    
    # 重塑: [B, A*85, H, W] → [B, A, 85, H, W]
    pred = pred.view(batch_size, num_anchors, num_params, grid_h, grid_w)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, A, H, W, 85]
    
    # 提取坐标偏移
    t_xy = pred[..., 0:2].sigmoid()  # σ(t_x), σ(t_y)
    t_wh = pred[..., 2:4]            # t_w, t_h
    
    # 网格坐标
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float().to(pred.device)
    grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]
    
    # anchor 尺寸
    anchors_t = torch.tensor(anchors, device=pred.device).float().view(
        num_anchors, 1, 1, 2)
    
    # 解码
    xy = (t_xy + grid) * stride              # 中心点 (网格坐标)
    wh = (anchors_t * t_wh.exp()) * stride    # 宽高
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)  # [x1, y1, x2, y2]
    obj_conf = pred[..., 4:5].sigmoid()       # 目标置信度
    cls_prob = pred[..., 5:].softmax(dim=-1)  # 类别概率
    
    return boxes, obj_conf, cls_prob


# ============================================================
# 2. YOLOv8 Anchor-free 解码 (l, t, r, b)
# ============================================================
def decode_yolov8(pred, stride, reg_max=16, img_size=640):
    """
    YOLOv8 anchor-free 解码。
    
    直接回归 (l, t, r, b) 四个边到网格中心的距离:
      l = 左边到网格中心的距离
      t = 上边到网格中心的距离
      r = 右边到网格中心的距离
      b = 下边到网格中心的距离
    
    使用 DFL (Distribution Focal Loss) 将离散分布转换为连续值:
      ŷ = Σ(i * softmax(logits_i))
    
    最终坐标:
      x1 = c_x - l,  y1 = c_y - t
      x2 = c_x + r,  y2 = c_y + b
    """
    batch_size, _, grid_h, grid_w = pred.shape
    num_classes = pred.shape[1] - 4 * reg_max
    
    # 分离 DFL 分布和分类
    dfl_pred = pred[:, :4 * reg_max, :, :]   # [B, 4*16, H, W]
    cls_pred = pred[:, 4 * reg_max:, :, :]   # [B, nc, H, W]
    
    # DFL 解码: 分布 → 连续值
    dfl_pred = dfl_pred.view(batch_size, 4, reg_max, grid_h, grid_w)
    dfl_pred = dfl_pred.permute(0, 1, 3, 4, 2)  # [B, 4, H, W, 16]
    dfl_probs = dfl_pred.softmax(dim=-1)          # softmax 在 16 个 bins 上
    
    # 加权求和: ŷ = Σ(i * p_i)
    bins = torch.arange(reg_max, device=pred.device).float().view(1, 1, 1, 1, -1)
    dist = (dfl_probs * bins).sum(dim=-1)  # [B, 4, H, W]
    
    # 网格坐标
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=0).float().to(pred.device)
    grid = grid.unsqueeze(0)  # [1, 2, H, W]
    
    # 解码 (l, t, r, b) → (x1, y1, x2, y2)
    # dist[0]=l, dist[1]=t, dist[2]=r, dist[3]=b
    x1 = (grid[:, 0:1] - dist[:, 0:1]) * stride
    y1 = (grid[:, 1:2] - dist[:, 1:2]) * stride
    x2 = (grid[:, 0:1] + dist[:, 2:3]) * stride
    y2 = (grid[:, 1:2] + dist[:, 3:4]) * stride
    
    boxes = torch.cat([x1, y1, x2, y2], dim=1)  # [B, 4, H, W]
    boxes = boxes.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 4]
    cls_prob = cls_pred.permute(0, 2, 3, 1).contiguous().sigmoid()
    
    return boxes, cls_prob


# ============================================================
# 3. 两种解码对比可视化
# ============================================================
def compare_decoding():
    """对比 anchor-based 和 anchor-free 的解码效果"""
    print("[解码对比]")
    print("=" * 60)
    
    # 模拟预测张量
    batch_size, grid_h, grid_w = 1, 3, 3
    img_size = 640
    stride = img_size // grid_h  # ≈ 213
    
    # YOLOv5: 每个网格 3 个 anchor, 85 参数
    anchors = [[10, 13], [16, 30], [33, 23]]
    pred_v5 = torch.randn(batch_size, 3 * 85, grid_h, grid_w)
    boxes_v5, obj_conf, cls_v5 = decode_yolov5(pred_v5, anchors, stride)
    print(f"  YOLOv5 (Anchor-based):")
    print(f"    预测张量: {tuple(pred_v5.shape)}")
    print(f"    解码框:   {tuple(boxes_v5.shape)}")
    print(f"    每个网格 {len(anchors)} 个 anchor 先验")
    print(f"    依赖 anchor 统计先验, 需聚类计算")
    
    # YOLOv8: 4*16 DFL + 80 类别
    reg_max = 16
    num_classes = 80
    pred_v8 = torch.randn(batch_size, 4 * reg_max + num_classes, grid_h, grid_w)
    boxes_v8, cls_v8 = decode_yolov8(pred_v8, stride, reg_max)
    print(f"\n  YOLOv8 (Anchor-free):")
    print(f"    预测张量: {tuple(pred_v8.shape)}")
    print(f"    解码框:   {tuple(boxes_v8.shape)}")
    print(f"    直接回归 (l,t,r,b), 无 anchor 先验")
    print(f"    使用 DFL 分布建模, 更灵活")
    
    print(f"\n  关键差异:")
    print(f"    - Anchor-based: 需要先验聚类, 解码公式复杂")
    print(f"    - Anchor-free:  直接回归边距, 简单直观")
    print(f"    - Anchor-based 对训练数据分布敏感")
    print(f"    - Anchor-free 泛化性更好")


# ============================================================
# 4. Object Query 实现 (DETR 风格)
# ============================================================
class ObjectQuery(nn.Module):
    """
    DETR 风格的 Object Query。
    
    每个 query 是一个可学习的嵌入向量, 代表一个潜在目标。
    通过交叉注意力融合图像特征, 输出检测结果。
    """
    def __init__(self, num_queries=100, d_model=256, num_classes=80):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        
        # 可学习的 query 嵌入
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # 交叉注意力: query 与图像特征交互
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, 
                                                batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8,
                                               batch_first=True)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 预测头
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 为无目标
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )
        
    def forward(self, image_features, pos_embed=None):
        """
        image_features: [B, N, d_model] 展平后的图像特征
        """
        B = image_features.shape[0]
        
        # 生成 query: [B, num_queries, d_model]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # 自注意力: query 之间互相交互
        q = queries
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm1(q)
        
        # 交叉注意力: query 与图像特征交互
        q = q + self.cross_attn(q, image_features, image_features)[0]
        q = self.norm2(q)
        
        # FFN
        q = q + self.ffn(q)
        q = self.norm3(q)
        
        # 预测
        class_logits = self.class_embed(q)  # [B, 100, 81]
        bbox = self.bbox_embed(q).sigmoid()  # [B, 100, 4] 归一化坐标
        
        return class_logits, bbox, q


# ============================================================
# 5. 观察 query 的多样性
# ============================================================
def visualize_query_diversity():
    """
    训练后 (或初始化后) 观察 query 的嵌入向量。
    好的 query 应该覆盖不同的位置和尺度。
    """
    print("\n[Object Query 多样性分析]")
    print("=" * 60)
    
    # 初始化 query 模块
    num_queries = 10  # 少量 query 便于观察
    d_model = 64
    query_embed = nn.Embedding(num_queries, d_model)
    
    # 生成随机图像特征
    B, N = 1, 100  # 100 个图像 patch
    img_features = torch.randn(B, N, d_model)
    
    # 前向传播
    model = ObjectQuery(num_queries, d_model, num_classes=10)
    class_logits, bbox, queries = model(img_features)
    
    print(f"  Query 嵌入: {num_queries} 个, 每个 {d_model} 维")
    print(f"  Query 输出形状: {tuple(queries.shape)}")
    
    # 计算 query 之间的相似度
    q_norm = F.normalize(queries[0], dim=-1)
    sim_matrix = torch.matmul(q_norm, q_norm.T)
    
    print(f"\n  Query 相似度矩阵 ({num_queries}×{num_queries}):")
    sim_str = "\n".join(
        ["    " + "  ".join([f"{v:.2f}" for v in row]) 
         for row in sim_matrix.detach().numpy()])
    print(sim_str)
    
    # 检查多样性
    diag_mean = sim_matrix.diag().mean().item()
    off_diag_mean = (sim_matrix.sum() - sim_matrix.diag().sum()).item() / (
        num_queries * num_queries - num_queries)
    
    print(f"\n  自相似度均值 (对角线): {diag_mean:.4f}")
    print(f"  互相似度均值 (非对角线): {off_diag_mean:.4f}")
    print(f"  差异度: {diag_mean - off_diag_mean:.4f} (越大表示 query 越多样)")
    
    if diag_mean - off_diag_mean > 0.1:
        print("  ✓ Query 具有多样性, 每个 query 关注不同区域")
    else:
        print("  ~ Query 差异较小, 可能收敛到相似模式")
    
    # 连接 LLM/VLM
    print(f"\n  连接 LLM/VLM 视角:")
    print(f"  - Object Query ≈ LLM 中的 Token Embedding")
    print(f"  - 每个 query 可视为一个'任务 token'")
    print(f"  - VLM 中类似: 视觉 token + 语言 token 的交叉注意力")
    print(f"  - 这就是从检测 → VLM 的桥梁: 集合预测 → 序列建模")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Day 10: 检测头与解码器 — 从 Anchor 到 Query")
    print("=" * 70)
    
    compare_decoding()
    visualize_query_diversity()
    
    print("\n" + "=" * 70)
    print("关键概念回顾:")
    print("  Anchor-based: 依赖先验统计, 需聚类")
    print("  Anchor-free:  直接回归, 更灵活, 配合 DFL")
    print("  Object Query: 可学习的潜在目标 token")
    print("  Query → LLM:  与 Token Embedding 同构")
    print("=" * 70)