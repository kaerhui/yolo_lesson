"""
Day 11 — 标签分配与损失函数中的"对齐"思想
========================================
内容:
  1. TaskAlignedAssigner (TAL) 实现
  2. 对齐度量公式: a = s^α * u^β
  3. 与单纯 IoU 匹配的对比实验
  4. YOLOv8 损失函数全景
  5. InfoNCE 对比损失实现
  6. 温度系数的作用分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ============================================================
# 1. TaskAlignedAssigner (TAL)
# ============================================================
class TaskAlignedAssigner:
    """
    TaskAlignedAssigner (TAL) — 任务对齐分配器。
    
    核心公式: align_metric = s^α * u^β
    其中:
      s: 分类分数 (预测类别与真实类别匹配的概率)
      u: IoU 分数 (预测框与真实框的 IoU)
      α: 分类权重 (默认 1.0)
      β: 定位权重 (默认 6.0)
    
    对齐度量同时考虑了"分类质量"和"定位质量"。
    """
    def __init__(self, alpha=1.0, beta=6.0, topk=10):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk

    def forward(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        """
        Args:
            pd_scores: [B, N, nc] 预测分类分数
            pd_bboxes: [B, N, 4] 预测框 (x1,y1,x2,y2)
            gt_labels: [B, M] 真实标签
            gt_bboxes: [B, M, 4] 真实框
        Returns:
            assigned_gt: [B, N] 每个预测匹配的 GT 索引
            assigned_labels: [B, N] 匹配的标签
            align_metrics: [B, N] 对齐度量
        """
        B, N, nc = pd_scores.shape
        M = gt_labels.shape[1]
        device = pd_scores.device

        # 1. 计算每对 (预测, GT) 的 IoU
        ious = self._bbox_iou(pd_bboxes, gt_bboxes)  # [B, N, M]
        
        # 2. 计算每对 (预测, GT) 的分类分数
        # 取 GT 类别对应的预测分数
        gt_labels_onehot = F.one_hot(gt_labels, nc).float()  # [B, M, nc]
        pd_scores_expanded = pd_scores.unsqueeze(2).expand(-1, -1, M, -1)
        gt_scores = (pd_scores_expanded * gt_labels_onehot.unsqueeze(1)).sum(-1)  # [B, N, M]
        
        # 3. 对齐度量: a = s^α * u^β
        align_metrics = (gt_scores ** self.alpha) * (ious ** self.beta)  # [B, N, M]
        
        # 4. Top-k 选择
        assigned_gt = torch.full((B, N), -1, device=device, dtype=torch.long)
        assigned_labels = torch.full((B, N), -1, device=device, dtype=torch.long)
        assigned_metrics = torch.zeros(B, N, device=device)
        
        for b in range(B):
            valid = gt_labels[b] >= 0  # 有效 GT
            if not valid.any():
                continue
            
            n_valid = valid.sum().item()
            k = min(self.topk, n_valid)
            
            # 对每个 GT 取 top-k 预测
            align_b = align_metrics[b]  # [N, M]
            ious_b = ious[b]
            
            for j in range(M):
                if not valid[j]:
                    continue
                # 取该 GT 的 top-k 预测
                scores_j = align_b[:, j]  # [N]
                _, topk_idx = torch.topk(scores_j, k, dim=0)
                
                # 分配 GT
                for idx in topk_idx:
                    if assigned_gt[b, idx] == -1 and scores_j[idx] > 0:
                        assigned_gt[b, idx] = j
                        assigned_labels[b, idx] = gt_labels[b, j]
                        assigned_metrics[b, idx] = scores_j[idx]
        
        return assigned_gt, assigned_labels, assigned_metrics

    def _bbox_iou(self, bbox1, bbox2):
        """计算两框 IoU"""
        # bbox1: [B, N, 4], bbox2: [B, M, 4]
        # 返回: [B, N, M]
        B, N, _ = bbox1.shape
        M = bbox2.shape[1]
        
        b1 = bbox1.unsqueeze(2).expand(-1, -1, M, -1)  # [B, N, M, 4]
        b2 = bbox2.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, M, 4]
        
        inter_x1 = torch.max(b1[..., 0], b2[..., 0])
        inter_y1 = torch.max(b1[..., 1], b2[..., 1])
        inter_x2 = torch.min(b1[..., 2], b2[..., 2])
        inter_y2 = torch.min(b1[..., 3], b2[..., 3])
        
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-7)


def test_tal():
    """测试 TAL 分配效果"""
    print("[TAL — TaskAlignedAssigner 测试]")
    print("=" * 60)
    
    B, N, M = 2, 20, 3  # 2 张图, 20 个预测, 3 个 GT
    nc = 10
    
    pd_scores = torch.rand(B, N, nc).sigmoid()
    pd_bboxes = torch.rand(B, N, 4)
    gt_labels = torch.randint(0, nc, (B, M))
    gt_bboxes = torch.rand(B, M, 4)
    
    # 确保 GT 框合理
    for b in range(B):
        for j in range(M):
            x1, y1 = torch.rand(2).uniform_(0, 0.3)
            x2, y2 = torch.rand(2).uniform_(0.7, 1.0)
            gt_bboxes[b, j] = torch.tensor([x1, y1, x2, y2])
    
    assigner = TaskAlignedAssigner(alpha=1.0, beta=6.0, topk=5)
    assigned_gt, assigned_labels, assign_metrics = assigner.forward(
        pd_scores, pd_bboxes, gt_labels, gt_bboxes)
    
    assigned_count = (assigned_gt >= 0).sum(dim=1)
    print(f"  每张图预测数: {N}")
    print(f"  每张图 GT 数: {M}")
    print(f"  每张图分配的预测: {assigned_count.tolist()}")
    print(f"  分配率: {(assigned_count.float().mean() / N * 100):.1f}%")
    
    # 单纯 IoU 匹配对比
    ious = assigner._bbox_iou(pd_bboxes, gt_bboxes)
    best_iou, best_gt = ious.max(dim=-1)  # [B, N]
    iou_assigned = (best_iou >= 0.5).sum(dim=1)
    print(f"\n  [对比] 单纯 IoU≥0.5 匹配个数: {iou_assigned.tolist()}")
    print(f"  [对比] 单纯 IoU≥0.7 匹配个数: {(best_iou >= 0.7).sum(dim=1).tolist()}")
    print(f"  [对比] TAL 分配个数:          {assigned_count.tolist()}")
    print(f"  → TAL 比单纯 IoU 匹配更灵活 (不仅看 IoU, 还看分类质量)")
    
    print(f"\n  TAL 核心公式: a = s^{assigner.alpha} * u^{assigner.beta}")
    print(f"  → 当 s 和 u 都高时, a 才高 → 同时考虑分类和定位质量")
    print(f"  → 对应 RLHF: 对齐信号 (human feedback) 筛选高质量生成")


# ============================================================
# 2. YOLOv8 损失函数全景
# ============================================================
def demonstrate_yolov8_loss():
    """展示 YOLOv8 的完整损失函数"""
    print("\n[YOLOv8 损失函数全景]")
    print("=" * 60)
    
    formula = """
    L_total = λ_box · L_CIoU + λ_cls · L_BCE + λ_dfl · L_DFL
    
    其中:
    L_CIoU = 1 - CIoU(pred_box, gt_box)          # 框回归损失
    L_BCE  = -[y·log(p) + (1-y)·log(1-p)]          # 分类损失
    L_DFL  = -[(y_{i+1}-y)·log(S_i) + (y-y_i)·log(S_{i+1})]  # 分布损失
    """
    print(formula)
    
    # 模拟损失计算
    pred_boxes = torch.randn(4, 4).sigmoid()
    gt_boxes = torch.rand(4, 4)
    pred_cls = torch.randn(4, 80).sigmoid()
    gt_cls = torch.zeros(4, 80)
    gt_cls[0, 3] = 1  # 假设类别 3
    
    # 模拟不同 loss gain 的对比
    loss_gains = {
        'v8n':  {'box': 7.5, 'cls': 0.5, 'dfl': 1.5},
        'v8s':  {'box': 7.5, 'cls': 0.5, 'dfl': 1.5},
        'v8m':  {'box': 7.5, 'cls': 0.5, 'dfl': 1.5},
        'v8l':  {'box': 7.5, 'cls': 0.5, 'dfl': 1.5},
        'v8x':  {'box': 7.5, 'cls': 0.5, 'dfl': 1.5},
    }
    
    print("  Loss gain 配置 (YOLOv8):")
    for name, gains in loss_gains.items():
        print(f"    {name}: box={gains['box']}, cls={gains['cls']}, dfl={gains['dfl']}")
    print("  → 所有版本使用相同的 loss gain, 通过缩放因子控制模型大小")


# ============================================================
# 3. InfoNCE 对比损失
# ============================================================
class InfoNCE(nn.Module):
    """
    InfoNCE 对比损失 (NT-Xent loss)。
    
    公式:
      L_i = -log(exp(sim(z_i, z_j)/τ) / Σ_{k!=i} exp(sim(z_i, z_k)/τ))
    
    其中:
      sim(z_i, z_j) = z_i^T z_j / (||z_i|| · ||z_j||)  (余弦相似度)
      τ: 温度系数
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: [B, D] 第 1 组特征
            z_j: [B, D] 第 2 组特征 (正样本对)
        Returns:
            loss: 标量
        """
        B = z_i.shape[0]
        device = z_i.device
        
        # 归一化
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # 拼接所有特征: [2B, D]
        z = torch.cat([z_i, z_j], dim=0)
        
        # 相似度矩阵: [2B, 2B]
        sim = torch.matmul(z, z.T) / self.temperature
        
        # 掩码: 排除自身
        mask = ~torch.eye(2 * B, device=device, dtype=torch.bool)
        
        # 正样本: (i, i+B) 和 (i+B, i)
        pos = torch.cat([
            torch.arange(B, device=device) + B,           # i → i+B
            torch.arange(B, device=device),               # i+B → i
        ])
        neg = torch.arange(2 * B, device=device)
        
        # 分母: 所有负样本 + 正样本
        logits = sim[mask].view(2 * B, -1)  # 移除自身
        labels = torch.zeros(2 * B, device=device, dtype=torch.long)
        labels[:B] = torch.arange(B, device=device) + B  # 正样本位置
        labels[B:] = torch.arange(B, device=device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


def test_infonce():
    """测试 InfoNCE 损失, 分析温度系数的作用"""
    print("\n[InfoNCE 对比损失 — 温度系数分析]")
    print("=" * 60)
    
    B = 8
    D = 16
    
    # 生成正样本对 (相似)
    z_i = torch.randn(B, D)
    z_j = z_i + torch.randn(B, D) * 0.1  # 正样本: 加小噪声
    
    # 测试不同温度
    temps = [0.01, 0.07, 0.5, 1.0, 5.0]
    print(f"  B={B}, D={D}")
    print(f"  z_i 与 z_j 的余弦相似度均值: {F.cosine_similarity(z_i, z_j).mean():.4f}")
    print()
    
    for tau in temps:
        nce = InfoNCE(temperature=tau)
        loss = nce(z_i, z_j)
        
        # 手动计算 logits 的分布
        z_i_norm = F.normalize(z_i, dim=-1)
        z_j_norm = F.normalize(z_j, dim=-1)
        sim_pos = (z_i_norm * z_j_norm).sum(dim=-1).mean().item()
        
        print(f"  τ={tau:<6}  Loss={loss.item():.4f}  "
              f"正样本相似度={sim_pos:.4f}")
        
        # 分析温度系数的作用
        if tau < 0.1:
            print(f"           → 低温: 尖锐分布, 仅关注最相似的正样本")
        elif tau < 0.5:
            print(f"           → 适中温度: 平衡正负样本, 推荐设置")
        else:
            print(f"           → 高温: 平滑分布, 对所有样本一视同仁")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Day 11: 标签分配与损失函数中的'对齐'思想")
    print("=" * 70)
    
    test_tal()
    demonstrate_yolov8_loss()
    test_infonce()
    
    print("\n" + "=" * 70)
    print("概念连接:")
    print("  TAL (对齐度量)  ←→  RLHF (人类反馈对齐)")
    print("  两者都是: 用对齐信号筛选高质量结果")
    print("  InfoNCE (对比学习)  ←→  CLIP (图文匹配)")
    print("  两者都是: 拉近正样本, 推远负样本")
    print("=" * 70)