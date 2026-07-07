"""
Day 7 - 周复盘、论文精读与知识体系构建
========================================
巩固本周所学，将零散的知识点串联成体系。

Author: YOLO Lesson Week1
"""

import os
import sys
from pathlib import Path


# ============================================================
# 1. 知识体系检查清单
# ============================================================

def knowledge_checklist():
    """第一周知识检查清单，用于自我评估。"""
    checklist = [
        # Day 1: IoU 变体
        ("Day 1", "IoU 数学定义与梯度特性", "理解 IoU = |A∩B|/|A∪B|，无重叠时梯度为0"),
        ("Day 1", "GIoU 公式与推导", "理解外接闭包框 C 的作用，GIoU = IoU - |C\\(A∪B)|/|C|"),
        ("Day 1", "DIoU 公式与推导", "DIoU = IoU - ρ²/c²，中心点距离归一化"),
        ("Day 1", "CIoU 公式与推导", "CIoU = IoU - (ρ²/c² + αv)，长宽比一致性惩罚"),
        ("Day 1", "梯度下降回归模拟", "能用纯张量梯度下降模拟框回归过程"),
        # Day 2: mAP
        ("Day 2", "TP/FP/FN 定义", "理解检测任务中的匹配逻辑"),
        ("Day 2", "PR 曲线绘制", "理解 Precision-Recall 曲线含义"),
        ("Day 2", "VOC 11点插值 AP", "11 点插值法计算 AP"),
        ("Day 2", "COCO 101点插值 AP", "101 点插值 + mAP@[0.5:0.95]"),
        ("Day 2", "pycocotools 验证", "能用 pycocotools 验证自己的实现"),
        # Day 3: NMS
        ("Day 3", "Greedy NMS 原理", "理解标准 NMS 的流程和缺陷"),
        ("Day 3", "Soft-NMS 两种衰减方式", "线性衰减 vs 高斯衰减"),
        ("Day 3", "DIoU-NMS 改进动机", "DIoU-NMS 对密集场景的优势"),
        ("Day 3", "NMS 适用场景选择", "根据不同场景选择合适的 NMS"),
        # Day 4: 感受野
        ("Day 4", "感受野逐层计算", "掌握 r_out = r_in + (k_eff-1)*j_in"),
        ("Day 4", "有效感受野概念", "理解理论 RF 与有效 RF 的区别"),
        ("Day 4", "YOLO 检测头 RF 分析", "理解 P3/P4/P5 与目标大小的关系"),
        ("Day 4", "空洞卷积对 RF 的影响", "理解 dilation 如何扩大感受野"),
        # Day 5: 评估管道
        ("Day 5", "YOLO 验证流程", "理解 postprocess → NMS → metric 流程"),
        ("Day 5", "混淆矩阵分析", "理解 TP/FP/FN 在混淆矩阵中的表示"),
        ("Day 5", "独立评估脚本", "能构建自己的评估脚本"),
        # Day 6: DFL
        ("Day 6", "DFL 核心思想", "离散概率分布建模回归任务"),
        ("Day 6", "DFL 损失函数公式", "交叉熵 + 线性插值"),
        ("Day 6", "从分布恢复坐标", "概率加权求和 y_hat = Σ(i*S_i)"),
        ("Day 6", "DFL 与传统回归对比", "理解 DFL 的优势和适用场景"),
    ]

    print("=" * 70)
    print("第一周知识检查清单")
    print("=" * 70)
    print(f"{'Day':<8} {'知识点':<20} {'掌握标准':<40}")
    print("=" * 70)

    for day, knowledge, standard in checklist:
        print(f"{day:<8} {knowledge:<20} {standard:<40}")

    print("=" * 70)
    print("请逐项检查自己是否掌握，在 □ 中打 ✓ 或 ✗")
    for day, knowledge, _ in checklist:
        print(f"  □ {day} - {knowledge}")


# ============================================================
# 2. 论文精读笔记模板
# ============================================================

def paper_reading_notes():
    """论文精读笔记模板。"""
    papers = [
        {
            "title": "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression",
            "authors": "Zheng et al., AAAI 2020",
            "core_ideas": [
                "DIoU: 在 IoU 基础上引入中心点距离归一化惩罚项",
                "CIoU: 在 DIoU 基础上增加长宽比一致性惩罚项 v 和 α",
                "DIoU-NMS: 将 DIoU 作为 NMS 的抑制判定标准",
                "解决了 IoU/GIoU 在无重叠和包含情况下的梯度问题",
            ],
            "key_formulas": [
                "DIoU = IoU - ρ²(b, b_gt) / c²",
                "CIoU = IoU - (ρ²(b,b_gt)/c² + αv)",
                "v = (4/π²) * (arctan(w_gt/h_gt) - arctan(w/h))²",
                "α = v / ((1 - IoU) + v)",
            ],
            "inspiration": "如何将 DIoU/CIoU 应用到自己的工业缺陷检测场景中？",
        },
        {
            "title": "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes",
            "authors": "Li et al., NeurIPS 2020",
            "core_ideas": [
                "Quality Focal Loss (QFL): 将目标置信度与 IoU 联合建模",
                "Distribution Focal Loss (DFL): 框回归的离散概率分布建模",
                "将分类和回归统一到同一种 focal loss 框架下",
            ],
            "key_formulas": [
                "DFL: L = -((yi+1 - y) * log(S_i) + (y - yi) * log(S_{i+1}))",
                "坐标恢复: ŷ = Σ(i * S_i)",
                "QFL: L = -|y - σ|^β * ((1-y)log(1-σ) + y*log(σ))",
            ],
            "inspiration": "DFL 的分布建模思路能否用于其他回归任务？",
        },
    ]

    for i, paper in enumerate(papers, 1):
        print(f"\n{'=' * 70}")
        print(f"论文 {i}: {paper['title']}")
        print(f"作者: {paper['authors']}")
        print(f"{'=' * 70}")

        print(f"\n核心创新点:")
        for idea in paper['core_ideas']:
            print(f"  • {idea}")

        print(f"\n关键公式:")
        for formula in paper['key_formulas']:
            print(f"  • {formula}")

        print(f"\n对自身场景的启发:")
        print(f"  {paper['inspiration']}")

    print(f"\n{'=' * 70}")
    print("建议: 在笔记本上完整推导每个公式，理解每一步的几何意义。")


# ============================================================
# 3. 代码仓库整理指南
# ============================================================

def organize_code_repo():
    """代码仓库整理建议。"""
    print("\n" + "=" * 70)
    print("代码仓库整理指南")
    print("=" * 70)
    print("""
推荐目录结构:

week1/
├── day1/
│   └── iou.py              # IoU, GIoU, DIoU, CIoU 实现
├── day2/
│   └── map.py              # mAP 计算 (VOC + COCO)
├── day3/
│   └── nms.py              # NMS, Soft-NMS, DIoU-NMS
├── day4/
│   └── receptive_field.py  # 感受野计算与分析
├── day5/
│   └── eval_script.py      # 完整评估脚本
├── day6/
│   └── dfl_loss.py         # DFL 从零实现
├── day7/
│   └── review.py           # 复盘与总结
└── README.md               # 周整体说明

整理建议:
1. 在每个文件头部添加详细的文档字符串 (已添加)
2. 为关键函数添加类型注解 (已添加)
3. 在 README.md 中记录本周学习心得和踩坑记录
4. 将可复用的函数 (如 iou, nms) 提取到 utils 模块
    """)


# ============================================================
# 4. 思维导图结构
# ============================================================

def mind_map():
    """第一周知识体系思维导图。"""
    print("\n" + "=" * 70)
    print("第一周知识体系思维导图")
    print("=" * 70)
    print("""
第一周: 推理评估与基础组件
│
├── IoU 家族  ──── IoU
│   ├── 公式: |A∩B|/|A∪B|
│   ├── 问题: 无重叠梯度为0
│   ├── GIoU ── 引入外接闭包框C
│   ├── DIoU ── 引入中心点距离 ρ²/c²
│   └── CIoU ── 引入长宽比惩罚 αv
│
├── mAP 评估  ──── TP/FP/FN 匹配
│   ├── PR 曲线 ── Precision-Recall 关系
│   ├── VOC AP ── 11 点插值
│   └── COCO AP ── 101 点插值 + mAP@[0.5:0.95]
│
├── NMS 算法族 ──── Greedy NMS
│   ├── 按置信度排序 + IoU 抑制
│   ├── Soft-NMS ── 得分衰减 (线性/高斯)
│   └── DIoU-NMS ── 改用 DIoU 判定
│
├── 感受野  ──── 理论 RF 计算
│   ├── 公式: r_out = r_in + (k_eff-1)*j_in
│   ├── 有效 RF ── 高斯分布，约为理论 1/3~1/2
│   └── YOLO 检测头 ── P3/8(小目标) P4/16(中) P5/32(大)
│
└── DFL ──── 离散概率分布
    ├── 每条边 reg_max 个离散值
    ├── Loss: 交叉熵 + 线性插值
    └── 恢复: 概率加权求和
    """)


# ============================================================
# 5. 相关资源汇总
# ============================================================

def resources():
    """本周所有相关资源汇总。"""
    print("\n" + "=" * 70)
    print("第一周学习资源汇总")
    print("=" * 70)

    resources_list = [
        ("论文", [
            ("Generalized Intersection over Union", "Rezatofighi et al., CVPR 2019"),
            ("Distance-IoU Loss (DIoU + CIoU)", "Zheng et al., AAAI 2020"),
            ("Soft-NMS", "Bodla et al., ICCV 2017"),
            ("Generalized Focal Loss (DFL)", "Li et al., NeurIPS 2020"),
            ("Understanding Effective Receptive Field", "Luo et al., NeurIPS 2016"),
            ("PASCAL VOC Challenge", "Everingham et al., IJCV 2010"),
        ]),
        ("源码", [
            ("Ultralytics: ultralytics/utils/metrics.py", "混淆矩阵、AP计算"),
            ("Ultralytics: ultralytics/utils/loss.py", "DFL、CIoU Loss"),
            ("Ultralytics: models/yolo/detect/val.py", "验证流程"),
            ("pycocotools: cocoeval.py", "COCO 官方评估实现"),
            ("torchvision.ops.nms", "PyTorch 官方 NMS"),
        ]),
        ("工具", [
            ("pytorch-receptive-field", "GitHub 感受野计算工具"),
            ("matplotlib", "画图工具"),
            ("numpy", "数值计算基础库"),
        ]),
        ("博客/教程", [
            ("知乎: 一文读懂 GIoU, DIoU, CIoU", "系列博客"),
            ("目标检测中的 mAP 是什么？", "多个技术博客"),
            ("Computing Receptive Fields of CNNs", "计算教程"),
        ]),
    ]

    for category, items in resources_list:
        print(f"\n{category}:")
        for name, desc in items:
            print(f"  • {name} — {desc}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Day 7 - 周复盘、论文精读与知识体系构建")
    print("=" * 80)

    print("\n" + "=" * 70)
    print("🚀 恭喜完成第一周学习！")
    print("=" * 70)
    print("""
  这七天非常硬核，每一个公式都动手推导，每一段核心代码都自己实现。
  完成本周训练，你将对模型输出的每一个指标、推理的每一个后处理步骤
  都建立起深层的数学直觉，这会是后续两周深入模型架构和压缩优化的坚实基础。
    """)

    # 1. 知识检查清单
    knowledge_checklist()

    # 2. 论文精读笔记
    paper_reading_notes()

    # 3. 代码仓库整理
    organize_code_repo()

    # 4. 思维导图
    mind_map()

    # 5. 资源汇总
    resources()

    print("\n" + "=" * 70)
    print("下周预告: 深入 YOLO 模型架构")
    print("=" * 70)
    print("""
  Day 8-9: YOLOv8 Backbone 详解 (CSPDarknet + SPPF)
  Day 10-11: YOLOv8 Neck 详解 (FPN + PAN)
  Day 12-13: YOLOv8 Head 详解 (Decoupled Head + Loss)
  Day 14: 周复盘与模型架构对比
    """)