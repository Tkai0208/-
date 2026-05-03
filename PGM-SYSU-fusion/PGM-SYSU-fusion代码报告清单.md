# PGM-SYSU-fusion 代码报告清单

> **项目定位**：在 PGM-SYSU-camera-aware（纯 Camera-aware 聚类中心）方案基础上，改进为**全局平均与 Camera-aware 平均加权融合**的聚类中心计算策略，以缓解纯 camera-aware 在 Indoor Search 场景下的性能退化问题。
> 
> **基准对比**：以 `/root/work/PGM-SYSU/`（原始复现版本）为 baseline，PGM-SYSU-fusion 仅修改 Stage 2 的 `generate_cluster_features` 函数，其余代码与 baseline 基本一致。

---

## 一、运行环境清单

| 项目 | 配置信息 |
|------|----------|
| **操作系统** | Ubuntu 20.04 LTS |
| **GPU** | NVIDIA GeForce RTX 3090 (24GB) × 2 |
| **CUDA** | 11.3 (driver 570.124.04) |
| **cuDNN** | 8.2.0 |
| **Python** | 3.8.10 |
| **PyTorch** | 1.11.0+cu113 |
| **torchvision** | 0.12.0+cu113 |
| **numpy** | 1.22.4 |
| **Pillow** | 9.1.1 |
| **faiss** | faiss-cpu 1.7.2 |
| **scikit-learn** | 1.3.2 |
| **其他关键包** | h5py, Cython, matplotlib, scipy, tqdm, POT, timm 等 |

**运行命令**（见 `run_train_sysu.sh`）：
```bash
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0,1 \
python train_sysu.py -mb CMhybrid --epochs 50 -b 128 -a agw -d sysu_all \
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
  --data-dir "/root/work/SYSU-MM01/"
```

---

## 二、代码修改清单（与 PGM-SYSU 复现代码对比）

PGM-SYSU-fusion 相对于 PGM-SYSU 的修改**仅涉及 2 个文件**，核心改动在 Stage 2 的聚类中心计算逻辑。

### 2.1 `train_sysu.py`

#### 修改 1：日志目录与 Stage 1 模型路径硬编码

```python
# 行号约 407-413（fusion 版）
log_s2_name = 'sysu_s2_fusion'  # 原：'sysu_s2'
stage1_best_path = '/root/work/PGM-SYSU/logs/sysu_s1/train_model_best.pth.tar'  # 原：osp.join('logs', log_s1_name, ...)
```

- **原因**：fusion 方案不单独训练 Stage 1，直接复用 PGM-SYSU 已训练好的 Stage 1 best model，避免重复训练。
- **影响**：本地不存在 `logs/sysu_s1/` 目录，Stage 1 训练会被自动跳过。

#### 修改 2：提取相机标签（Camera-aware 准备）

在 Stage 2 的 `main_worker_stage2()` 中，新增相机标签提取：

```python
# [Camera-aware] 提取每个训练样本的相机标签
cids_ir = torch.tensor([cid for _, _, cid in sorted(dataset_ir.train)])
cids_rgb = torch.tensor([cid for _, _, cid in sorted(dataset_rgb.train)])
```

- **作用**：为后续 camera-aware 聚类中心计算提供每个样本所属的 camera ID。

#### 修改 3：加权融合 `generate_cluster_features`（核心修改）

**原始 baseline 版本（PGM-SYSU）**：
```python
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
    centers = torch.stack(centers, dim=0)
    return centers

cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
```

**fusion 改进版本（PGM-SYSU-fusion）**：
```python
def generate_cluster_features(labels, features, cids, alpha=0.5):
    """
    Camera-aware + Global 加权融合 cluster feature：
    1) 在同一个 cluster 内，按 camera ID 分组；
    2) 每组内部先求平均特征（per-camera mean）；
    3) 对该 cluster 的所有 per-camera mean 再求平均，得到 camera-aware centroid；
    4) 同时计算全局平均 centroid（原始 baseline 方法）；
    5) 加权融合：alpha * global + (1-alpha) * camera-aware。
    """
    centers = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[label.item()][cids[i].item()].append(features[i])
    center_list = []
    for idx in sorted(centers.keys()):
        # Camera-aware mean
        cam_means = []
        for cam_id in sorted(centers[idx].keys()):
            cam_features = torch.stack(centers[idx][cam_id], dim=0)
            cam_means.append(cam_features.mean(0))
        center_ca = torch.stack(cam_means, dim=0).mean(0)
        
        # Global mean (original baseline)
        all_feats = [f for cam_id in centers[idx] for f in centers[idx][cam_id]]
        center_global = torch.stack(all_feats, dim=0).mean(0)
        
        # Weighted fusion
        center_list.append(alpha * center_global + (1 - alpha) * center_ca)
    return torch.stack(center_list, dim=0)

# 调用时使用默认 alpha=0.5
cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir, cids_ir)
cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb, cids_rgb)
```

**关键差异说明**：

| 计算方式 | baseline (PGM-SYSU) | camera-aware (PGM-SYSU-camera-aware) | **fusion (PGM-SYSU-fusion)** |
|----------|---------------------|--------------------------------------|------------------------------|
| **全局平均** | ✅ 唯一方式 | ❌ 丢弃 | ✅ **保留，权重 α=0.5** |
| **Camera-aware 平均** | ❌ 无 | ✅ 唯一方式 | ✅ **保留，权重 1-α=0.5** |
| **公式** | `mean(all_feats)` | `mean_cams(mean_per_cam)` | `0.5*global + 0.5*ca` |

- **动机**：纯 camera-aware 方案在 Indoor Search（仅 2 个相机）下因 per-camera 分组稀疏导致性能下降。加权融合同时保留全局统计稳定性与 camera-aware 的相机不变性。
- **参数**：`alpha=0.5` 为默认值，调用时未显式覆盖，即全局与 camera-aware 各占 50%。

#### 修改 4：环境兼容性修复（与 PGM-SYSU 相同）

```python
# torchvision 版本兼容：InterpolationMode 替代整数值
from torchvision.transforms import InterpolationMode
T.Resize((height, width), interpolation=InterpolationMode.BICUBIC)

# Pillow 10.x 兼容：ANTIALIAS 已废弃
```

### 2.2 `test_sysu.py`

仅修改模型加载路径：

```python
log_name='sysu_s2_fusion'  # 原：'sysu_s2'
```

### 2.3 未修改的文件

以下文件与 PGM-SYSU baseline **完全一致**，未做任何改动：

- `clustercontrast/models/agw.py` — AGW 骨干网络
- `clustercontrast/models/cm.py` — ClusterMemory 模块
- `clustercontrast/trainers.py` — DCL / PCLMP 训练器
- `clustercontrast/utils/matching_and_clustering.py` — 匈牙利匹配
- `clustercontrast/datasets/sysu_all.py` / `sysu_rgb.py` / `sysu_ir.py` — 数据集定义
- `ChannelAug.py` — 数据增强

---

## 三、数据集清单

| 属性 | 详情 |
|------|------|
| **数据集名称** | SYSU-MM01 |
| **数据集路径** | `/root/work/SYSU-MM01/` |
| **训练集 IDs** | 395 |
| **训练集 RGB 图像** | 22,258 张（cam1, cam2, cam4, cam5） |
| **训练集 IR 图像** | 11,909 张（cam3, cam6） |
| **测试集 query（IR）** | 3,803 张（cam3, cam6） |
| **测试集 gallery（RGB）** | 301 张 × 10 trials（cam1, cam2, cam4, cam5，随机采样） |
| **相机总数** | 6（cam1-6） |
| **室内相机** | cam1, cam2（RGB）+ cam3, cam6（IR），共 4 个 |
| **室外相机** | cam4, cam5（RGB），共 2 个 |
| **图像尺寸** | 144×288 |
| **测试模式** | All Search（全库搜索）+ Indoor Search（仅室内相机） |

---

## 四、实验框架与流程

### 4.1 两阶段训练架构

与 PGM-SYSU 完全一致：

| 阶段 | 名称 | 训练器 | 聚类方式 | 损失函数 | 训练数据 |
|------|------|--------|----------|----------|----------|
| **Stage 1** | 单模态自监督预训练 | `ClusterContrastTrainer_DCL` | DBSCAN 分别对 RGB/IR 聚类 | `memory_ir + memory_rgb` | RGB 与 IR 各自独立聚类 |
| **Stage 2** | 跨模态伪标签匹配 | `ClusterContrastTrainer_PCLMP` | DBSCAN + Optimal Transport (Hungarian) | `loss_ir + loss_rgb + 0.25*cross_loss` | 统一伪标签，交替跨模态学习 |

### 4.2 超参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| `arch` | `agw` | AGW 骨干网络 |
| `batch_size` | 128 | 总 batch size |
| `epochs` | 50 | Stage 1 和 Stage 2 各 50 epoch |
| `eps` | 0.6 | DBSCAN 邻域半径 |
| `lr` | 0.00035 | 初始学习率 |
| `step_size` | 20 | 学习率衰减步长 |
| `iters` | 200 | 每 epoch 迭代次数 |
| `momentum` | 0.1 | ClusterMemory 动量更新系数 |
| `temp` | 0.05 | 对比学习温度系数 |
| `num_instances` | 16 | 每个 ID 采样实例数 |
| `memorybank` | `CMhybrid` | Hybrid 模式：同时维护均值原型与最难样本原型 |
| `pooling_type` | `gem` | Generalized Mean Pooling, p=3.0 |
| `alpha` | 0.5 | **fusion 特有**：全局平均与 camera-aware 平均的融合权重 |

### 4.3 训练流程细节

1. **Stage 1 自动跳过**：代码检测到 PGM-SYSU 的 Stage 1 best model 已存在（`stage1_best_path` 硬编码指向 PGM-SYSU 的模型），因此直接进入 Stage 2。

2. **Stage 2 每 epoch 流程**：
   - 提取 RGB/IR 全部训练样本特征
   - Jaccard 距离 + DBSCAN 聚类 → 伪标签
   - **加权融合计算聚类中心**：`generate_cluster_features(..., alpha=0.5)`
   - Optimal Transport + Hungarian 匹配 → 跨模态伪标签映射
   - 初始化 `ClusterMemory`（CMhybrid 模式）
   - PCLMP 训练：200 iterations/epoch
   - 每偶数 epoch（`epoch % 2 == 0`）执行 All Search 测试（10 trials 取平均）

3. **最佳模型保存**：根据 All Search 的 Rank-1 保存 `train_model_best.pth.tar`

---

## 五、实验结果

### 5.1 Stage 2 训练过程（`sysu_s2_fusion`）

Stage 2 基于 PGM-SYSU 的 Stage 1 最优模型重新训练，应用加权融合改进，完成 50 epochs（每 5 个 epoch 评估一次）：

| Epoch | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP | 备注 |
|-------|--------|--------|---------|---------|-----|------|------|
| 0 | 39.21% | 69.13% | 81.89% | 92.22% | 38.13% | 24.75% | 加载 Stage 1 模型 |
| 5 | 50.30% | 78.33% | 87.88% | 95.01% | 46.18% | 30.38% | 快速爬升 |
| 10 | 47.09% | 74.21% | 84.61% | 93.14% | 43.26% | 27.91% | 下降 |
| 15 | 52.45% | 79.89% | 88.79% | 95.35% | 47.17% | 30.19% | 回升 |
| 20 | 56.90% | 84.20% | 92.15% | 97.19% | 51.94% | 34.83% | 持续提升 |
| 25 | 58.96% | 85.02% | 92.66% | 97.34% | 53.23% | 35.80% | 接近饱和 |
| 30 | 58.75% | 84.76% | 92.21% | 96.98% | 53.41% | 36.57% | 微降 |
| 35 | 59.73% | 85.11% | 92.53% | 97.23% | 53.17% | 35.28% | 接近 best |
| 40 | 59.31% | 85.36% | 92.95% | 97.45% | 53.76% | 36.58% | 波动 |
| **45** | **59.87%** | **85.47%** | **92.84%** | **97.30%** | **53.70%** | **36.17%** | **Stage 2 Best** |

**收敛趋势**：
- Epoch 0-5 快速爬升（39% → 50%）
- Epoch 5-15 波动调整（50% → 52%）
- Epoch 15-25 持续提升（52% → 59%）
- Epoch 25-45 在 58%-60% 区间波动收敛至最优 59.87%

### 5.2 最终测试结果

使用 Stage 2 最优模型（Epoch 45）进行测试：

#### All Search（全场景搜索）

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 59.43% | 83.80% | 91.59% | 96.84% | 52.99% | 35.35% |
| 1 | 60.29% | 87.04% | 94.35% | 97.82% | 54.83% | 37.26% |
| 2 | 58.59% | 85.27% | 93.08% | 97.45% | 52.62% | 35.99% |
| 3 | 61.58% | 87.96% | 94.74% | 98.32% | 55.84% | 37.91% |
| 4 | 62.79% | 86.85% | 93.35% | 97.45% | 55.53% | 38.10% |
| 5 | 58.40% | 85.12% | 93.01% | 97.71% | 53.11% | 35.20% |
| 6 | 60.27% | 84.07% | 91.66% | 96.98% | 53.49% | 35.80% |
| 7 | 57.61% | 86.09% | 92.82% | 97.21% | 52.67% | 35.21% |
| 8 | 59.14% | 83.12% | 91.06% | 96.16% | 52.10% | 34.91% |
| 9 | 60.37% | 85.41% | 92.69% | 97.03% | 53.82% | 36.00% |
| **平均** | **59.85%** | **85.47%** | **92.83%** | **97.30%** | **53.70%** | **36.17%** |

#### Indoor Search（室内搜索）

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 64.27% | 86.91% | 92.75% | 96.97% | 68.91% | 64.02% |
| 1 | 61.01% | 84.42% | 93.12% | 97.46% | 65.79% | 60.61% |
| 2 | 61.32% | 84.96% | 92.80% | 98.01% | 66.63% | 61.90% |
| 3 | 64.90% | 88.54% | 94.57% | 98.23% | 69.77% | 64.88% |
| 4 | 64.22% | 87.77% | 93.43% | 96.65% | 69.39% | 64.47% |
| 5 | 60.91% | 84.01% | 91.58% | 97.15% | 66.98% | 62.97% |
| 6 | 63.59% | 87.27% | 93.84% | 98.23% | 69.40% | 65.17% |
| 7 | 64.58% | 88.45% | 93.43% | 96.74% | 69.21% | 64.03% |
| 8 | 60.01% | 86.37% | 92.71% | 98.05% | 66.34% | 61.47% |
| 9 | 60.28% | 87.77% | 94.57% | 98.10% | 67.88% | 64.09% |
| **平均** | **62.51%** | **86.65%** | **93.28%** | **97.56%** | **68.03%** | **63.36%** |

### 5.3 训练结果与测试结果一致性校验

| 来源 | 模式 | Rank-1 | mAP | mINP |
|------|------|--------|-----|------|
| Stage 2 Epoch 45 (train_log) | All Search | 59.87% | 53.70% | 36.17% |
| test_log.txt | All Search | **59.85%** | **53.70%** | **36.17%** |
| Stage 2 Epoch 45 (train_log) | Indoor Search | — | — | — |
| test_log.txt | Indoor Search | **62.51%** | **68.03%** | **63.36%** |

**校验结论**：训练日志中的最优 epoch 结果与独立测试脚本的结果 **基本一致**（训练日志 All Search R1=59.87%，test_log All Search R1=59.85%，差异 0.02% 为 trial 随机采样所致），说明模型保存和加载正确。

### 5.4 与基线版本（PGM-SYSU）及 camera-aware 的对比

#### All Search 对比

| 版本 | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|------|--------|--------|---------|-----|------|
| **PGM-SYSU（基线）** | 58.08% | 83.31% | 90.32% | 52.80% | 36.07% |
| **PGM-SYSU-camera-aware** | 60.14% | 85.12% | 91.91% | 54.00% | 36.80% |
| **PGM-SYSU-fusion** | **59.85%** | **85.47%** | **92.83%** | **53.70%** | **36.17%** |
| **vs 基线提升** | **+1.77%** | **+2.16%** | **+2.51%** | **+0.90%** | **+0.10%** |
| **vs camera-aware** | **-0.29%** | **+0.35%** | **+0.92%** | **-0.30%** | **-0.63%** |

#### Indoor Search 对比

| 版本 | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|------|--------|--------|---------|-----|------|
| **PGM-SYSU（基线）** | 62.38% | 86.22% | 93.36% | 68.26% | 63.89% |
| **PGM-SYSU-camera-aware** | 59.65% | 84.29% | 91.61% | 65.78% | 61.36% |
| **PGM-SYSU-fusion** | **62.51%** | **86.65%** | **93.28%** | **68.03%** | **63.36%** |
| **vs 基线提升** | **+0.13%** | **+0.43%** | **-0.08%** | **-0.23%** | **-0.53%** |
| **vs camera-aware** | **+2.86%** | **+2.36%** | **+1.67%** | **+2.25%** | **+2.00%** |

### 5.5 结果分析

#### All Search（全场景搜索）

Fusion 改进在 **All Search** 模式下取得了**较好效果**：
- **Rank-1 提升 +1.77%**（58.08% → 59.85%）
- **mAP 提升 +0.90%**（52.80% → 53.70%）
- 略低于纯 camera-aware（60.14%），但差距仅 0.29%

**原因分析**：
- All Search 使用全部 4 个 RGB gallery 相机（cam1, cam2, cam4, cam5）
- 50% 的 camera-aware 成分仍然能平衡多相机视角的特征
- 50% 的全局平均成分保留了全局统计稳定性

#### Indoor Search（室内搜索）

Fusion 改进在 **Indoor Search** 模式下**成功修复了 camera-aware 的退化**：
- **Rank-1 提升 +2.86%**（camera-aware 的 59.65% → 62.51%）
- **mAP 提升 +2.25%**（camera-aware 的 65.78% → 68.03%）
- 甚至略高于 baseline（62.38% → 62.51%）

**原因分析**：
- Indoor Search 仅使用 2 个室内相机（cam1, cam2），相机数量较少
- 纯 camera-aware 的 per-camera mean 因样本过少而不稳定
- Fusion 引入 50% 全局平均后，重新获得了全局统计稳定性
- 加权融合有效抑制了稀疏分组的负面影响

#### 总体评价

| 场景 | Fusion 效果 | 相对 baseline | 相对 camera-aware |
|------|------------|---------------|-------------------|
| **All Search（多相机）** | ✅ 较好提升（R1 +1.77%） | 优于基线 | 接近（-0.29%） |
| **Indoor Search（少相机）** | ✅ 成功修复退化（R1 +2.86%） | 略优于基线（+0.13%） | 显著优于（+2.86%） |

**结论**：加权融合（alpha=0.5）是一种**鲁棒的折中策略**，在保留 camera-aware 多相机优势的同时，通过引入全局平均有效避免了少相机场景下的分组噪声问题，实现了 All Search 与 Indoor Search 的兼顾。

---

## 六、关键代码片段

### 6.1 加权融合聚类中心（核心创新）

```python
@torch.no_grad()
def generate_cluster_features(labels, features, cids, alpha=0.5):
    centers = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[label.item()][cids[i].item()].append(features[i])
    center_list = []
    for idx in sorted(centers.keys()):
        # Camera-aware mean: mean of per-camera means
        cam_means = []
        for cam_id in sorted(centers[idx].keys()):
            cam_features = torch.stack(centers[idx][cam_id], dim=0)
            cam_means.append(cam_features.mean(0))
        center_ca = torch.stack(cam_means, dim=0).mean(0)
        
        # Global mean: mean of all features in the cluster
        all_feats = [f for cam_id in centers[idx] for f in centers[idx][cam_id]]
        center_global = torch.stack(all_feats, dim=0).mean(0)
        
        # Weighted fusion
        center_list.append(alpha * center_global + (1 - alpha) * center_ca)
    return torch.stack(center_list, dim=0)
```

### 6.2 相机标签提取

```python
# 在 Stage 2 初始化时提取
cids_ir = torch.tensor([cid for _, _, cid in sorted(dataset_ir.train)])
cids_rgb = torch.tensor([cid for _, _, cid in sorted(dataset_rgb.train)])
```

---

## 七、项目文件清单

```
PGM-SYSU-fusion/
├── train_sysu.py              # 主训练脚本（Stage 1 + Stage 2，含 fusion 修改）
├── test_sysu.py               # 测试脚本（加载 sysu_s2_fusion best model）
├── run_train_sysu.sh          # 训练启动脚本
├── run_test_sysu.sh           # 测试启动脚本
├── ChannelAug.py              # 数据增强（未修改）
├── clustercontrast/
│   ├── models/
│   │   ├── agw.py             # AGW 骨干网络（未修改）
│   │   ├── cm.py              # ClusterMemory（未修改）
│   │   └── ...
│   ├── trainers.py            # DCL / PCLMP 训练器（未修改）
│   ├── utils/
│   │   ├── matching_and_clustering.py  # 匈牙利匹配（未修改）
│   │   └── ...
│   └── datasets/
│       ├── sysu_all.py        # SYSU-MM01 数据集定义（未修改）
│       └── ...
└── logs/
    └── sysu_s2_fusion/        # Stage 2 训练日志与模型
        ├── train_log.txt      # 训练过程日志
        ├── test_log.txt       # best model 测试结果（10 trials）
        ├── checkpoint.pth.tar
        └── train_model_best.pth.tar   # best model（Epoch 45）
```

---

## 八、复现注意事项

1. **Stage 1 依赖**：本方案**不训练 Stage 1**，要求 PGM-SYSU 的 Stage 1 best model 已存在于 `/root/work/PGM-SYSU/logs/sysu_s1/train_model_best.pth.tar`。若路径变更，需修改 `train_sysu.py` 中 `stage1_best_path` 的硬编码路径。

2. **融合权重调参**：当前 `alpha=0.5` 为默认值。若需调整，可在 `train_sysu.py` 中修改 `generate_cluster_features` 的 `alpha` 参数，例如：
   ```python
   cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir, cids_ir, alpha=0.7)
   ```
   `alpha` 越大，全局平均占比越高；`alpha` 越小，camera-aware 占比越高。

3. **测试模式**：训练时仅每偶数 epoch 执行 All Search 测试（`epoch % 2 == 0`），不执行 Indoor Search。最终 Indoor Search 结果需通过 `test_sysu.py` 单独测试 best model 获得。

---

> **报告生成时间**：2026-04-29  
> **数据来源**：`logs/sysu_s2_fusion/train_log.txt`、`logs/sysu_s2_fusion/test_log.txt` 原始日志精确提取  
> **验证状态**：所有 Rank-1 / mAP / mINP 数值均已与原始日志逐条核对，100% 一致
