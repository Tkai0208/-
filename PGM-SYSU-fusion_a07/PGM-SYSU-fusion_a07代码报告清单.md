# PGM-SYSU-fusion_a07 代码报告清单

> **项目定位**：在 PGM-SYSU-fusion（加权融合 α=0.5）基础上，将融合权重调整为 **α=0.7**（全局平均占 70%，Camera-aware 占 30%），探索更高全局权重对 All Search 与 Indoor Search 性能的影响。
>
> **基准对比**：以 `/root/work/PGM-SYSU-fusion/`（α=0.5）为直接对比基准，PGM-SYSU-fusion_a07 仅修改 `generate_cluster_features` 的默认 alpha 参数，其余代码与 fusion 基本一致。

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

## 二、代码修改清单（与 PGM-SYSU-fusion 对比）

PGM-SYSU-fusion_a07 相对于 PGM-SYSU-fusion 的修改**仅涉及 2 个文件**，核心改动仅一行 alpha 参数。

### 2.1 `train_sysu.py`

#### 修改 1：融合权重 alpha 从 0.5 调整为 0.7

```python
# fusion 版本（α=0.5）
def generate_cluster_features(labels, features, cids, alpha=0.5):

# fusion_a07 版本（α=0.7）
def generate_cluster_features(labels, features, cids, alpha=0.7):
```

对应注释同步更新：
```python
# fusion 版本
5) 加权融合：alpha * global + (1-alpha) * camera-aware。

# fusion_a07 版本
5) 加权融合：alpha * global + (1-alpha) * camera-aware (alpha=0.7)。
```

**权重含义变化**：

| 方案 | 全局平均权重 | Camera-aware 权重 | 公式 |
|------|------------|-------------------|------|
| **PGM-SYSU-fusion** | 50% | 50% | `0.5*global + 0.5*ca` |
| **PGM-SYSU-fusion_a07** | **70%** | **30%** | **`0.7*global + 0.3*ca`** |

- **动机**：fusion（α=0.5）的 Indoor Search 虽已修复 camera-aware 的退化（62.51%），但仍低于 baseline（62.38% vs 68.26% mAP）。提高全局平均权重至 70%，预期可进一步增强 Indoor Search 的稳定性。
- **代价**：Camera-aware 成分降至 30%，All Search 中多相机视角的平衡能力可能减弱。

#### 修改 2：日志目录

```python
# 行号约 409
log_s2_name = 'sysu_s2_fusion_a07'  # 原：'sysu_s2_fusion'
```

#### 修改 3：删除 Stage 1 中的冗余测试代码块

diff 显示 fusion 版中 Stage 1 存在一个 `if epoch % 2 == 0` 的测试代码块（约 63 行），在 a07 版中被删除。

**分析**：该代码块位于 `main_worker_stage1()` 函数中，但实际训练时 Stage 1 被自动跳过（直接复用 PGM-SYSU 的 Stage 1 best model），因此该删除**不影响实际训练流程**，仅为代码清理。

### 2.2 `test_sysu.py`

仅修改模型加载路径：

```python
log_name='sysu_s2_fusion_a07'  # 原：'sysu_s2_fusion'
```

### 2.3 未修改的文件

以下文件与 PGM-SYSU-fusion **完全一致**，未做任何改动：

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

与 PGM-SYSU-fusion 完全一致：

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
| `alpha` | **0.7** | **fusion_a07 特有**：全局平均权重 70%，camera-aware 权重 30% |

### 4.3 训练流程细节

1. **Stage 1 自动跳过**：代码检测到 PGM-SYSU 的 Stage 1 best model 已存在，直接进入 Stage 2。

2. **Stage 2 每 epoch 流程**：
   - 提取 RGB/IR 全部训练样本特征
   - Jaccard 距离 + DBSCAN 聚类 → 伪标签
   - **加权融合计算聚类中心**：`generate_cluster_features(..., alpha=0.7)`
   - Optimal Transport + Hungarian 匹配 → 跨模态伪标签映射
   - 初始化 `ClusterMemory`（CMhybrid 模式）
   - PCLMP 训练：200 iterations/epoch
   - 每偶数 epoch（`epoch % 2 == 0`）执行 All Search 测试（10 trials 取平均）

3. **最佳模型保存**：根据 All Search 的 Rank-1 保存 `train_model_best.pth.tar`

---

## 五、实验结果

### 5.1 Stage 2 训练过程（`sysu_s2_fusion_a07`）

Stage 2 基于 PGM-SYSU 的 Stage 1 最优模型重新训练，应用加权融合（α=0.7）改进，完成 50 epochs（每 2 个 epoch 评估一次）：

| Epoch | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP | 备注 |
|-------|--------|--------|---------|---------|-----|------|------|
| 0 | 41.37% | 70.11% | 81.75% | 91.50% | 38.67% | 24.03% | 加载 Stage 1 模型 |
| 2 | 37.14% | 65.66% | 78.39% | 89.41% | 35.70% | 22.43% | 下降 |
| 4 | 42.42% | 71.44% | 82.85% | 92.25% | 40.32% | 26.34% | 回升 |
| 6 | 41.23% | 68.01% | 79.65% | 89.91% | 38.41% | 24.35% | 波动 |
| 8 | 45.61% | 74.57% | 84.44% | 92.13% | 42.77% | 27.69% | 持续提升 |
| 10 | 49.07% | 77.56% | 87.95% | 95.07% | 45.44% | 29.94% | 快速提升 |
| 12 | 46.53% | 75.23% | 85.06% | 92.60% | 43.44% | 28.50% | 下降 |
| 14 | 47.29% | 75.17% | 84.54% | 92.19% | 43.22% | 27.37% | 波动 |
| 16 | 46.69% | 75.23% | 85.63% | 93.53% | 43.63% | 28.33% | 波动 |
| 18 | 49.22% | 77.81% | 87.07% | 93.83% | 45.81% | 30.30% | 回升 |
| 20 | 55.62% | 82.09% | 89.88% | 95.41% | 51.40% | 35.54% | 明显提升 |
| 22 | 57.45% | 83.29% | 91.13% | 96.20% | 53.25% | 37.48% | 持续提升 |
| 24 | 57.82% | 83.58% | 91.37% | 96.42% | 53.53% | 37.79% | 接近 best |
| 26 | 56.81% | 83.12% | 91.05% | 96.31% | 52.87% | 37.10% | 微降 |
| 28 | 57.20% | 83.22% | 91.10% | 96.17% | 52.98% | 37.22% | 回升 |
| 30 | 57.36% | 83.35% | 91.01% | 95.97% | 53.67% | 38.21% | 波动 |
| 32 | 57.48% | 83.24% | 90.83% | 95.70% | 53.34% | 37.35% | 波动 |
| 34 | 56.63% | 82.24% | 89.92% | 94.99% | 52.03% | 36.12% | 下降 |
| **36** | **58.15%** | **83.84%** | **91.47%** | **96.17%** | **53.56%** | **37.47%** | **Stage 2 Best** |
| 38 | 57.33% | 83.59% | 91.37% | 96.15% | 53.35% | 37.47% | 微降 |
| 40 | 57.52% | 84.13% | 91.68% | 96.18% | 53.00% | 36.75% | 波动 |
| 42 | 57.86% | 83.86% | 91.50% | 95.98% | 53.15% | 36.91% | 接近 best |
| 44 | 58.06% | 84.12% | 91.58% | 96.05% | 53.29% | 36.98% | 接近 best |
| 46 | 57.80% | 83.79% | 91.36% | 95.88% | 53.03% | 36.73% | 波动 |
| 48 | 58.13% | 84.15% | 91.68% | 96.00% | 53.48% | 37.20% | 接近 best |

**收敛趋势**：
- Epoch 0-10 波动爬升（41% → 49%）
- Epoch 10-20 快速提升（49% → 55.6%）
- Epoch 20-24 持续增长至 57.8%
- Epoch 24-48 在 56%-58% 区间波动收敛，最优 58.15% 出现在 Epoch 36

**Best Epoch**：36（R1=58.15%，训练日志 `best_epoch:36`）

### 5.2 最终测试结果

使用 Stage 2 最优模型（Epoch 36）进行测试：

#### All Search（全场景搜索）

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 58.16% | 84.09% | 91.51% | 95.87% | 53.48% | 37.25% |
| 1 | 56.03% | 84.38% | 91.82% | 95.85% | 53.56% | 38.62% |
| 2 | 59.03% | 83.49% | 91.09% | 95.90% | 52.94% | 36.05% |
| 3 | 59.24% | 85.93% | 93.29% | 97.21% | 55.65% | 39.86% |
| 4 | 59.14% | 85.70% | 93.01% | 97.34% | 55.73% | 40.77% |
| 5 | 58.22% | 83.22% | 90.69% | 96.61% | 53.68% | 37.20% |
| 6 | 57.24% | 82.75% | 90.98% | 95.58% | 52.41% | 35.86% |
| 7 | 58.53% | 84.46% | 91.90% | 96.32% | 52.61% | 35.00% |
| 8 | 57.88% | 81.44% | 90.01% | 95.53% | 52.05% | 36.15% |
| 9 | 57.93% | 82.96% | 90.32% | 95.53% | 53.49% | 37.94% |
| **平均** | **58.14%** | **83.84%** | **91.46%** | **96.17%** | **53.56%** | **37.47%** |

#### Indoor Search（室内搜索）

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 67.35% | 88.95% | 94.25% | 97.96% | 72.37% | 68.27% |
| 1 | 65.22% | 87.14% | 93.48% | 98.46% | 70.11% | 65.59% |
| 2 | 63.81% | 86.23% | 94.16% | 98.51% | 69.19% | 64.84% |
| 3 | 66.80% | 89.49% | 94.61% | 97.46% | 71.75% | 67.34% |
| 4 | 65.85% | 88.68% | 93.48% | 95.56% | 71.53% | 67.58% |
| 5 | 63.86% | 87.68% | 92.39% | 96.88% | 70.26% | 66.61% |
| 6 | 66.49% | 89.27% | 94.25% | 97.60% | 72.27% | 68.33% |
| 7 | 66.44% | 88.18% | 93.21% | 97.74% | 70.93% | 66.05% |
| 8 | 66.03% | 85.96% | 92.80% | 98.01% | 70.72% | 66.51% |
| 9 | 65.94% | 89.45% | 95.02% | 98.14% | 72.65% | 69.57% |
| **平均** | **65.78%** | **88.10%** | **93.76%** | **97.63%** | **71.18%** | **67.07%** |

### 5.3 训练结果与测试结果一致性校验

| 来源 | 模式 | Rank-1 | mAP | mINP |
|------|------|--------|-----|------|
| Stage 2 Epoch 36 (train_log) | All Search | 58.15% | 53.56% | 37.47% |
| test_log.txt | All Search | **58.14%** | **53.56%** | **37.47%** |
| Stage 2 Epoch 36 (train_log) | Indoor Search | — | — | — |
| test_log.txt | Indoor Search | **65.78%** | **71.18%** | **67.07%** |

**校验结论**：训练日志中的最优 epoch 结果与独立测试脚本的结果 **基本一致**（训练日志 All Search R1=58.15%，test_log All Search R1=58.14%，差异 0.01% 为 trial 随机采样所致），说明模型保存和加载正确。

### 5.4 与基线版本及不同 alpha 的对比

#### All Search 对比

| 版本 | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|------|--------|--------|---------|-----|------|
| **PGM-SYSU（基线）** | 58.08% | 83.31% | 90.32% | 52.80% | 36.07% |
| **PGM-SYSU-fusion (α=0.5)** | 59.85% | 85.47% | 92.83% | 53.70% | 36.17% |
| **PGM-SYSU-fusion_a07 (α=0.7)** | **58.14%** | **83.84%** | **91.46%** | **53.56%** | **37.47%** |
| **vs 基线** | **+0.06%** | **+0.53%** | **+1.14%** | **+0.76%** | **+1.40%** |
| **vs α=0.5** | **-1.71%** | **-1.63%** | **-1.37%** | **-0.14%** | **+1.30%** |

#### Indoor Search 对比

| 版本 | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|------|--------|--------|---------|-----|------|
| **PGM-SYSU（基线）** | 62.38% | 86.22% | 93.36% | 68.26% | 63.89% |
| **PGM-SYSU-fusion (α=0.5)** | 62.51% | 86.65% | 93.28% | 68.03% | 63.36% |
| **PGM-SYSU-fusion_a07 (α=0.7)** | **65.78%** | **88.10%** | **93.76%** | **71.18%** | **67.07%** |
| **vs 基线** | **+3.40%** | **+1.88%** | **+0.40%** | **+2.92%** | **+3.18%** |
| **vs α=0.5** | **+3.27%** | **+1.45%** | **+0.48%** | **+3.15%** | **+3.71%** |

### 5.5 结果分析

#### All Search（全场景搜索）

Fusion_a07（α=0.7）在 **All Search** 模式下表现**略优于基线，但明显低于 α=0.5**：
- **Rank-1 = 58.14%**：比基线高 0.06%，但比 α=0.5（59.85%）低 1.71%
- **mAP = 53.56%**：比基线高 0.76%，但比 α=0.5（53.70%）低 0.14%

**原因分析**：
- All Search 使用全部 4 个 RGB gallery 相机，多相机场景下 camera-aware 成分有显著价值
- α=0.7 将 camera-aware 降至 30%，削弱了多相机视角特征平衡能力
- 尽管如此，全局平均的 70% 权重仍保证了基本性能不落后于基线

#### Indoor Search（室内搜索）

Fusion_a07（α=0.7）在 **Indoor Search** 模式下取得了**突破性提升**：
- **Rank-1 = 65.78%**：比基线（62.38%）高 **+3.40%**，比 α=0.5（62.51%）高 **+3.27%**
- **mAP = 71.18%**：比基线（68.26%）高 **+2.92%**，比 α=0.5（68.03%）高 **+3.15%**

**原因分析**：
- Indoor Search 仅使用 2 个室内相机，camera-aware 的 per-camera 分组非常稀疏
- α=0.7 大幅提高全局平均权重后，聚类中心获得更强的全局统计稳定性
- 70% 全局 + 30% camera-aware 的组合在室内场景下远优于 50:50 分配
- 这表明 Indoor Search 场景下，全局平均比 camera-aware 更重要

#### Alpha 参数的影响规律

| alpha | 全局权重 | CA 权重 | All Search R1 | Indoor Search R1 | 适用场景 |
|-------|---------|---------|---------------|------------------|----------|
| **0.0** | 0% | 100% | 60.14% | 59.65% | 纯 camera-aware，All Search 最佳 |
| **0.5** | 50% | 50% | 59.85% | 62.51% | 平衡策略，兼顾两者 |
| **0.7** | 70% | 30% | 58.14% | **65.78%** | **Indoor Search 最佳** |
| **1.0** | 100% | 0% | 58.08% | 62.38% | 纯全局平均（baseline） |

**核心发现**：
- **All Search** 随 alpha 增大而单调下降（60.14% → 58.14%），说明多相机场景下 camera-aware 越纯越好
- **Indoor Search** 随 alpha 增大先升后降（59.65% → 65.78%），在 α=0.7 达到峰值，说明少相机场景需要更多全局稳定性
- **α=0.7 是 Indoor Search 的最优平衡点**，但代价是 All Search 下降 1.71%

#### 总体评价

| 场景 | α=0.7 效果 | 相对 α=0.5 | 推荐度 |
|------|-----------|-----------|--------|
| **All Search（多相机）** | ⚠️ 略低于 α=0.5（-1.71%） | 劣化 | 不推荐 |
| **Indoor Search（少相机）** | ✅ **大幅提升（+3.27%）** | 显著优化 | **强烈推荐** |

**结论**：α=0.7 是一种**场景特化策略**，当评估指标以 Indoor Search 为主时，α=0.7 能带来 3% 以上的 Rank-1 提升；但若需要兼顾 All Search，α=0.5 仍是更均衡的选择。

---

## 六、关键代码片段

### 6.1 加权融合聚类中心（α=0.7）

```python
@torch.no_grad()
def generate_cluster_features(labels, features, cids, alpha=0.7):
    """
    Camera-aware + Global 加权融合 cluster feature（alpha=0.7）：
    1) 在同一个 cluster 内，按 camera ID 分组；
    2) 每组内部先求平均特征（per-camera mean）；
    3) 对该 cluster 的所有 per-camera mean 再求平均，得到 camera-aware centroid；
    4) 同时计算全局平均 centroid（原始 baseline 方法）；
    5) 加权融合：alpha * global + (1-alpha) * camera-aware (alpha=0.7)。
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
        
        # Global mean
        all_feats = [f for cam_id in centers[idx] for f in centers[idx][cam_id]]
        center_global = torch.stack(all_feats, dim=0).mean(0)
        
        # Weighted fusion: 70% global + 30% camera-aware
        center_list.append(alpha * center_global + (1 - alpha) * center_ca)
    return torch.stack(center_list, dim=0)
```

### 6.2 调用方式

```python
# 使用默认 alpha=0.7
cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir, cids_ir)
cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb, cids_rgb)
```

---

## 七、项目文件清单

```
PGM-SYSU-fusion_a07/
├── train_sysu.py              # 主训练脚本（Stage 1 + Stage 2，含 α=0.7 修改）
├── test_sysu.py               # 测试脚本（加载 sysu_s2_fusion_a07 best model）
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
    └── sysu_s2_fusion_a07/    # Stage 2 训练日志与模型
        ├── train_log.txt      # 训练过程日志
        ├── test_log.txt       # best model 测试结果（10 trials）
        ├── checkpoint.pth.tar
        └── train_model_best.pth.tar   # best model（Epoch 36）
```

---

## 八、复现注意事项

1. **Stage 1 依赖**：本方案**不训练 Stage 1**，要求 PGM-SYSU 的 Stage 1 best model 已存在于 `/root/work/PGM-SYSU/logs/sysu_s1/train_model_best.pth.tar`。若路径变更，需修改 `train_sysu.py` 中 `stage1_best_path` 的硬编码路径。

2. **Alpha 调参指南**：
   - `alpha=0.0`：纯 camera-aware，All Search 最优（60.14%），Indoor Search 最差（59.65%）
   - `alpha=0.5`：均衡策略，All Search 较好（59.85%），Indoor Search 中等（62.51%）
   - `alpha=0.7`：全局偏重，All Search 一般（58.14%），Indoor Search 最优（65.78%）
   - `alpha=1.0`：纯全局平均（baseline），All Search=58.08%，Indoor Search=62.38%

3. **测试模式**：训练时仅每偶数 epoch 执行 All Search 测试（`epoch % 2 == 0`），不执行 Indoor Search。最终 Indoor Search 结果需通过 `test_sysu.py` 单独测试 best model 获得。

4. **最佳模型选择**：本方案 Best Epoch 为 36（R1=58.15%），早于 fusion（α=0.5）的 Epoch 45，说明更高全局权重使模型更早收敛到平台期。

---

> **报告生成时间**：2026-04-29  
> **数据来源**：`logs/sysu_s2_fusion_a07/train_log.txt`、`logs/sysu_s2_fusion_a07/test_log.txt` 原始日志精确提取  
> **验证状态**：所有 Rank-1 / mAP / mINP 数值均已与原始日志逐条核对，100% 一致
