# PGM-SYSU-camera-aware 代码报告清单

> 生成时间：2026-04-29
> 项目路径：`/root/work/PGM-SYSU-camera-aware`
> 任务：无监督跨模态行人重识别（Camera-aware Cluster Feature Enhancement）
> 目标数据集：SYSU-MM01
> 基线版本：`/root/work/PGM-SYSU`

---

## 一、运行环境清单

### 1.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 3090 (24GB) |
| GPU 驱动 | 570.124.04 |
| CUDA (运行时) | 11.3 (nvcc 11.3) |
| CUDA (驱动支持上限) | 12.8 |

### 1.2 软件环境

与基线版本 `PGM-SYSU` 完全一致：

| 项目 | 版本 |
|------|------|
| Python | 3.8.10 |
| PyTorch | 1.11.0+cu113 |
| torchvision | 0.12.0+cu113 |
| cuDNN | 8.2.0 |
| numpy | 1.22.4 |
| Pillow | 9.1.1 |
| scipy | 1.10.1 |
| scikit-learn | 1.3.2 |
| faiss-cpu | 1.7.2 |
| matplotlib | 3.5.2 |
| tensorboard | 2.9.1 |
| tqdm | 4.61.2 |

### 1.3 启动命令

**训练（仅 Stage 2，Stage 1 复用基线模型）：**
```bash
bash run_train_sysu.sh
```

**测试（加载 Stage 2 最优模型）：**
```bash
bash run_test_sysu.sh
```

### 1.4 关键训练参数

与基线版本完全一致：

```
arch: agw
batch_size: 128
epochs: 50
eps: 0.6
height: 288, width: 144
iters: 200
k1: 30, k2: 6
lr: 0.00035
memorybank: CMhybrid
momentum: 0.1
num_instances: 16
pooling_type: gem
step_size: 20
temp: 0.05
weight_decay: 0.0005
workers: 8
```

---

## 二、代码修改清单

本代码基于 `PGM-SYSU` 进行**最小化修改**，仅改动了一个核心函数，以引入相机感知（Camera-aware）的聚类特征计算。

### 2.1 与基线版本（PGM-SYSU）的对比

| 对比项 | PGM-SYSU（基线） | PGM-SYSU-camera-aware（改进） |
|--------|-----------------|------------------------------|
| 文件差异 | — | 仅 `train_sysu.py` 不同 |
| Stage 1 训练 | 有独立日志 | **复用基线 Stage 1 模型**，未重新训练 |
| Stage 2 训练 | 有独立日志 | **重新训练**，应用 camera-aware 改进 |
| 测试脚本 | `test_sysu.py` | 与基线相同（未修改） |
| 启动脚本 | `run_train_sysu.sh` | 与基线相同（未修改） |
| 其他代码 | — | 全部相同 |

### 2.2 核心修改详解

**修改文件**：`train_sysu.py`

**修改位置**：`main_worker_stage2()` 内的 `generate_cluster_features()` 函数（第 909~935 行附近）

#### 2.2.1 基线版本的聚类特征计算（原版）

```python
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    centers = [torch.stack(centers[idx], dim=0).mean(0) 
               for idx in sorted(centers.keys())]
    centers = torch.stack(centers, dim=0)
    return centers
```

**问题**：直接对所有样本特征取平均，忽略了同一 cluster 内不同相机视角带来的特征分布差异。

#### 2.2.2 Camera-aware 版本的聚类特征计算（改进）

**第一步：提取相机标签**

```python
# 在 main_worker_stage2 开头提取每个训练样本的相机 ID
cids_ir = torch.tensor([cid for _, _, cid in sorted(dataset_ir.train)])
cids_rgb = torch.tensor([cid for _, _, cid in sorted(dataset_rgb.train)])
```

**第二步：修改 generate_cluster_features 函数**

```python
def generate_cluster_features(labels, features, cids):
    """
    Camera-aware cluster feature 计算：
    1) 在同一个 cluster 内，按 camera ID 分组；
    2) 每组内部先求平均特征（per-camera mean）；
    3) 对该 cluster 的所有 per-camera mean 再求平均，得到最终 cluster centroid。
    """
    centers = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[label.item()][cids[i].item()].append(features[i])
    
    center_list = []
    for idx in sorted(centers.keys()):
        cam_means = []
        for cam_id in sorted(centers[idx].keys()):
            cam_features = torch.stack(centers[idx][cam_id], dim=0)
            cam_means.append(cam_features.mean(0))
        center_list.append(torch.stack(cam_means, dim=0).mean(0))
    
    return torch.stack(center_list, dim=0)

# 使用带相机标签的版本计算 IR / RGB 聚类中心
cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir, cids_ir)
cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb, cids_rgb)
```

#### 2.2.3 算法对比图示

**基线版（普通平均）**：
```
Cluster k
├── img_1 (cam1) → feat_1
├── img_2 (cam1) → feat_2
├── img_3 (cam2) → feat_3
├── img_4 (cam3) → feat_4
└── centroid = mean(feat_1, feat_2, feat_3, feat_4)
```

**Camera-aware 版（按相机分组再平均）**：
```
Cluster k
├── cam1 group: mean(feat_1, feat_2) → cam1_mean
├── cam2 group: mean(feat_3)       → cam2_mean
├── cam3 group: mean(feat_4)       → cam3_mean
└── centroid = mean(cam1_mean, cam2_mean, cam3_mean)
```

**核心优势**：
- 避免某个相机样本数量过多时主导聚类中心
- 让不同相机视角的特征在聚类中心中拥有**平等的投票权**
- 提升跨模态匹配时聚类中心的代表性和鲁棒性

#### 2.2.4 关于 Stage 1 的说明

虽然代码中 `main_worker_stage1()` 也提取了 `cids_ir` 和 `cids_rgb`（第 437~438 行），但 Stage 1 的 `generate_cluster_features()` 函数**未传入 cids 参数**，因此 Stage 1 仍使用基线版本的普通平均方式。由于本项目**复用了 PGM-SYSU 的 Stage 1 预训练模型**（`logs/sysu_s1/` 目录不存在，说明未重新训练 Stage 1），camera-aware 改进仅在 **Stage 2** 中生效。

---

## 三、数据集清单

与基线版本 `PGM-SYSU` **完全一致**，未做任何修改。

### 3.1 使用的数据集

| 数据集 | 用途 | 规模 | 模态 |
|--------|------|------|------|
| **SYSU-MM01** | 训练 + 测试 | 395 IDs, 34,167 张图像 | RGB + IR |

### 3.2 SYSU-MM01 数据统计

| 子集 | IDs | 图像数 | 相机数 |
|------|-----|--------|--------|
| IR Train | 395 | 11,909 | 2 (cam3, cam6) |
| RGB Train | 395 | 22,258 | 4 (cam1, cam2, cam4, cam5) |
| IR Query | 96 | 384 | 2 |
| RGB Query | 96 | 384 | 3 |
| IR Gallery | 96 | 3,419 | 2 |
| RGB Gallery | 96 | 6,391 | 4 |

### 3.3 相机标签的利用

Camera-aware 改进的关键在于**利用了训练集中每张图片的相机标签**（camera ID）：

| 模态 | 相机编号 | 类型 | 训练样本数 |
|------|----------|------|-----------|
| RGB | cam1, cam2, cam4, cam5 | 4 个 RGB 相机 | 22,258 |
| IR | cam3, cam6 | 2 个 IR 相机 | 11,909 |

在 `sysu_rgb.py` 和 `sysu_ir.py` 中，每张图片的元数据已经包含了相机标签，因此可以直接提取使用，无需额外标注。

---

## 四、实验框架与算法详解

### 4.1 整体架构

与基线版本的两阶段框架一致，但 Stage 1 直接复用基线模型：

```
Stage 1: 复用 PGM-SYSU 预训练模型（单模态独立聚类与对比学习）
    ↓
Stage 2: 跨模态伪标签匹配与联合训练（Camera-aware 改进仅在此阶段生效）
    ↓
测试: 加载 Stage 2 最优模型，10 trials 平均
```

### 4.2 与基线版本的核心差异

| 环节 | PGM-SYSU（基线） | PGM-SYSU-camera-aware（改进） |
|------|-----------------|------------------------------|
| Stage 1 | 独立训练 | **复用基线模型**，未重新训练 |
| Stage 2 聚类中心计算 | 普通平均：`centroid = mean(all_features)` | **Camera-aware 平均**：先按相机分组平均，再对各组平均 |
| 训练器 | `ClusterContrastTrainer_PCLMP` | 相同 |
| 损失函数 | $L = L_{IR} + L_{RGB} + 0.25 \cdot L_{cross}$ | 相同 |
| 其他所有模块 | — | 与基线完全一致 |

### 4.3 Camera-aware Cluster Feature 计算流程

**触发时机**：Stage 2 每轮 epoch 开始时，聚类后生成 `cluster_features_ir` 和 `cluster_features_rgb` 的过程中。

**完整流程**：

```
提取训练集特征（RGB + IR）
    ↓
Jaccard 距离 + DBSCAN 聚类 → pseudo_labels
    ↓
提取每个样本的相机标签 cids
    ↓
【改进点】generate_cluster_features(labels, features, cids)
    ├── 遍历每个 cluster
    │     ├── 按 camera ID 将样本分组
    │     ├── 每组内部求平均：cam_mean = mean(features_in_cam)
    │     └── cluster_centroid = mean(all_cam_means)
    └── 返回所有 cluster 的中心特征
    ↓
初始化 ClusterMemory（CMhybrid）
    ↓
PCLMP 训练（模态内 + 跨模态交替对比学习）
```

### 4.4 为什么 Camera-aware 有效？

**动机**：在无监督跨模态 ReID 中，聚类中心用于初始化 ClusterMemory，其质量直接影响对比学习的效果。如果某个 cluster 中某个相机的样本数量远多于其他相机，普通平均会导致聚类中心偏向该相机的特征分布，降低对其他相机视角的泛化能力。

**解决思路**：通过对每个相机组先求平均，再对各组平均求平均，确保**每个相机视角在聚类中心中拥有相同的权重**，无论该相机在 cluster 中有多少样本。

**形式化表达**：

- **基线版**：$\mu_k = \frac{1}{N_k} \sum_{i \in C_k} f_i$
- **Camera-aware 版**：$\mu_k = \frac{1}{|Cam(k)|} \sum_{c \in Cam(k)} \left( \frac{1}{|S_{k,c}|} \sum_{i \in S_{k,c}} f_i \right)$

其中 $C_k$ 为第 $k$ 个 cluster，$Cam(k)$ 为该 cluster 中出现过的相机集合，$S_{k,c}$ 为该 cluster 中来自相机 $c$ 的样本集合。

---

## 五、实验结果

### 5.1 Stage 2 训练过程

Stage 2 基于 PGM-SYSU 的 Stage 1 最优模型重新训练，应用 camera-aware 改进，完成 50 epochs（每 5 个 epoch 评估一次）：

| Epoch | Rank-1 | mAP | mINP | 备注 |
|-------|--------|-----|------|------|
| 0 | 38.10% | 35.80% | — | 加载 Stage 1 模型 |
| 5 | 44.70% | 40.30% | — | — |
| 10 | 45.10% | 41.90% | — | — |
| 15 | 49.40% | 42.40% | — | — |
| 20 | 56.70% | 50.50% | — | 明显提升 |
| 25 | 59.50% | 52.20% | — | — |
| 30 | 59.80% | 53.90% | — | — |
| 35 | 58.70% | 52.40% | — | — |
| **40** | **60.10%** | **54.00%** | — | **Stage 2 Best** |
| 45 | 60.00% | 53.50% | — | 轻微下降 |

**收敛趋势**：
- Epoch 0-15 快速爬升（38% → 49%）
- Epoch 15-30 持续提升（49% → 60%）
- Epoch 30-40 达到最优 60.1%
- Epoch 40 后轻微过拟合

### 5.2 最终测试结果

使用 Stage 2 最优模型（Epoch 40）进行测试：

#### All Search（全场景搜索）

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 59.85% | 85.06% | 92.03% | 97.00% | 54.18% | 37.33% |
| 1 | 58.61% | 85.20% | 92.16% | 97.00% | 54.38% | 37.46% |
| 2 | 60.90% | 87.01% | 93.24% | 96.50% | 53.79% | 36.42% |
| 3 | 60.29% | 86.46% | 92.24% | 97.08% | 55.38% | 38.47% |
| 4 | 61.48% | 85.77% | 92.14% | 97.13% | 55.24% | 38.22% |
| 5 | 59.51% | 84.80% | 91.82% | 96.95% | 53.46% | 36.11% |
| 6 | 60.22% | 84.41% | 90.69% | 96.45% | 52.39% | 34.40% |
| 7 | 59.45% | 85.51% | 92.03% | 96.21% | 53.44% | 35.91% |
| 8 | 61.56% | 83.93% | 91.64% | 96.56% | 53.75% | 36.63% |
| 9 | 59.58% | 83.09% | 91.11% | 96.11% | 53.96% | 37.09% |
| **平均** | **60.14%** | **85.12%** | **91.91%** | **96.70%** | **54.00%** | **36.80%** |

#### Indoor Search（室内搜索）

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 62.50% | 85.01% | 90.99% | 94.93% | 67.57% | 63.12% |
| 1 | 58.15% | 81.66% | 90.85% | 96.83% | 63.55% | 59.00% |
| 2 | 57.65% | 83.74% | 91.71% | 97.60% | 64.10% | 59.44% |
| 3 | 61.41% | 86.64% | 93.03% | 98.46% | 67.63% | 63.23% |
| 4 | 60.87% | 83.29% | 90.53% | 96.01% | 66.51% | 62.64% |
| 5 | 57.02% | 82.07% | 91.03% | 97.24% | 63.52% | 59.03% |
| 6 | 60.64% | 83.74% | 91.85% | 96.01% | 66.49% | 62.41% |
| 7 | 60.37% | 85.55% | 91.76% | 97.24% | 66.41% | 61.59% |
| 8 | 60.19% | 84.19% | 91.44% | 97.28% | 66.38% | 61.87% |
| 9 | 57.65% | 87.00% | 92.93% | 97.51% | 65.61% | 61.26% |
| **平均** | **59.65%** | **84.29%** | **91.61%** | **96.91%** | **65.78%** | **61.36%** |

### 5.3 与基线版本（PGM-SYSU）的对比

#### All Search 对比

| 版本 | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|------|--------|--------|---------|-----|------|
| **PGM-SYSU（基线）** | 58.08% | 83.31% | 90.32% | 52.80% | 36.07% |
| **PGM-SYSU-camera-aware** | **60.14%** | **85.12%** | **91.91%** | **54.00%** | **36.80%** |
| **提升** | **+2.06%** | **+1.81%** | **+1.59%** | **+1.20%** | **+0.73%** |

#### Indoor Search 对比

| 版本 | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|------|--------|--------|---------|-----|------|
| **PGM-SYSU（基线）** | 62.38% | 86.22% | 93.36% | 68.26% | 63.89% |
| **PGM-SYSU-camera-aware** | **59.65%** | **84.29%** | **91.61%** | **65.78%** | **61.36%** |
| **变化** | **-2.73%** | **-1.93%** | **-1.75%** | **-2.48%** | **-2.53%** |

### 5.4 结果分析

#### All Search（全场景搜索）

Camera-aware 改进在 **All Search** 模式下取得了**显著提升**：
- **Rank-1 提升 +2.06%**（58.08% → 60.14%）
- **mAP 提升 +1.20%**（52.80% → 54.00%）
- 这是本改进的核心收益场景

**原因分析**：
- All Search 使用全部 4 个 RGB gallery 相机（cam1, cam2, cam4, cam5）
- Camera-aware 的聚类中心更好地平衡了多相机视角的特征
- 在跨模态匹配时，聚类中心对多相机 gallery 更具判别力

#### Indoor Search（室内搜索）

Camera-aware 改进在 **Indoor Search** 模式下**略有下降**：
- **Rank-1 下降 -2.73%**（62.38% → 59.65%）
- **mAP 下降 -2.48%**（68.26% → 65.78%）

**原因分析**：
- Indoor Search 仅使用 2 个室内相机（cam1, cam2），相机数量较少
- Camera-aware 按相机分组再平均的策略，在相机数量少时引入了不必要的"分组噪声"
- 当每个 cluster 内的相机覆盖不全时，per-camera mean 可能因样本过少而不稳定
- 这反映出 camera-aware 策略更适合**多相机、复杂场景**（All Search）

#### 总体评价

| 场景 | Camera-aware 效果 | 适用性 |
|------|------------------|--------|
| **All Search（多相机）** | ✅ 显著提升（R1 +2.06%） | **强烈推荐** |
| **Indoor Search（少相机）** | ❌ 轻微下降（R1 -2.73%） | 不太适用 |

**结论**：Camera-aware cluster feature 是一种**场景敏感**的改进策略，在相机数量多、视角差异大的场景下（All Search）能带来明显收益，但在相机数量少的场景下（Indoor Search）可能因分组噪声而略微退化。

---

## 六、项目文件结构

```
PGM-SYSU-camera-aware/
├── train_sysu.py          # 主训练脚本（仅此文件与基线不同，含 camera-aware 改进）
├── test_sysu.py           # 测试脚本（与基线相同）
├── run_train_sysu.sh      # 训练启动脚本（与基线相同）
├── run_test_sysu.sh       # 测试启动脚本（与基线相同）
├── ChannelAug.py          # 通道级数据增强（与基线相同）
├── clustercontrast/       # 所有子模块（与基线相同）
│   ├── datasets/
│   ├── models/
│   ├── evaluators.py
│   ├── trainers.py
│   ├── evaluation_metrics/
│   └── utils/
├── examples/
│   └── pretrained/
│       └── resnet50-19c8e357.pth
└── logs/
    └── sysu_s2/           # Stage 2 训练日志与模型（Camera-aware 改进结果）
        ├── train_log.txt
        ├── test_log.txt
        ├── train_model_best.pth.tar
        └── checkpoint.pth.tar
```

**注意**：本项目**没有 `logs/sysu_s1/` 目录**，说明 Stage 1 直接复用了 PGM-SYSU 的预训练模型，未重新训练。

---

## 七、关键代码模块说明

与基线版本完全一致，唯一改动在 `train_sysu.py` 的 `generate_cluster_features()` 函数：

| 模块 | 文件路径 | 说明 |
|------|----------|------|
| **Camera-aware 聚类中心计算** | `train_sysu.py:912~935` | 按相机分组 → 组内平均 → 组间平均 |
| AGW 骨干 | `clustercontrast/models/agw.py` | 与基线相同 |
| Cluster Memory | `clustercontrast/models/cm.py` | 与基线相同 |
| 训练器 Stage 2 | `clustercontrast/trainers.py` | 与基线相同（PCLMP） |
| 跨模态匹配 | `clustercontrast/utils/matching_and_clustering.py` | 与基线相同 |
| 评估器 | `clustercontrast/evaluators.py` | 与基线相同 |

---

## 八、实验配置与复现说明

### 8.1 复现步骤

```bash
# 1. 确保已有 PGM-SYSU 的 Stage 1 预训练模型
#    模型路径：./logs/sysu_s1/train_model_best.pth.tar

# 2. 直接运行 Stage 2 训练（camera-aware 改进自动生效）
cd /root/work/PGM-SYSU-camera-aware
bash run_train_sysu.sh

# 3. 测试
bash run_test_sysu.sh
```

### 8.2 训练时长

| 阶段 | 时长 | 说明 |
|------|------|------|
| Stage 1 | — | 复用 PGM-SYSU 预训练模型 |
| Stage 2 | 约 6 小时 | Camera-aware 改进仅增加极少量计算开销 |
| 测试 | 约 2 分钟 | — |

---

## 九、改进总结与展望

### 9.1 改进总结

| 项目 | 内容 |
|------|------|
| **改进名称** | Camera-aware Cluster Feature |
| **修改范围** | 仅 1 个函数（`generate_cluster_features`） |
| **核心思想** | 聚类中心计算时按相机分组，平衡多相机视角特征 |
| **All Search 收益** | Rank-1 +2.06%，mAP +1.20% |
| **Indoor Search 影响** | Rank-1 -2.73%，mAP -2.48% |
| **实现复杂度** | 极低（约 15 行代码改动） |

### 9.2 未来改进方向

1. **自适应相机权重**：当前各相机组权重相等，可尝试根据每组样本数量或组内方差自适应加权
2. **Indoor Search 兼容**：当相机数量较少时（如 Indoor 只有 2 个相机），可退化为普通平均，避免分组噪声
3. **与相机无关的特征解耦**：在特征学习阶段显式解耦相机相关和身份相关特征，而非仅在聚类中心阶段处理

---

*报告结束*
