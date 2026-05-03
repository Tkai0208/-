# RPNR-SYSU-ASM 复现代码报告清单

> 项目名称：Augmented and Softened Matching for Unsupervised Visible-Infrared Person Re-Identification (ICCV 2025) —— 基于 RPNR 的 Cross-modality Augmented Matching 改进  
> 复现代码路径：`/root/work/RPNR-SYSU-ASM`  
> 基线对比路径：`/root/work/RPNR-SYSU`  
> 报告生成时间：2026-04-29

---

## 一、项目概述

`RPNR-SYSU-ASM` 是在 `RPNR-SYSU` 基础上实现的改进方案，对应论文 **"Augmented and Softened Matching for Unsupervised Visible-Infrared Person Re-Identification" (ICCV 2025)** 的第 3.3 节 **"Cross-modality Augmented Matching (CAM)"**。

核心思想是：在 RGB→IR 的跨模态聚类匹配过程中，额外引入**灰度增强模态（Grayscale Augmented）**作为中间桥梁。将 RGB 图像转换为灰度图（模拟 IR 的单通道特性），提取灰度特征后与 RGB 特征共同参与跨模态匹配，通过**双sigmoid收缩融合**得到更鲁棒的跨模态相似度矩阵，进而构建加权 Hybrid Memory。

> ⚠️ **重要说明**：经与论文原文逐节核对，当前代码**仅实现了论文中的 CAM 组件**，而未实现论文完整 ASM 方法所需的 **SMU（Soft-labels Momentum Update）**、**L_csc（Cross-modality Soft Contrastive Loss）** 和 **L_chc（Cross-modality Hard Contrastive Loss）**。代码实际使用的训练损失仍为 RPNR 的原始损失（ACCL + NPC + RC + CMhcl），即论文消融实验中的 **L_accl**。因此，当前代码的实验结果应对应论文 Table 3 中的 **M2（CAM + L_accl）** 级别，而非完整 ASM（M6）。详见第 6.4 节和第八节对比分析。

---

## 二、RPNR-SYSU-ASM 与 RPNR-SYSU 的代码差异

经 `diff` 逐文件比对，除环境兼容性修改外，算法层面的核心差异如下：

| 差异项 | RPNR-SYSU（基线） | RPNR-SYSU-ASM（改进） | 影响说明 |
|--------|------------------|----------------------|----------|
| **跨模态匹配方式** | OTPM：仅用 RGB 聚类中心与 IR 聚类中心做最优传输 | **Augmented Matching**：引入灰度增强特征，融合 RGB→IR 和 Gray→IR 两个相似度矩阵 | 核心改进，利用灰度模态缩小 RGB 与 IR 的模态鸿沟 |
| **Hybrid Memory 构建** | 硬匹配一对一平均：`mean(IR[i], RGB[i2r[i]])` | **软加权融合**：`0.5*IR[j] + 0.5*Σ(weights*RGB)`，权重来自融合矩阵 M_var 的第 j 列 | 从硬分配变为软加权，降低错误匹配的冲击 |
| **聚类特征提取** | 仅提取 RGB 原始特征 | 额外提取 **Grayscale 增强特征**（ITU-R BT.601 系数转灰度，复制3通道） | 新增 `extract_features_augmented()` 函数 |
| **日志目录** | `logs/sysu_s2/` | `logs/sysu_asm_s2/` | 独立存储实验结果 |
| **PIL 兼容性** | `Image.Resampling.LANCZOS` | `Image.LANCZOS` | Pillow 版本兼容（ASM 直接使用旧版属性） |
| **state_dict 兼容性** | 直接 `model.load_state_dict(checkpoint['state_dict'])` | 自动处理 `module.` 前缀不匹配问题 | 增强 checkpoint 加载鲁棒性 |
| **CUDA→NumPy** | `feat` 直接转 numpy（CUDA tensor 可能报错） | `feat_cpu = feat.cpu()` 后再转 numpy | 修复潜在运行时错误 |
| **测试脚本** | 无 `--resume` 参数 | 新增 `--resume` 参数支持指定模型路径 | 更灵活的测试调用 |

**非算法改动说明**：
- `trainers.py` 的 `train()` 函数签名增加了 `data_loader_gray=None` 和 `optimizer=None`（默认参数），但**实际训练流程中并未使用灰度数据加载器**，灰度特征仅在聚类阶段通过 `extract_features_augmented()` 提取。

## 三、代码实现与论文的对应关系

论文完整 ASM 方法由以下四个核心组件构成（见论文 Section 3）：

| 组件 | 论文描述 | 代码实现状态 | 说明 |
|------|---------|-------------|------|
| **CAM** | Cross-modality Augmented Matching（灰度增强跨模态匹配） | ✅ **已实现** | `extract_features_augmented()` + `augmented_matching_fusion()` |
| **SMU** | Soft-labels Momentum Update（软标签动量更新） | ❌ **未实现** | 代码仍使用硬伪标签（one-hot），无 soft-label 动量更新机制 |
| **L_csc** | Cross-modality Soft Contrastive Loss（公式 10–13） | ❌ **未实现** | 代码仍使用 RPNR 的 ACCL/NPC Loss |
| **L_chc** | Cross-modality Hard Contrastive Loss（公式 15–17） | ❌ **未实现** | 代码仍使用 RPNR 的 RC Loss + CMhcl |

**论文总损失公式（Eq. 18）**：
```
L_total = L_icc + λ_csc · L_csc + λ_chc · L_chc    (λ_csc=0.5, λ_chc=0.5)
```

**代码实际损失**：
```
loss = loss_ir + loss_rgb + 0.25*cross_loss + 0.5*loss_hybrid + 10.0*loss_RC
```
即论文中的 **L_accl**（与 RPNR 基线一致）。

因此，当前代码 ≈ 论文消融实验 **M2 = CAM + L_accl**，而非完整 **M6 = CAM + L_csc + L_chc**。

---

## 四、代码框架与文件结构

```
RPNR-SYSU-ASM/
├── README.md                          # 仍为 RPNR 原始 README（未更新 ASM 指标）
├── ChannelAug.py                      # 跨模态数据增强（与 RPNR-SYSU 相同）
├── prepare_sysu.py                    # SYSU-MM01 预处理（与 RPNR-SYSU 相同）
├── train_sysu.py                      # 训练主脚本（含 Augmented Matching 核心实现）
│   ├── extract_features_augmented()   # 灰度增强特征提取
│   ├── augmented_matching_fusion()    # 双sigmoid融合相似度矩阵
│   └── main_worker()                  # 训练流程（聚类→灰度特征→融合→Hybrid Memory）
├── test_sysu.py                       # 测试脚本（新增 --resume 参数）
├── run_train_sysu_asm.sh              # ASM 训练启动脚本
├── run_test_sysu.sh                   # 测试启动脚本
├── environment.yml / requirements.txt # 运行环境（与 RPNR-SYSU 一致）
├── clustercontrast/
│   ├── trainers.py                    # ClusterContrastTrainer_RPNR（签名微调，逻辑不变）
│   ├── models/cm.py                   # ClusterMemory（CMhcl，与基线一致）
│   ├── models/losses.py               # RCLoss 等（与基线一致）
│   └── ...
└── logs/sysu_asm_s2/
    ├── log.txt                        # 完整训练日志（50 epochs + 最终 10-trial 测试）
    └── test_log.txt                   # 独立测试日志
```

---

## 五、运行环境清单

> ⚠️ **环境说明**：本项目的实验结果（`log.txt`、`test_log.txt`）是在**当前系统环境**（下表所列）下实际运行生成的，并非代码目录中 `requirements.txt` / `environment.yml` 所记录的原始论文环境（PyTorch 1.8.0 / CUDA 10.2 / faiss-gpu 1.6.3）。当前系统已安装的兼容版本可直接运行代码并复现实验结果。

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| Python | 3.8.10 | 主解释器 |
| PyTorch | 1.11.0+cu113 | GPU 版本，CUDA 11.3 编译 |
| torchvision | 0.12.0+cu113 | 与 PyTorch 配套 |
| CUDA Toolkit | 11.3 | 与 PyTorch 编译版本一致 |
| scikit-learn | 1.2.2 | DBSCAN |
| POT | 0.9.3 | Optimal Transport（Sinkhorn） |
| faiss-gpu | 1.6.3 | Jaccard distance 计算（GPU 版本） |
| Pillow | 9.1.1 | 图像处理 |
| numpy | 1.22.4 | 数值计算 |
| scipy | 1.10.1 | 科学计算 |
| tqdm | 4.61.2 | 进度条 |
| infomap | — | 未安装（代码中未实际调用） |

**硬件环境**：2 × NVIDIA GeForce RTX 3090（24GB 显存），Driver 570.124.04 |

**运行命令**：
```bash
# 训练（ASM Stage-2）
sh run_train_sysu_asm.sh
# 等效命令：
CUDA_VISIBLE_DEVICES=0,1 \
python train_sysu.py -mb CMhcl -b 128 -a agw -d sysu_all \
  --epochs 50 --num-instances 16 --iters 200 \
  --momentum 0.1 --eps 0.6 \
  --data-dir "/root/work/SYSU-MM01/"

# 测试
sh run_test_sysu.sh
# 等效命令：
CUDA_VISIBLE_DEVICES=0,1 \
python test_sysu.py -b 64 -a agw -d sysu_all \
  --resume logs/sysu_asm_s2/model_best.pth.tar \
  --data-dir "/root/work/SYSU-MM01/"
```

---

## 六、数据集清单

与 `RPNR-SYSU` 使用完全相同的 SYSU-MM01 预处理数据。

| 输出目录 | 内容 | 图像数量 |
|----------|------|----------|
| `SYSU-MM01/ir_modify/bounding_box_train` | IR 训练集 | 11,909 |
| `SYSU-MM01/rgb_modify/bounding_box_train` | RGB 训练集 | 22,258 |

**数据集统计（来自 `log.txt` 实际加载输出）**：

```text
=> sysu_ir loaded
  train    |   395 |    11909 |         2
=> sysu_rgb loaded
  train    |   395 |    22258 |         4
```

---

## 七、核心算法与代码逻辑

### 6.1 整体改进流程（对比 RPNR-SYSU）

RPNR-SYSU 的每 epoch 流程中，**步骤 4（OTPM）和步骤 5（Hybrid Memory）被替换为 Augmented Matching Fusion**：

```
1. 特征提取（eval mode）
   ├── 提取 RGB 训练集特征 → features_rgb
   ├── 提取 Gray 训练集特征 → features_rgb_aug   [ASM 新增]
   └── 提取 IR 训练集特征  → features_ir

2. 模态内聚类（DBSCAN + Jaccard distance）→ pseudo_labels_rgb / pseudo_labels_ir

3. 伪标签修正（Label Correction）→ pseudo_labels_rgb_hat / pseudo_labels_ir_hat

4. [ASM 替换 OTPM] Augmented Matching Fusion
   ├── cluster_features_rgb_aug = generate_cluster_features_corr(pseudo_labels_rgb, features_rgb_aug)
   ├── M_vr = cosine_similarity(cluster_features_rgb, cluster_features_ir)      # Visible→IR
   ├── M_ar = cosine_similarity(cluster_features_rgb_aug, cluster_features_ir)  # Augmented→IR
   ├── M_var = 1 / ((1 + exp(-gamma_v * M_vr)) * (1 + exp(-gamma_a * M_ar)))   # 融合矩阵
   └── 对 M_var 继续使用 Sinkhorn (reg=5) 求解最优传输 → r2i / i2r

5. [ASM 改进 Hybrid Memory]
   ├── 对 IR 簇 j：weights = M_var[:, j]（所有 Visible 簇对 IR 簇 j 的软权重）
   ├── weighted_rgb = weights^T @ cluster_features_rgb
   └── cluster_features_hybrid[j] = 0.5 * cluster_features_ir[j] + 0.5 * weighted_rgb

6. 训练（与 RPNR-SYSU 完全一致：ACCL + NPC Loss + CMhcl）
```

### 6.2 核心新增函数详解

#### 6.2.1 `extract_features_augmented()`（`train_sysu.py` 第 98–127 行）

```python
def extract_features_augmented(model, data_loader, print_freq=50, modal=1):
    for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
        imgs_gray = imgs.clone()
        # ITU-R BT.601 灰度转换
        gray = 0.2989 * imgs_gray[:, 0:1, :, :] + \
               0.5870 * imgs_gray[:, 1:2, :, :] + \
               0.1140 * imgs_gray[:, 2:3, :, :]
        imgs_gray = gray.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        imgs_gray = imgs_gray.cuda()
        outputs = model(imgs_gray, None, modal=modal)  # 使用 visible_module
```

- 将 RGB 图像按标准 ITU-R BT.601 系数转为单通道灰度，再复制为 3 通道输入网络
- 使用 `modal=1`（visible_module）提取特征，与 RGB 特征维度一致（2048-d）

#### 6.2.2 `augmented_matching_fusion()`（`train_sysu.py` 第 65–95 行）

```python
def augmented_matching_fusion(cluster_features_visible, cluster_features_augmented,
                               cluster_features_ir, gamma_v=1.0, gamma_a=1.0):
    M_vr = compute_similarity_matrix(cluster_features_visible, cluster_features_ir)
    M_ar = compute_similarity_matrix(cluster_features_augmented, cluster_features_ir)
    # 公式(8): 双sigmoid收缩融合
    M_var = 1.0 / ((1.0 + torch.exp(-gamma_v * M_vr)) * (1.0 + torch.exp(-gamma_a * M_ar)))
    return M_var, M_vr, M_ar
```

- `compute_similarity_matrix()` 先对特征做 L2 归一化，再计算余弦相似度
- **融合公式**：`M_var = 1 / [(1+e^(-γ_v·M_vr)) · (1+e^(-γ_a·M_ar))]`
- 该公式通过两个 sigmoid 函数分别对 Visible→IR 和 Augmented→IR 的相似度做非线性收缩，乘积形式实现**双重置信度筛选**
- SYSU-MM01 配置：`gamma_v=2.0, gamma_a=1.0`（Visible 分支权重更高，比例 2:1）

#### 6.2.3 改进的 Hybrid Memory 构建（`train_sysu.py` 第 670–689 行）

```python
cluster_features_hybrid = torch.zeros(cluster_features_ir.shape[0], cluster_features_ir.shape[1]).cuda()
for j in range(num_cluster_ir):
    weights = matching_matrix[:, j].cuda()  # [N_visible]
    weighted_rgb = torch.mm(weights.unsqueeze(0), cluster_features_rgb).squeeze(0)
    cluster_features_hybrid[j] = 0.5 * cluster_features_ir[j] + 0.5 * weighted_rgb
```

- **软加权**：不同于 RPNR-SYSU 的硬匹配一对一平均，ASM 使用融合矩阵 `M_var` 的第 j 列作为所有 RGB 簇对 IR 簇 j 的权重
- **加权融合**：50% IR 中心 + 50% 加权 RGB 中心

---

## 八、实验框架与流程

### 7.1 训练阶段关键超参数（与 `log.txt` 中 Args 一致）

| 参数 | 值 | 来源 |
|------|-----|------|
| `arch` | `agw` | 默认 |
| `pooling_type` | `gem` | 默认 |
| `batch_size` | 128 | `run_train_sysu_asm.sh` |
| `iters` | 200 | `run_train_sysu_asm.sh` |
| `epochs` | 50 | 默认 |
| `lr` | 0.00035 | 默认 |
| `step_size` | 20 | 默认（lr ×0.1） |
| `num_instances` | 16 | `run_train_sysu_asm.sh` |
| `memorybank` | `CMhcl` | `run_train_sysu_asm.sh` |
| `momentum` | 0.1 | `run_train_sysu_asm.sh` |
| `temp` | 0.05 | 默认 |
| `eps` | 0.6 | `run_train_sysu_asm.sh` |
| `k1` / `k2` | 30 / 6 | 默认 |
| **ASM 特有参数** | | |
| `gamma_v` | 2.0 | 代码硬编码（SYSU 配置） |
| `gamma_a` | 1.0 | 代码硬编码（SYSU 配置） |

### 7.2 训练过程日志特征

从 `logs/sysu_asm_s2/log.txt` 可观察到 ASM 的特有输出：

```text
==> Extract augmented (grayscale) features for RGB data
Extract augmented features: [50/87]
Augmented features shape: torch.Size([22258, 2048])
...
Cluster features augmented shape: torch.Size([628, 2048])
Augmented Matching: RGB + Augmented <-> IR
Augmented Matching Fusion: gamma_v=2.0, gamma_a=1.0
  M_vr range: [-0.4065, 0.9130]
  M_ar range: [-0.4373, 0.9089]
  M_var range: [0.1206, 0.6139]
Augmented Matching Done
```

- **Epoch 44** 达到 best epoch：`best R1: 62.0%   best mAP: 57.1%(best_epoch:44)`
- 后续 epoch（45–49）Rank-1 在 61.5%–61.8% 之间波动，未超越 epoch 44

---

## 九、实验结果

以下数据**全部直接摘抄自 `RPNR-SYSU-ASM/logs/sysu_asm_s2/log.txt`** 和 **`test_log.txt`**（best epoch = 44）。

### 8.1 All Search（10 trials 平均）

| 指标 | 值 |
|------|-----|
| **Rank-1** | **60.69%** |
| **Rank-5** | **85.44%** |
| **Rank-10** | **92.05%** |
| **Rank-20** | **96.37%** |
| **mAP** | **56.56%** |
| **mINP** | **41.12%** |

**10 trials 明细**：

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 61.98 | 85.01 | 91.87 | 95.69 | 57.10 | 41.64 |
| 1 | 58.43 | 85.17 | 92.16 | 96.61 | 56.30 | 41.54 |
| 2 | 62.06 | 85.35 | 91.90 | 96.13 | 55.66 | 39.63 |
| 3 | 61.90 | 86.56 | 93.19 | 96.74 | 58.67 | 43.26 |
| 4 | 62.58 | 86.46 | 92.58 | 96.71 | 58.66 | 43.37 |
| 5 | 58.61 | 84.70 | 91.98 | 97.00 | 55.64 | 40.04 |
| 6 | 59.74 | 85.09 | 91.69 | 96.66 | 55.30 | 39.53 |
| 7 | 59.90 | 86.93 | 93.27 | 96.45 | 55.14 | 39.00 |
| 8 | 60.69 | 83.75 | 90.82 | 95.82 | 55.91 | 40.83 |
| 9 | 60.98 | 85.33 | 91.06 | 95.90 | 57.23 | 42.31 |
| **Avg** | **60.69** | **85.44** | **92.05** | **96.37** | **56.56** | **41.12** |

### 8.2 Indoor Search（10 trials 平均）

| 指标 | 值 |
|------|-----|
| **Rank-1** | **67.79%** |
| **Rank-5** | **90.03%** |
| **Rank-10** | **95.62%** |
| **Rank-20** | **98.64%** |
| **mAP** | **73.02%** |
| **mINP** | **68.95%** |

**10 trials 明细**：

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 70.43 | 90.44 | 95.02 | 98.32 | 74.60 | 70.48 |
| 1 | 64.86 | 89.04 | 95.11 | 98.55 | 70.28 | 65.69 |
| 2 | 64.49 | 89.13 | 96.01 | 99.37 | 70.25 | 65.99 |
| 3 | 70.74 | 91.44 | 96.20 | 98.91 | 74.70 | 70.04 |
| 4 | 69.20 | 89.45 | 93.98 | 97.46 | 73.65 | 69.46 |
| 5 | 66.21 | 89.58 | 96.20 | 98.96 | 72.63 | 69.36 |
| 6 | 67.62 | 90.35 | 94.84 | 98.14 | 73.90 | 70.77 |
| 7 | 69.25 | 90.58 | 96.51 | 98.87 | 74.07 | 69.65 |
| 8 | 67.93 | 88.72 | 95.56 | 99.23 | 72.05 | 67.24 |
| 9 | 67.16 | 91.53 | 96.78 | 98.55 | 74.04 | 70.78 |
| **Avg** | **67.79** | **90.03** | **95.62** | **98.64** | **73.02** | **68.95** |

### 8.3 与 RPNR-SYSU 基线对比

| 场景 | 方法 | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|------|------|--------|--------|---------|---------|-----|------|
| **All Search** | RPNR-SYSU | **63.83** | **86.62** | **92.52** | **96.65** | **56.54** | **38.87** |
| **All Search** | RPNR-SYSU-ASM | 60.69 | 85.44 | 92.05 | 96.37 | 56.56 | 41.12 |
| **Indoor Search** | RPNR-SYSU | **68.04** | **90.41** | **95.61** | **98.64** | **73.04** | **68.67** |
| **Indoor Search** | RPNR-SYSU-ASM | 67.79 | 90.03 | 95.62 | 98.64 | 73.02 | 68.95 |

**对比分析**：
- **All Search**：ASM 的 Rank-1 比基线低 **3.14%**（60.69 vs 63.83），mAP 基本持平（+0.02%），mINP 高 **2.25%**。在该场景下，仅引入 CAM 未能带来 Rank-1 提升，但 mINP 有所改善。
- **Indoor Search**：ASM 的 Rank-1 比基线低 **0.25%**（67.79 vs 68.04），mAP 基本持平（-0.02%），mINP 高 **0.28%**。 Indoor 场景下 ASM 与基线几乎持平。
- 两者最佳 epoch 不同：RPNR-SYSU 在 epoch 33 达到最佳，ASM 在 epoch 44 达到最佳。

### 8.4 gamma 参数消融对比

| 场景 | 配置 | gamma_v | gamma_a | Rank-1 | mAP | mINP |
|------|------|---------|---------|--------|-----|------|
| **All Search** | **RPNR-SYSU（基线）** | — | — | **63.83** | **56.54** | **38.87** |
| **All Search** | RPNR-SYSU-ASM（2:1） | 2.0 | 1.0 | 60.69 | 56.56 | 41.12 |
| **All Search** | ASM-gamma1.0（1:1） | 1.0 | 1.0 | **64.33** | **58.54** | **42.03** |
| **Indoor Search** | **RPNR-SYSU（基线）** | — | — | **68.04** | **73.04** | **68.67** |
| **Indoor Search** | RPNR-SYSU-ASM（2:1） | 2.0 | 1.0 | 67.79 | 73.02 | 68.95 |
| **Indoor Search** | ASM-gamma1.0（1:1） | 1.0 | 1.0 | 66.80 | 72.33 | 68.20 |

**关键发现**：
- **All Search**：等权重配置（gamma_v=1.0, gamma_a=1.0）显著优于 2:1 配置，R1 高出 **3.64%**（64.33 vs 60.69），mAP 高出 **1.98%**（58.54 vs 56.56）。说明 gamma_v 取 1.0 在 All Search 上效果更好。
- **Indoor Search**：2:1 配置略优于 1:1 配置，R1 高 **0.99%**（67.79 vs 66.80），mAP 高 **0.69%**（73.02 vs 72.33）。但两者均低于基线（R1=68.04）。
- **综合**：gamma=1:1 在 All Search 上表现最佳，但 Indoor 略弱于 gamma=2:1；两者均未全面超越 RPNR-SYSU 基线。

### 8.5 与论文消融实验（Table 3）的对比

论文 Section 4.4 的消融实验（Table 3）使用 PGM / CAM 与 L_accl / L_csc / L_chc 的组合，结果如下：

| 方法 | 组件 | All Search Rank-1 | All Search mAP | Indoor Search Rank-1 | Indoor Search mAP |
|------|------|-------------------|----------------|----------------------|-------------------|
| **M1** | PGM + L_accl | 56.87 | 51.62 | 55.53 | 63.13 |
| **M2** | CAM + L_accl | **59.63** | **57.16** | **60.25** | **67.27** |
| **M3** | PGM + L_csc | 58.21 | 53.97 | 57.02 | 63.99 |
| **M4** | CAM + L_csc | 62.87 | 61.09 | 68.89 | 75.04 |
| **M5** | PGM + L_csc + L_chc | 61.56 | 59.05 | 62.99 | 69.63 |
| **M6** | CAM + L_csc + L_chc (完整 ASM) | **65.07** | **63.37** | **71.08** | **76.91** |
| **当前代码** | CAM + L_accl (RPNR 损失) | **60.69** | **56.56** | **67.79** | **73.02** |

**关键发现**：
1. **当前代码结果与论文 M2 高度吻合**：当前代码 All Search R1=60.69、mAP=56.56，与论文 M2 (CAM + L_accl) 的 R1=59.63、mAP=57.16 处于同一水平。这验证了代码中的 CAM 实现是正确的。
2. **当前代码 ≠ 完整 ASM**：论文声称的完整 ASM（M6）All Search R1=65.07、mAP=63.37，比当前代码高出约 **4.4% R1** 和 **6.8% mAP**。差距主要来自未实现的 **L_csc** 和 **L_chc**。
3. **当前代码 Indoor 结果优于论文 M2**：当前代码 Indoor R1=67.79、mAP=73.02，显著高于论文 M2 的 R1=60.25、mAP=67.27。这可能是因为基线 RPNR-SYSU 本身已优于论文使用的 PGM baseline。

---

## 十、总结

1. **改进完整性**：`RPNR-SYSU-ASM` **实现了论文中的 CAM（Cross-modality Augmented Matching）组件**，包括灰度增强特征提取、双 sigmoid 相似度融合、软加权 Hybrid Memory 构建。但**未实现**论文完整 ASM 所需的 SMU、L_csc 和 L_chc。
2. **与基线关系**：在 RPNR-SYSU 的基础上，仅替换了 OTPM 和 Hybrid Memory 构建模块，其余训练流程（DBSCAN 聚类、伪标签修正、ACCL、NPC Loss、CMhcl）完全保留。
3. **代码 ≈ 论文 M2 级别**：当前代码的实验结果（All Search R1=60.69, mAP=56.56）与论文消融实验 **M2（CAM + L_accl）** 的结果（R1=59.63, mAP=57.16）高度吻合，验证了 CAM 实现的正确性。
4. **与完整 ASM 的差距**：论文完整 ASM（M6）All Search R1=65.07, mAP=63.37，比当前代码高约 **4.4% R1** 和 **6.8% mAP**，差距主要来自未实现的 L_csc 和 L_chc。
5. **结果可复现**：训练日志完整记录了 50 epochs 的过程，最终 best model（epoch 44）在 SYSU-MM01 上取得了 **Rank-1=60.69%、mAP=56.56%（All Search）** 与 **Rank-1=67.79%、mAP=73.02%（Indoor Search）** 的指标，所有数据均可与 `log.txt` / `test_log.txt` 逐行核对。

---

*报告结束。*
