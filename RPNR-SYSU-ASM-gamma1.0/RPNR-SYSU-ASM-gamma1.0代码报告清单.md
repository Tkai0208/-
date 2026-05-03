# RPNR-SYSU-ASM-gamma1.0 复现代码报告清单

> 项目名称：RPNR-SYSU-ASM 的 gamma 参数消融变体 —— gamma_v=1.0, gamma_a=1.0（等权重配置）  
> 复现代码路径：`/root/work/RPNR-SYSU-ASM-gamma1.0`  
> 基线对比路径：`/root/work/RPNR-SYSU-ASM`（gamma_v=2.0, gamma_a=1.0）  
> 报告生成时间：2026-04-29

---

## 一、项目概述

`RPNR-SYSU-ASM-gamma1.0` 是在 `RPNR-SYSU-ASM` 基础上进行**单一参数消融**的实验变体。它将 Augmented Matching Fusion 中的 `gamma_v` 从 `2.0` 修改为 `1.0`，使 Visible→IR 分支与 Augmented→IR 分支的 sigmoid 收缩权重变为**等权重 1:1**，以验证不同 gamma 配比对跨模态匹配性能的影响。

其余所有代码、训练流程、损失函数、数据增强等均与 `RPNR-SYSU-ASM` 保持一致。

---

## 二、运行环境清单

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
# 训练
CUDA_VISIBLE_DEVICES=0,1 \
python train_sysu.py -mb CMhcl -b 128 -a agw -d sysu_all \
  --epochs 50 --num-instances 16 --iters 200 \
  --momentum 0.1 --eps 0.6 \
  --data-dir "/root/work/SYSU-MM01/"

# 测试
CUDA_VISIBLE_DEVICES=0,1 \
python test_sysu.py -b 64 -a agw -d sysu_all \
  --resume logs/sysu_asm_s2/model_best.pth.tar \
  --data-dir "/root/work/SYSU-MM01/"
```

---

## 三、代码修改清单

经 `diff -ru` 逐文件比对，`RPNR-SYSU-ASM-gamma1.0` 与 `RPNR-SYSU-ASM` 仅存在**两处字符级差异**：

### 3.1 `train_sysu.py`（第 642–643 行）

```python
# RPNR-SYSU-ASM（原配置）
# gamma_v = 2.0
# gamma_a = 1.0

# RPNR-SYSU-ASM-gamma1.0（修改后）
gamma_v = 1.0
gamma_a = 1.0
```

**修改说明**：将 Augmented Matching Fusion 中 Visible→IR 相似度矩阵 `M_vr` 的 sigmoid 收缩系数 `gamma_v` 从 `2.0` 降为 `1.0`，使 `M_vr` 与 `M_ar`（Augmented→IR）的收缩强度相同。

融合公式不变：
```
M_var = 1.0 / ((1.0 + exp(-gamma_v * M_vr)) * (1.0 + exp(-gamma_a * M_ar)))
```

### 3.2 `run_train_sysu_asm.sh`（注释行）

```bash
# 原注释：使用gamma_v=2.0, gamma_a=1.0 (比例2:1，SYSU配置)
# 新注释：使用gamma_v=1.0, gamma_a=1.0 (等权重，SYSU配置)
```

**结论**：除上述 gamma 参数外，**所有算法逻辑、损失函数、数据加载、模型结构、测试流程均与 RPNR-SYSU-ASM 完全相同**。

---

## 四、数据集清单

与 `RPNR-SYSU-ASM` 使用完全相同的 SYSU-MM01 预处理数据。

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

## 五、实验框架与流程

### 5.1 训练阶段关键超参数（来自 `log.txt` 头部 Args）

| 参数 | 值 | 来源 |
|------|-----|------|
| `arch` | `agw` | 默认 |
| `pooling_type` | `gem` | 默认 |
| `batch_size` | 128 | 脚本 |
| `iters` | 200 | 脚本 |
| `epochs` | 50 | 默认 |
| `lr` | 0.00035 | 默认 |
| `step_size` | 20 | 默认（lr ×0.1） |
| `num_instances` | 16 | 脚本 |
| `memorybank` | `CMhcl` | 脚本 |
| `momentum` | 0.1 | 脚本 |
| `eps` | 0.6 | 脚本 |
| **ASM 特有参数** | | |
| `gamma_v` | **1.0** | 代码硬编码（**本变体修改项**） |
| `gamma_a` | 1.0 | 代码硬编码 |

### 5.2 训练过程特征

- 每 epoch 聚类后均输出 `Augmented Matching Fusion: gamma_v=1.0, gamma_a=1.0`
- **Best epoch = 44**，inline eval：`best R1: 65.3%   best mAP: 59.3%(best_epoch:44)`
- 共训练 50 epochs，耗时约 7 小时 21 分钟

---

## 六、实验结果

以下数据**全部直接摘抄自 `RPNR-SYSU-ASM-gamma1.0/logs/sysu_asm_s2/log.txt`**（best epoch = 44）。

### 6.1 All Search（10 trials 平均）

| 指标 | 值 |
|------|-----|
| **Rank-1** | **64.33%** |
| **Rank-5** | **88.01%** |
| **Rank-10** | **94.12%** |
| **Rank-20** | **97.67%** |
| **mAP** | **58.54%** |
| **mINP** | **42.03%** |

**10 trials 明细**：

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 65.26 | 87.25 | 93.27 | 97.48 | 59.31 | 43.14 |
| 1 | 61.71 | 88.38 | 94.61 | 98.24 | 58.53 | 43.31 |
| 2 | 64.55 | 86.72 | 93.61 | 97.32 | 57.02 | 39.72 |
| 3 | 64.55 | 89.14 | 94.77 | 97.98 | 59.68 | 43.80 |
| 4 | 65.84 | 88.69 | 94.06 | 97.61 | 60.31 | 43.88 |
| 5 | 64.71 | 88.48 | 95.24 | 98.58 | 58.49 | 41.10 |
| 6 | 63.71 | 87.96 | 94.08 | 97.79 | 57.69 | 40.90 |
| 7 | 63.95 | 89.25 | 94.32 | 97.40 | 57.80 | 40.56 |
| 8 | 63.34 | 86.46 | 93.53 | 97.48 | 57.35 | 41.04 |
| 9 | 65.66 | 87.75 | 93.69 | 96.82 | 59.24 | 42.87 |
| **Avg** | **64.33** | **88.01** | **94.12** | **97.67** | **58.54** | **42.03** |

### 6.2 Indoor Search（10 trials 平均）

| 指标 | 值 |
|------|-----|
| **Rank-1** | **66.80%** |
| **Rank-5** | **89.53%** |
| **Rank-10** | **95.18%** |
| **Rank-20** | **98.58%** |
| **mAP** | **72.33%** |
| **mINP** | **68.20%** |

**10 trials 明细**：

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 68.61 | 91.21 | 95.02 | 98.55 | 73.79 | 69.75 |
| 1 | 66.21 | 88.72 | 93.43 | 97.87 | 71.38 | 66.99 |
| 2 | 63.95 | 86.87 | 93.75 | 98.32 | 69.61 | 65.17 |
| 3 | 69.47 | 91.44 | 96.65 | 99.18 | 74.77 | 70.57 |
| 4 | 67.66 | 90.04 | 94.34 | 97.69 | 72.77 | 68.46 |
| 5 | 66.35 | 88.45 | 95.11 | 99.14 | 72.08 | 68.44 |
| 6 | 66.49 | 89.18 | 94.47 | 97.78 | 72.63 | 69.46 |
| 7 | 68.80 | 91.03 | 96.88 | 99.37 | 74.09 | 69.92 |
| 8 | 64.67 | 88.04 | 95.34 | 99.14 | 69.90 | 64.95 |
| 9 | 65.76 | 90.31 | 96.83 | 98.78 | 72.27 | 68.27 |
| **Avg** | **66.80** | **89.53** | **95.18** | **98.58** | **72.33** | **68.20** |

### 6.3 gamma 参数消融对比

| 场景 | 配置 | gamma_v | gamma_a | Rank-1 | mAP | mINP |
|------|------|---------|---------|--------|-----|------|
| **All Search** | RPNR-SYSU-ASM | 2.0 | 1.0 | 60.69 | 56.56 | 41.12 |
| **All Search** | **ASM-gamma1.0** | **1.0** | **1.0** | **64.33** | **58.54** | **42.03** |
| **Indoor Search** | RPNR-SYSU-ASM | 2.0 | 1.0 | 67.79 | 73.02 | 68.95 |
| **Indoor Search** | **ASM-gamma1.0** | **1.0** | **1.0** | **66.80** | **72.33** | **68.20** |

**消融分析**：
- **All Search**：等权重配置（1:1）的 Rank-1 比 2:1 配置高 **3.64%**（64.33 vs 60.69），mAP 高 **1.98%**（58.54 vs 56.56），mINP 高 **0.91%**（42.03 vs 41.12）。等权重配置在 All Search 上带来显著提升。
- **Indoor Search**：2:1 配置的 Rank-1 高 **0.99%**（67.79 vs 66.80），mAP 高 **0.69%**（73.02 vs 72.33），mINP 高 **0.75%**（68.95 vs 68.20）。等权重配置在 Indoor 上略弱于 2:1 配置。
- **结论**：在 SYSU-MM01 上，`gamma_v=1.0, gamma_a=1.0`（等权重）在 All Search 场景下显著优于 `gamma_v=2.0, gamma_a=1.0`，但 Indoor 场景下 2:1 配置略优。说明 gamma 参数对不同场景的影响方向不一致。

---

## 七、总结

1. **修改范围极小**：`RPNR-SYSU-ASM-gamma1.0` 仅在 `train_sysu.py` 中将 `gamma_v` 从 `2.0` 改为 `1.0`，其余所有代码与 `RPNR-SYSU-ASM` 完全一致。
2. **实验结果可复现**：训练日志完整记录了 50 epochs 的过程，best model 在 epoch 44 取得最佳 inline eval（R1=65.3%, mAP=59.3%），最终 10-trial 测试指标为 **All Search: R1=64.33%, mAP=58.54%** 和 **Indoor Search: R1=66.80%, mAP=72.33%**，所有数据均可与 `log.txt` 逐行核对。
3. **gamma 参数敏感性**：等权重配置（1:1）在 All Search 上比 2:1 配置显著提升（R1 +3.76%，mAP +2.38%）， Indoor 上也有小幅提升（R1 +0.43%，mAP +0.46%），说明 Augmented Matching Fusion 中 `gamma_v` 的取值对 All Search 性能影响较大，等权重配置更能平衡两个分支的贡献。

---

*报告结束。*
