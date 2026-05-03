# RPNR-SYSU 复现代码报告清单

> 项目名称：Robust Pseudo-label Learning with Neighbor Relation for Unsupervised Visible-Infrared Person Re-Identification  
> 论文来源：arXiv:2405.05613  
> 复现代码路径：`/root/work/RPNR-SYSU`  
> 原始代码路径：`/root/work/RPNR-main`
  
> 报告生成时间：2026-04-29

---

## 一、项目概述

RPNR（Robust Pseudo-label Learning with Neighbor Relation）是一篇面向**无监督跨模态行人重识别（USL-VI-ReID）**的工作。核心思想是：在没有任何身份标注的情况下，仅利用可见光（RGB）与红外（IR）图像的模态内聚类生成伪标签，并通过**邻居关系一致性（NPC Loss）**、**最优传输伪标签匹配（OTPM）**与**交替跨模态对比学习（ACCL）**提升伪标签质量与跨模态特征对齐能力。

本报告基于 `/root/work/RPNR-SYSU` 目录下的完整代码、日志与运行结果，逐项梳理代码框架、算法逻辑、运行环境、数据集处理及最终实验指标，并确保所有数据均能与实际代码与日志文件一一对应。

---

## 二、RPNR-SYSU 与 RPNR-main 的代码差异

经 `diff` 逐文件比对，`RPNR-SYSU` 在**算法逻辑上完全复现**了 `RPNR-main`，差异集中在**环境兼容性修复、路径本地化与性能优化**上，具体如下：

| 差异项 | RPNR-main（原始） | RPNR-SYSU（复现） | 影响说明 |
|--------|------------------|-------------------|----------|
| **PIL 版本兼容** | `Image.ANTIALIAS` | `Image.Resampling.LANCZOS` | Pillow ≥9.1 已移除 `ANTIALIAS`，复现版兼容当前环境 Pillow 9.1.1 |
| **DataParallel 解包** | 评估时直接传入 `model`（含 DataParallel） | 新增 `get_single_model()`，eval 时自动取 `model.module` | 避免特征提取阶段因 DataParallel 导致的严重性能回退 |
| **faiss 版本兼容** | `faiss.cast_integer_to_idx_t_ptr` | `faiss.cast_integer_to_long_ptr` | faiss-gpu 1.6.3 API 变更兼容 |
| **checkpoint 加载兼容** | 直接 `model.load_state_dict(state_dict)` | 自动处理 `module.` 前缀（添加/移除） | 兼容 DataParallel 与非 DataParallel 保存的模型 |
| **数据集路径** | 原服务器绝对路径 `/data/yxb/datasets/...` | 改为 `/root/work/SYSU-MM01/` | 适配本地目录 |
| **日志目录** | `logs/sysu_s2_log.txt`（单文件） | `logs/sysu_s2/log.txt` + `logs/sysu_s2/test_log.txt` | 训练日志与测试日志分开存储 |
| **baseline 模型** | 无 `baseline/` 目录 | 新增 `baseline/sysu_s1/model_best.pth.tar` | 保存了 stage-1 预训练权重，供二阶段训练加载 |

**结论**：`RPNR-SYSU` 的算法实现（`trainers.py`、`models/cm.py`、`models/losses.py`）与原始代码**逐字符一致**，未做任何算法层面的修改，属于**忠实复现**。

---

## 三、代码框架与文件结构

```
RPNR-SYSU/
├── README.md                          # 论文说明与官方指标
├── ChannelAug.py                      # 跨模态数据增强（ChannelExchange / ChannelAdapGray / ChannelT / ChannelRandomErasing）
├── prepare_sysu.py                    # SYSU-MM01 数据集预处理脚本
├── train_sysu.py                      # 二阶段训练主脚本（Stage-2：RPNR 无监督微调）
├── test_sysu.py                       # 测试脚本（All Search + Indoor Search，各 10 trials）
├── run_train_sysu.sh                  # 训练启动脚本
├── run_test_sysu.sh                   # 测试启动脚本
├── environment.yml                    # Conda 环境完整配置（含版本号）
├── requirements.txt                   # pip 依赖精简清单
├── meters.py                          # 训练过程指标打印辅助
├── clustercontrast/                   # 核心算法库
│   ├── trainers.py                    # 训练器：ClusterContrastTrainer_RPNR（核心）
│   ├── models/
│   │   ├── cm.py                      # ClusterMemory（CM / CMhcl）
│   │   ├── losses.py                  # 损失函数：RCLoss（NPC Loss）、CrossEntropyLabelSmooth 等
│   │   ├── agw.py                     # AGW 网络 backbone
│   │   ├── resnet*.py                 # ResNet 系列 backbone
│   │   └── ...
│   ├── datasets/
│   │   ├── sysu_rgb.py                # RGB 模态数据集读取（Market-1501 格式）
│   │   ├── sysu_ir.py                 # IR 模态数据集读取
│   │   └── sysu_all.py                # 全量数据集（供测试时统一接口）
│   ├── utils/
│   │   ├── faiss_rerank.py            # Jaccard distance / k-reciprocal encoding
│   │   ├── data/preprocessor.py       # 数据预处理与双增强（Preprocessor_aug）
│   │   └── ...
│   └── evaluators.py                  # 特征提取函数 extract_features
└── logs/sysu_s2/
    ├── log.txt                        # 完整训练日志（50 epochs + 最终 10-trial 测试）
    └── test_log.txt                   # 独立测试日志（加载 best model）
```

---

## 四、运行环境清单

以下环境为实际运行时的系统环境（通过 `python --version`、`torch.__version__`、`nvidia-smi` 等现场确认）。代码目录下的 `requirements.txt` 和 `environment.yml` 仍保留原始论文配置（PyTorch 1.8.0 / CUDA 10.2），但实际运行时使用当前系统已安装的兼容版本即可正常执行。

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| Python | 3.8.10 | 主解释器版本 |
| PyTorch | 1.11.0+cu113 | GPU 版本，CUDA 11.3 编译 |
| torchvision | 0.12.0+cu113 | 与 PyTorch 配套 |
| CUDA Toolkit | 11.3 | 与 PyTorch 编译版本一致 |
| scikit-learn | 1.2.2 | DBSCAN 聚类依赖 |
| POT | 0.9.3 | Optimal Transport（Sinkhorn）依赖 |
| faiss-gpu | 1.6.3 | k-reciprocal 与 Jaccard distance 加速 |
| Pillow | 9.1.1 | 图像读取与增强 |
| numpy | 1.22.4 | 数值计算 |
| scipy | 1.10.1 | 科学计算 |
| tqdm | 4.61.2 | 进度条 |

**硬件环境**：2 × NVIDIA GeForce RTX 3090（24GB 显存），Driver 570.124.04 |

**运行命令示例**：
```bash
# 训练（二阶段）
sh run_train_sysu.sh
# 等效命令：
CUDA_VISIBLE_DEVICES=0,1 \
python train_sysu.py -mb CMhcl -b 128 -a agw -d sysu_all \
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
  --data-dir "/root/work/SYSU-MM01/"

# 测试
sh run_test_sysu.sh
# 等效命令：
CUDA_VISIBLE_DEVICES=0,1 \
python test_sysu.py -b 64 -a agw -d sysu_all \
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
  --data-dir "/root/work/SYSU-MM01/"
```

---

## 五、数据集清单与预处理

### 5.1 原始数据集
- **数据集名称**：SYSU-MM01
- **原始路径**：`/root/work/SYSU-MM01`
- **模态**：可见光（RGB，cam1/2/4/5）+ 红外（IR，cam3/6）
- **官方划分**：
  - `exp/train_id.txt`：训练身份 ID
  - `exp/val_id.txt`：验证身份 ID（与训练集合并用于无监督训练）
  - `exp/test_id.txt`：测试身份 ID

### 5.2 预处理脚本 `prepare_sysu.py`

该脚本将原始 SYSU-MM01 整理为 **Market-1501 格式**，便于统一的数据加载器读取：

| 输出目录 | 内容 | 图像数量（复现环境实测） |
|----------|------|------------------------|
| `SYSU-MM01/rgb_modify/bounding_box_train` | RGB 训练集 | 22,258 |
| `SYSU-MM01/ir_modify/bounding_box_train` | IR 训练集 | 11,909 |
| `SYSU-MM01/rgb_modify/query` | RGB query（测试用，前 4 张） | 384 |
| `SYSU-MM01/rgb_modify/bounding_box_test` | RGB gallery（其余） | 6,391 |
| `SYSU-MM01/ir_modify/query` | IR query（测试用，前 4 张） | 384 |
| `SYSU-MM01/ir_modify/bounding_box_test` | IR gallery（其余） | 3,419 |

**命名规则**：`ID_cCameraNumber_ImageName.jpg`，例如 `0001_c1_000001.jpg`。

### 5.3 数据集统计（来自 `log.txt` 实际加载输出）

```text
=> sysu_ir loaded
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   395 |    11909 |         2
  query    |    96 |      384 |         2
  gallery  |    96 |     3419 |         2

=> sysu_rgb loaded
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   395 |    22258 |         4
  query    |    96 |      384 |         3
  gallery  |    96 |     6391 |         4
```

---

## 六、核心算法与代码逻辑

### 6.1 整体训练流程（`train_sysu.py` → `main_worker`）

RPNR 采用 **两阶段训练**：
1. **Stage-1（Baseline）**：有监督训练一个跨模态基础模型（代码外完成，权重放在 `baseline/sysu_s1/model_best.pth.tar`）。
2. **Stage-2（RPNR 无监督微调）**：加载 Stage-1 权重，在无标签数据上进行 50 个 epoch 的伪标签自训练。

每个 epoch 的详细流程如下：

```
1. 特征提取（eval mode）
   ├── 提取 RGB 训练集特征 → features_rgb
   └── 提取 IR 训练集特征  → features_ir

2. 模态内聚类（DBSCAN + Jaccard distance reranking）
   ├── rerank_dist_ir  = compute_jaccard_distance(features_ir,  k1=30, k2=6)
   ├── pseudo_labels_ir = DBSCAN(eps=0.6, min_samples=4).fit_predict(rerank_dist_ir)
   └── 同理生成 pseudo_labels_rgb

3. 伪标签修正（Label Correction）
   ├── generate_cluster_features_corr(): 对每个聚类，基于 cosine similarity 的 rho 分数选出 top-20 原型
   ├── 用原型均值修正离群伪标签 → pseudo_labels_ir_hat / pseudo_labels_rgb_hat
   └── 过滤 label == -1 的噪声样本

4. 最优传输伪标签匹配（OTPM）
   ├── 计算 RGB 与 IR 聚类中心的相似度矩阵 similarity = exp(cosine / 1)
   ├── cost = 1 / similarity
   └── result = ot.sinkhorn(a, b, M, reg=5, numItermax=5000) → 得到 r2i / i2r 映射

5. 构建 Hybrid Memory
   ├── cluster_features_hybrid[i] = mean(cluster_features_ir[i], cluster_features_rgb[i2r[i]])
   └── 初始化 memory_ir、memory_rgb、memory_hybrid（CMhcl 模式）

6. 数据增强与训练（`ClusterContrastTrainer_RPNR.train`）
   ├── RGB 双增强：标准增强 + ChannelExchange
   ├── IR 双增强：ChannelAdapGray + ChannelT（ColorJitter）
   └── 损失 = L_ir + L_rgb + 0.25*L_cross + 0.5*L_hybrid + 10.0*L_RC

7. 测试评估（每 epoch 结束后）
   ├── All Search：1 trial（随机选一张 RGB gallery）
   └── 保存 best model（按 Rank-1）
```

### 6.2 核心算法模块详解

#### 6.2.1 ClusterContrastTrainer_RPNR（`clustercontrast/trainers.py`）

```python
class ClusterContrastTrainer_RPNR(object):
    def train(self, epoch, data_loader_ir, data_loader_rgb, optimizer,
              print_freq=10, train_iters=400, i2r=None, r2i=None):
        # 1. 读取双增强数据
        inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
        inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
        inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
        labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)

        # 2. 前向传播（AGW backbone，modal=0 表示双模态同时输入）
        _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = \
            self._forward(inputs_rgb, inputs_ir, ...)

        # 3. 模态内对比损失
        loss_ir  = self.memory_ir(f_out_ir, labels_ir)
        loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

        # 4. 交替跨模态对比学习（ACCL）
        if r2i:
            rgb2ir_labels = torch.tensor([r2i[key.item()] for key in labels_rgb]).cuda()
            ir2rgb_labels = torch.tensor([i2r[key.item()] for key in labels_ir]).cuda()
            if epoch % 2 == 1:
                cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                loss_hybrid = self.memory_hybrid(f_out_ir, labels_ir)
            else:
                cross_loss = self.memory_ir(f_out_rgb, rgb2ir_labels.long())
                loss_hybrid = self.memory_hybrid(f_out_rgb, rgb2ir_labels)

        # 5. 总损失
        loss = loss_ir + loss_rgb + 0.25*cross_loss + 0.5*loss_hybrid

        # 6. NPC Loss（Neighbor Relation Consistency）
        f_out_rgb_de = f_out_rgb.detach()
        f_out_ir_de  = f_out_ir.detach()
        loss_RC_rgb = self.RCLoss(f_out_rgb, f_out_rgb_de)
        loss_RC_ir  = self.RCLoss(f_out_ir,  f_out_ir_de)
        loss_RC = loss_RC_rgb + loss_RC_ir
        loss = loss + 10.0 * loss_RC
```

**关键点**：
- **ACCL（Alternating Cross-Contrastive Learning）**：奇偶 epoch 交替将 IR 特征推入 RGB memory 或将 RGB 特征推入 IR memory，避免同时优化导致的冲突。
- **Hybrid Memory**：通过 OTPM 对齐后的跨模态混合中心，引导模型学习模态无关的簇中心。
- **NPC Loss（RCLoss）**：基于实例间成对距离，保持当前 batch 特征与 detach 特征之间的邻居结构一致性，抑制噪声伪标签带来的特征漂移。

#### 6.2.2 ClusterMemory（`clustercontrast/models/cm.py`）

支持两种模式：
- **CM**：标准簇记忆库，动量更新 `features[y] = momentum * features[y] + (1-momentum) * x`
- **CMhcl（Hard Cluster Learning）**：每个簇维护两个中心（mean + hard），损失为 `0.5 * (CE(hard) + relu(CE(mean) - r))`，其中 `r=0.2` 为松弛阈值

复现实验使用的是 **CMhcl**（启动参数 `-mb CMhcl`）。

#### 6.2.3 RCLoss（`clustercontrast/models/losses.py`）

```python
class RCLoss(nn.Module):
    def forward(self, s_emb, t_emb):
        N = len(s_emb)
        S_dist = torch.cdist(s_emb, s_emb) / S_dist.mean(1, keepdim=True)
        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb)
            W = torch.exp(-T_dist.pow(2) / self.sigma)          # 正样本权重
            pos_weight = W * (1 - I)
            neg_weight = (1 - W) * (1 - I)
        pull = relu(S_dist).pow(2) * pos_weight                  # 拉近邻居
        push = relu(self.delta - S_dist).pow(2) * neg_weight     # 推远非邻居
        return (pull.sum() + push.sum()) / (N * (N - 1))
```

- `sigma=1`, `delta=1`
- `t_emb` 为 detach 的目标特征（上一状态），`s_emb` 为当前待优化特征
- 实现**邻居关系保持**：在特征更新过程中，维持样本间的相对距离结构

#### 6.2.4 最优传输伪标签匹配（OTPM，位于 `train_sysu.py`）

```python
similarity = (torch.mm(cluster_features_rgb_norm, cluster_features_ir_norm.T) / 1).exp().cpu()
cost = 1 / similarity
a = np.ones(cost.shape[0]) / cost.shape[0]
b = np.ones(cost.shape[1]) / cost.shape[1]
M = np.array(cost)
result = ot.sinkhorn(a, b, M, reg=5, numItermax=5000, stopThr=1e-5)
```

- 将 RGB 与 IR 的聚类中心视为两个分布，用 Sinkhorn 算法求解最优传输计划
- `reg=5` 为正则化系数，控制熵正则强度
- 根据传输矩阵的 argmax 得到双向映射 `r2i`（RGB→IR）与 `i2r`（IR→RGB）

---

## 七、实验框架与流程

### 7.1 训练阶段关键超参数（与 `log.txt` 中 Args 完全一致）

| 参数 | 值 | 来源 |
|------|-----|------|
| `arch` | `agw` | `run_train_sysu.sh` |
| `pooling_type` | `gem` | 默认参数 |
| `batch_size` | 128 | `run_train_sysu.sh` |
| `iters` | 200 | `run_train_sysu.sh` |
| `epochs` | 50 | 默认参数 |
| `lr` | 0.00035 | 默认参数 |
| `step_size` | 20 | 默认参数（lr 每 20 epoch ×0.1） |
| `weight_decay` | 5e-4 | 默认参数 |
| `num_instances` | 16 | `run_train_sysu.sh` |
| `memorybank` | `CMhcl` | `run_train_sysu.sh` |
| `momentum` | 0.1 | `run_train_sysu.sh`（memory 更新动量） |
| `temp` | 0.05 | 默认参数（对比温度） |
| `eps` | 0.6 | `run_train_sysu.sh`（DBSCAN 邻域半径） |
| `k1` | 30 | 默认参数（Jaccard distance） |
| `k2` | 6 | 默认参数（Jaccard distance） |
| `height` / `width` | 288 / 144 | 默认参数 |
| `seed` | 1 | 默认参数 |

### 7.2 训练过程日志特征

从 `logs/sysu_s2/log.txt` 可观察到以下规律：

- **Epoch 0**：IR 聚类数 ≈ 430，RGB 聚类数 ≈ 744，与真实身份数 395 接近但略有噪声。
- **Epoch 20**：学习率下降（step_size=20），Rank-1 跳升至 **58%** 左右。
- **Epoch 33**：达到 best epoch，best R1 = **65.79%**，best mAP = **57.72%**。
- **Epoch 50**：最终模型 Rank-1 约 64.7%，mAP 约 58.1%，best 模型保持在 epoch 33。

### 7.3 测试协议

- **All Search**：IR query（cam3/6）搜索 RGB gallery（cam1/2/4/5），gallery 每人随机选 1 张，共 **10 trials** 取平均。
- **Indoor Search**：IR query（cam3/6）搜索 indoor RGB gallery（cam1/2），同样 10 trials。
- **评价指标**：Rank-1 / Rank-5 / Rank-10 / Rank-20 / mAP / mINP。
- **特征提取**：水平翻转（fliplr）后取平均，L2 归一化，2048-d。

---

## 八、实验结果

以下数据**全部直接摘抄自 `RPNR-SYSU/logs/sysu_s2/test_log.txt`**（best epoch = 33），确保与代码运行结果一一对应。

### 8.1 All Search（10 trials 平均）

| 指标 | 值 |
|------|-----|
| **Rank-1** | **63.83%** |
| **Rank-5** | **86.62%** |
| **Rank-10** | **92.52%** |
| **Rank-20** | **96.65%** |
| **mAP** | **56.54%** |
| **mINP** | **38.87%** |

**10 trials 明细**：

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 65.79 | 85.85 | 91.98 | 95.56 | 57.72 | 40.25 |
| 1 | 59.16 | 86.14 | 92.82 | 96.69 | 55.52 | 39.53 |
| 2 | 64.40 | 86.22 | 92.37 | 96.42 | 55.92 | 37.51 |
| 3 | 64.53 | 87.72 | 93.48 | 97.19 | 58.14 | 40.55 |
| 4 | 64.40 | 87.64 | 93.45 | 97.05 | 58.47 | 41.57 |
| 5 | 62.08 | 85.09 | 91.51 | 97.08 | 55.62 | 38.17 |
| 6 | 64.74 | 86.59 | 91.90 | 96.45 | 56.06 | 38.13 |
| 7 | 64.21 | 88.46 | 93.40 | 96.84 | 55.15 | 35.81 |
| 8 | 64.61 | 86.17 | 92.24 | 96.92 | 56.28 | 38.58 |
| 9 | 64.42 | 86.27 | 92.06 | 96.32 | 56.47 | 38.63 |
| **Avg** | **63.83** | **86.62** | **92.52** | **96.65** | **56.54** | **38.87** |

### 8.2 Indoor Search（10 trials 平均）

| 指标 | 值 |
|------|-----|
| **Rank-1** | **68.04%** |
| **Rank-5** | **90.41%** |
| **Rank-10** | **95.61%** |
| **Rank-20** | **98.64%** |
| **mAP** | **73.04%** |
| **mINP** | **68.67%** |

**10 trials 明细**：

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 70.20 | 90.90 | 95.20 | 98.37 | 74.67 | 70.43 |
| 1 | 65.58 | 88.90 | 94.66 | 98.19 | 70.57 | 65.76 |
| 2 | 65.04 | 89.58 | 95.74 | 99.09 | 70.25 | 65.43 |
| 3 | 69.02 | 91.26 | 95.56 | 99.00 | 73.60 | 68.86 |
| 4 | 70.61 | 90.08 | 95.79 | 97.92 | 74.65 | 70.27 |
| 5 | 67.39 | 90.08 | 96.06 | 98.73 | 72.60 | 68.42 |
| 6 | 70.52 | 89.45 | 95.15 | 98.41 | 74.86 | 71.09 |
| 7 | 68.03 | 91.53 | 96.15 | 98.96 | 73.50 | 69.23 |
| 8 | 66.67 | 90.40 | 95.61 | 99.00 | 72.01 | 67.27 |
| 9 | 67.39 | 91.89 | 96.15 | 98.73 | 73.69 | 69.91 |
| **Avg** | **68.04** | **90.41** | **95.61** | **98.64** | **73.04** | **68.67** |

### 8.3 与论文/官方指标对比

README 中给出的官方整理代码指标（非论文原始值）如下：

| 场景 | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|------|--------|--------|---------|---------|-----|------|
| 官方 All Search | 65.49 | 87.48 | 93.29 | 97.14 | 59.97 | 44.02 |
| 官方 Indoor | 68.46 | 89.67 | 94.75 | 97.89 | 73.01 | 68.61 |
| **复现 All Search** | **63.83** | **86.62** | **92.52** | **96.65** | **56.54** | **38.87** |
| **复现 Indoor** | **68.04** | **90.41** | **95.61** | **98.64** | **73.04** | **68.67** |

**差距分析**：
- All Search：Rank-1 差距约 **1.66%**，mAP 差距约 **3.43%**。
- Indoor：Rank-1 差距约 **0.42%**，mAP 差距约 **-0.03%**（几乎一致）。
- 分析：Indoor 场景复现效果接近官方，All Search 仍有差距，可能主要源于 PyTorch 1.11.0/CUDA 11.3 与官方 PyTorch 1.8.0/CUDA 10.2 的版本差异。

---

## 九、总结

1. **复现完整性**：`RPNR-SYSU` 完整复现了 RPNR 算法的所有核心模块（DBSCAN 聚类、伪标签修正、OTPM、Hybrid Memory、ACCL、NPC Loss、CMhcl），代码与原始版本逻辑一致。
2. **兼容性改进**：针对本地环境做了 PIL 兼容、DataParallel 性能优化、路径修正等必要调整，不影响算法本身。
3. **结果可复现**：训练日志完整记录了 50 epochs 的过程，最终 best model（epoch 33）在 SYSU-MM01 上取得了 **Rank-1=63.83%、mAP=56.54%（All Search）** 与 **Rank-1=68.04%、mAP=73.04%（Indoor Search）** 的指标。Indoor 场景接近官方，All Search 与官方差距约 1.66%（R1），可能源于 PyTorch/CUDA 版本差异。

---

*报告结束。*
