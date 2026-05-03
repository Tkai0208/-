# PGM-SYSU 代码报告清单

> 生成时间：2026-04-29
> 项目路径：`/root/work/PGM-SYSU`
> 任务：无监督跨模态行人重识别（Unsupervised Visible-Infrared Person Re-ID）
> 目标数据集：SYSU-MM01

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

**训练（两阶段端到端）：**
```bash
bash run_train_sysu.sh
```

**测试（加载 Stage 2 最优模型）：**
```bash
bash run_test_sysu.sh
```

### 1.4 关键训练参数

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

本代码基于原始 PGM 源代码（`/root/work/源代码/PGM-new`）进行修改，主要改动如下：

### 2.1 删除的代码（专注 SYSU）

| 删除项 | 说明 |
|--------|------|
| `train_regdb.py` | RegDB 训练脚本 |
| `test_regdb.py` | RegDB 测试脚本 |
| `run_train_regdb.sh` | RegDB 训练启动脚本 |
| `run_test_regdb.sh` | RegDB 测试启动脚本 |

### 2.2 核心功能修改

| 文件 | 修改内容 |
|------|----------|
| `train_sysu.py` | **新增 Stage 1 自动跳过逻辑**：若 `logs/sysu_s1/train_model_best.pth.tar` 已存在，自动跳过 Stage 1，直接执行 Stage 2 |
| `train_sysu.py` | **新增 `get_single_model()` 函数**：eval / clustering 时绕过 `DataParallel` 通信开销，提升效率 |
| `train_sysu.py` | **所有特征提取调用改为单卡模式**：`extract_features(get_single_model(model), ...)` |
| `train_sysu.py` | **PIL / torchvision 兼容性修复**：`interpolation=3` → `InterpolationMode.BICUBIC`；`Image.ANTIALIAS` → `Image.Resampling.LANCZOS` |
| `train_sysu.py` | **DataLoader `num_workers`**：4 → 8 |
| `train_sysu.py` | **checkpoint 命名规范化**：`model_best.pth.tar` → `save_name + 'model_best.pth.tar'` |
| `test_sysu.py` | **测试模型路径改为 Stage 2**：`log_name='sysu_s2'`（原为 `sysu_s1`） |
| `test_sysu.py` | **简化特征提取**：去除 `proj_feat` 返回值和 `feat_concate` 逻辑，只保留基础 `feat_fc` |
| `test_sysu.py` | **数据路径改为命令行传入**：去除硬编码路径 `/data/wml/dataset/SYSU-MM01/`，使用 `args.data_dir` |
| `run_train_sysu.sh` | 增加 `OMP_NUM_THREADS=8`；去掉 `--sample_ratio 0.5`；数据路径指向 `/root/work/SYSU-MM01/` |
| `run_test_sysu.sh` | 增加 shebang 和 `OMP_NUM_THREADS=8`；指定 `--data-dir` |
| `clustercontrast/datasets/sysu_all.py` | **删除硬编码 root 路径**：不再强制覆盖为 `/data/yxb/datasets/ReIDData/SYSU-MM01/` |

---

## 三、数据集清单

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

### 3.3 数据集文件

| 文件 | 说明 |
|------|------|
| `clustercontrast/datasets/sysu_all.py` | SYSU 全模态数据集定义 |
| `clustercontrast/datasets/sysu_ir.py` | 红外模态数据集 |
| `clustercontrast/datasets/sysu_rgb.py` | 可见光模态数据集 |

### 3.4 数据存放路径

```
/root/work/SYSU-MM01/
├── cam1/          # RGB 相机1
├── cam2/          # RGB 相机2
├── cam3/          # IR 相机3
├── cam4/          # RGB 相机4
├── cam5/          # RGB 相机5
├── cam6/          # IR 相机6
├── exp/           # 训练/测试划分文件
├── ir_modify/     # 处理后的 IR 图像
└── rgb_modify/    # 处理后的 RGB 图像
```

---

## 四、实验框架与算法详解

### 4.1 整体架构

PGM-SYSU 采用 **两阶段无监督训练框架**：

```
Stage 1: 单模态独立聚类与对比学习
    ↓
Stage 2: 跨模态伪标签匹配与联合训练
    ↓
测试: 加载 Stage 2 最优模型，10 trials 平均
```

### 4.2 AGW 模型架构详解

AGW（Attention Generalized mean Pooling with Weighted triplet）是本项目的骨干网络，专为跨模态行人重识别设计。

#### 4.2.1 整体结构

```
输入图像 (RGB 或 IR)
    ↓
模态特定浅层编码器 (visible_module / thermal_module)
    ├── ResNet50 的 conv1 + bn1 + relu + maxpool
    └── 各自独立，不共享权重
    ↓
共享深层编码器 (base_resnet)
    ├── layer1 (256通道) + Non-local × 0
    ├── layer2 (512通道) + Non-local × 2
    ├── layer3 (1024通道) + Non-local × 3
    └── layer4 (2048通道) + Non-local × 0
    ↓
全局特征聚合
    ├── GeM Pooling (p=3.0)  或  AdaptiveAvgPool
    └── 输出 2048-d 特征向量
    ↓
BNNeck (BatchNorm1d)
    └── 训练时输出 feat；测试时输出 L2 归一化特征
```

#### 4.2.2 关键组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **visible_module** | `models/agw.py` | RGB 模态独立的浅层特征提取（conv1~maxpool） |
| **thermal_module** | `models/agw.py` | IR 模态独立的浅层特征提取（conv1~maxpool） |
| **base_resnet** | `models/agw.py` | 共享的 ResNet50 layer1~layer4 |
| **Non_local** | `models/agw.py` | 非局部注意力模块，嵌入 layer2/layer3 |
| **GeM Pooling** | `models/agw.py` | 广义均值池化，$x_{pool} = (\frac{1}{HW}\sum x^p)^{1/p}$，$p=3.0$ |
| **BNNeck** | `models/agw.py` | BatchNorm1d(2048)，freeze bias，用于特征降维和正则化 |
| **L2Norm** | `models/agw.py` | 测试时对特征做 L2 归一化 |

#### 4.2.3 Non-local 注意力模块

Non-local 模块嵌入在 ResNet 的 layer2 和 layer3 中，用于捕获长距离空间依赖关系：

```python
# 前向流程
theta_x = theta(x)   # 1x1 conv -> reshape
phi_x = phi(x)       # 1x1 conv -> reshape
f = matmul(theta_x, phi_x)  # 计算注意力图
f_div_C = f / N      # 归一化
y = matmul(f_div_C, g_x)    # 加权聚合
z = W(y) + x         # 残差连接
```

- **layer1**: 0 个 Non-local
- **layer2**: 2 个 Non-local（512通道）
- **layer3**: 3 个 Non-local（1024通道）
- **layer4**: 0 个 Non-local

#### 4.2.4 模态控制（modal 参数）

模型通过 `modal` 参数控制前向传播路径：

| modal | 行为 |
|-------|------|
| 0 | 双模态模式：x1→visible_module, x2→thermal_module，拼接后进入共享层 |
| 1 | RGB 单模态：仅 visible_module |
| 2 | IR 单模态：仅 thermal_module |

### 4.3 ClusterMemory（聚类对比学习内存库）

#### 4.3.1 核心思想

ClusterMemory 将每个聚类中心视为一个"类别中心"，通过对比学习拉近样本与对应聚类中心的距离。它维护一个动量更新的特征内存库 `features`，在反向传播时自动更新。

#### 4.3.2 三种模式对比

| 模式 | 内存维度 | 更新策略 | 适用场景 |
|------|----------|----------|----------|
| **CM** | `[num_samples, dim]` | 单特征动量更新：$f_c \leftarrow m \cdot f_c + (1-m) \cdot x$ | 标准聚类对比 |
| **CMhybrid** | `[2*num_samples, dim]` | **双特征更新**：均值特征 + 最难样本特征 | 本项目使用，兼顾类内多样性和鲁棒性 |

#### 4.3.3 CMhybrid 的更新机制（核心算法）

CMhybrid 维护两个特征向量：
- `features[0:num_samples]`：**均值特征**（mean prototype）
- `features[num_samples:2*num_samples]`：**最难样本特征**（hard prototype）

**前向传播**：
```python
outputs = inputs.mm(features.t())  # 计算样本与所有聚类中心的相似度
outputs /= temp                    # temperature scaling
loss = cross_entropy(outputs, targets)
```

**反向传播更新**（`CM_Hybrid.backward`）：
```python
for each cluster index:
    # 1. 收集该簇的所有样本特征
    batch_centers[index] = [feature_1, feature_2, ...]
    
    # 2. 更新均值特征（mean prototype）
    mean = torch.stack(features).mean(0)
    features[index] = momentum * features[index] + (1-momentum) * mean
    features[index] /= norm(features[index])
    
    # 3. 更新最难样本特征（hard prototype）
    hard = argmin(similarity(feature_i, features[index]))
    features[index+nums] = momentum * features[index+nums] + (1-momentum) * features[hard]
    features[index+nums] /= norm(features[index+nums])
```

**为什么需要 hardest 特征？**
- 均值特征代表类的中心，但可能过于平滑
- 最难样本特征保留了类内最难匹配的样本信息，增强对困难样本的判别力

#### 4.3.4 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `temp` | 0.05 | Temperature scaling，控制分布锐度 |
| `momentum` | 0.1 | 特征更新动量，越小更新越快 |
| `mode` | CMhybrid | 使用混合原型模式 |

### 4.4 伪标签生成流程

#### 4.4.1 Stage 1 伪标签生成

```
训练集特征提取（get_single_model，单卡模式）
    ↓
Jaccard 距离计算（compute_jaccard_distance）
    ├── k1=30, k2=6
    ├── k-倒数近邻搜索（faiss）
    └── 计算扩展的 k-倒数集合
    ↓
DBSCAN 聚类（eps=0.6）
    ├── 生成伪标签 pseudo_labels
    └── 离群点标记为 -1
    ↓
过滤离群点
    ↓
生成聚类中心特征
    └── 对每个聚类内的特征取平均
    ↓
初始化 ClusterMemory
```

#### 4.4.2 k-倒数重排序（Jaccard 距离）

这是计算样本间相似度的关键步骤，基于 CVPR 2017 的 Re-ranking 论文：

```python
# 1. 找到每个样本的 k1 近邻（使用 faiss 加速）
initial_rank = faiss_search(features, k1)

# 2. 计算 k-倒数近邻集合
for each sample i:
    forward_k = initial_rank[i, :k1+1]      # i 的 k1 近邻
    backward_k = initial_rank[forward_k, :k1+1]  # 这些近邻的 k1 近邻
    k_reciprocal = forward_k[where(backward_k == i)]  # 互为近邻的样本

# 3. 扩展 k-倒数集合
for each candidate in k_reciprocal:
    candidate_k_reciprocal = k_reciprocal_neigh(candidate, k1/2)
    if overlap(candidate_k_reciprocal, k_reciprocal) > 2/3:
        k_reciprocal_expansion.append(candidate_k_reciprocal)

# 4. 计算 Jaccard 距离
V[i, j] = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|  # 局部编码相似度
Jaccard_dist = 1 - V  # 转换为距离
```

**参数**：
- `k1=30`：初始近邻数量
- `k2=6`：重排序时的近邻数量
- `search_option=3`：使用 CPU faiss（当前环境配置）

#### 4.4.3 DBSCAN 聚类

```python
from sklearn.cluster import DBSCAN
cluster = DBSCAN(eps=0.6, min_samples=4, metric='precomputed')
pseudo_labels = cluster.fit_predict(jaccard_distance)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `eps` | 0.6 | 邻域半径阈值 |
| `min_samples` | 4 | 核心点最小样本数 |
| `metric` | precomputed | 使用预计算的距离矩阵 |

**聚类结果示例**（Stage 2）：
- IR 聚类数：约 380-420
- RGB 聚类数：约 720-780
- RGB 离群点：约 200-350 / 22258
- IR 离群点：约 30-80 / 11909

#### 4.4.4 Stage 2 跨模态匹配（two_step_hungarian_matching）

**目标**：将 IR 聚类和 RGB 聚类进行一对一匹配，生成跨模态一致的伪标签。

**输入**：
- `proxy_features_rgb`：RGB 各聚类的中心特征 `[N_rgb, dim]`
- `proxy_features_ir`：IR 各聚类的中心特征 `[N_ir, dim]`

**算法流程**：

```python
# Step 1: 基于余弦相似度的初始匈牙利匹配
similarity = exp(proxy_rgb @ proxy_ir.T)   # 计算相似度矩阵
cost = 1 / similarity                       # 转换为代价矩阵
row_ind, col_ind = linear_sum_assignment(cost)  # 匈牙利算法

# 记录匹配对
for (r, c) in zip(row_ind, col_ind):
    if c < N_ir:        # 有效匹配
        R.append((r, c))
        r2i[r] = c       # RGB index → IR index
        i2r[c] = r       # IR index → RGB index
    else:
        unmatched_row.append(r)  # 未匹配的 RGB 聚类

# Step 2: 对未匹配的 RGB 聚类进行二次匹配
if len(unmatched_row) > N_ir:
    # 再次使用匈牙利算法对未匹配行进行匹配
    ...

# Step 3: 最终匹配（处理剩余未匹配行）
if len(unmatched_row) > 0:
    # 第三次匹配
    ...

# 生成统一的伪标签
pseudo_label = -1 * ones(N_rgb + N_ir)
cnt = 0
for (rgb_idx, ir_idx) in matched_pairs:
    pseudo_label[rgb_idx] = cnt
    pseudo_label[ir_idx + N_rgb] = cnt
    cnt += 1
```

**输出**：
- `i2r`：IR → RGB 的映射张量
- `r2i`：RGB → IR 的映射张量
- `pseudo_label`：统一的跨模态伪标签

### 4.5 Stage 1 训练器：ClusterContrastTrainer_DCL

**DCL = Deep Clustering Learning**

**目标**：分别对 RGB 和 IR 模态进行独立的聚类对比学习。

**每轮迭代流程**：
```python
# 1. 加载数据
inputs_ir, labels_ir  ←  IR DataLoader
inputs_rgb, inputs_rgb1, labels_rgb  ←  RGB DataLoader (含两种增强)

# 2. 拼接 RGB 的两份增强样本
inputs_rgb = cat(inputs_rgb, inputs_rgb1)
labels_rgb = cat(labels_rgb, labels_rgb)

# 3. 前向传播
feat_rgb, feat_ir, pool_rgb, pool_ir = encoder(inputs_rgb, inputs_ir, modal=0)

# 4. 计算损失
loss_ir  = memory_ir(feat_ir, labels_ir)    # IR 模态对比损失
loss_rgb = memory_rgb(feat_rgb, labels_rgb) # RGB 模态对比损失
loss = loss_ir + loss_rgb

# 5. 反向传播更新
loss.backward()
optimizer.step()
```

**注意**：
- RGB 数据经过两种不同的数据增强（`train_transformer_rgb` 和 `train_transformer_rgb1`），拼接后送入模型
- 这增加了 RGB 模态的样本多样性
- 损失只有模态内对比损失，**没有跨模态交互**

### 4.6 Stage 2 训练器：ClusterContrastTrainer_PCLMP

**PCLMP = Pseudo-label Cross-modal Learning with Matching Prototypes**

**目标**：在 Stage 1 基础上，通过跨模态伪标签匹配进行联合训练。

**核心差异**：增加了 **跨模态对比损失（cross_loss）**

**每轮迭代流程**：
```python
# 1-3. 同 Stage 1（数据加载 + 前向传播）

# 4. 模态内损失
loss_ir  = memory_ir(feat_ir, labels_ir)
loss_rgb = memory_rgb(feat_rgb, labels_rgb)

# 5. 跨模态对比损失（交替学习）
if epoch % 2 == 1:
    # 奇数 epoch：IR 特征 → RGB 内存库
    cross_loss = memory_rgb(feat_ir, ir2rgb_labels)
else:
    # 偶数 epoch：RGB 特征 → IR 内存库
    cross_loss = memory_ir(feat_rgb, rgb2ir_labels)

# 6. 总损失
loss = loss_ir + loss_rgb + 0.25 * cross_loss

# 7. 反向传播
loss.backward()
optimizer.step()
```

**交替学习策略（Alternate Cross-modal Learning）**：
- 不是同时计算双向跨模态损失，而是**轮流交替**
- 奇数 epoch：IR 特征 → RGB 聚类中心
- 偶数 epoch：RGB 特征 → IR 聚类中心
- 这样可以避免内存库更新冲突，训练更稳定

**`i2r` 和 `r2i` 映射**：
- `i2r[ir_label] = rgb_label`：IR 聚类 → RGB 聚类的映射
- `r2i[rgb_label] = ir_label`：RGB 聚类 → IR 聚类的映射
- 由 `two_step_hungarian_matching()` 生成

### 4.7 损失函数详解

#### 4.7.1 Stage 1 损失

$$\mathcal{L}_{Stage1} = \mathcal{L}_{IR} + \mathcal{L}_{RGB}$$

其中：
- $\mathcal{L}_{IR} = -\log \frac{\exp(f_{ir} \cdot c_{ir} / \tau)}{\sum_j \exp(f_{ir} \cdot c_j / \tau)}$
- $\mathcal{L}_{RGB} = -\log \frac{\exp(f_{rgb} \cdot c_{rgb} / \tau)}{\sum_j \exp(f_{rgb} \cdot c_j / \tau)}$

$f$ 为样本特征，$c$ 为聚类中心特征，$\tau=0.05$ 为温度系数。

#### 4.7.2 Stage 2 损失

$$\mathcal{L}_{Stage2} = \mathcal{L}_{IR} + \mathcal{L}_{RGB} + \lambda \cdot \mathcal{L}_{cross}$$

其中：
- $\lambda = 0.25$（跨模态损失权重）
- **交替跨模态损失**：
  - 奇数 epoch：$\mathcal{L}_{cross} = -\log \frac{\exp(f_{ir} \cdot c_{rgb} / \tau)}{\sum_j \exp(f_{ir} \cdot c_j / \tau)}$
  - 偶数 epoch：$\mathcal{L}_{cross} = -\log \frac{\exp(f_{rgb} \cdot c_{ir} / \tau)}{\sum_j \exp(f_{rgb} \cdot c_j / \tau)}$

### 4.8 数据增强详解

#### 4.8.1 基础增强

| 增强方式 | 说明 |
|----------|------|
| Resize | 288×144 |
| RandomCrop | Pad 10 + RandomCrop |
| RandomHorizontalFlip | p=0.5 |
| Color Jitter | 亮度、对比度、饱和度、色调 |

#### 4.8.2 通道级增强（ChannelAug）

`ChannelAug.py` 实现了 5 种通道级数据增强方法，专门针对跨模态（RGB→IR）的域差异：

| 增强方法 | 类名 | 作用 |
|----------|------|------|
| **通道交换** | `ChannelExchange` | 随机将 RGB 的三个通道复制为同一通道（R-only / G-only / B-only），模拟灰度/单通道特征 |
| **通道自适应** | `ChannelAdap` | 以概率 p 随机将图像转为单通道灰度，降低对颜色信息的依赖 |
| **通道自适应灰度** | `ChannelAdapGray` | 更激进的灰度转换，强制学习模态无关特征 |
| **通道随机擦除** | `ChannelRandomErasing` | 在通道维度上进行随机擦除，增强鲁棒性 |
| **灰度化** | `Gray` | 标准灰度转换（0.2989R + 0.5870G + 0.1140B） |

**使用位置**：
- `train_transformer_rgb1` 中应用了 `ChannelAdapGray(p=0.5)`
- 这是 RGB 模态特有的第二路增强，与第一路增强拼接后送入模型

### 4.9 特征提取与测试机制

#### 4.9.1 特征提取（extract_features）

```python
def extract_features(model, data_loader, mode):
    model.eval()
    for imgs in data_loader:
        # 1. 原始图像特征
        feat = model(imgs, imgs, modal=mode)
        
        # 2. 水平翻转图像特征
        imgs_flip = fliplr(imgs)
        feat_flip = model(imgs_flip, imgs_flip, modal=mode)
        
        # 3. 取平均并归一化
        final_feat = (feat + feat_flip) / 2.0
        # 注意：模型内部在 eval 模式下已做 L2 归一化
```

**flip 测试**：水平翻转后提取特征取平均，减少姿态变化的影响。

#### 4.9.2 距离计算

```python
# 训练时聚类：使用 Jaccard 距离（rerank 距离矩阵）
# 测试时检索：使用余弦相似度（矩阵乘法）
distmat = np.matmul(query_feat, gallery_feat.T)
```

测试时使用矩阵乘法直接计算余弦相似度，因为特征已 L2 归一化。

#### 4.9.3 评估指标

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| **Rank-k (CMC)** | 对每张 query，按相似度排序 gallery，计算前 k 个结果中命中正确 ID 的比例 | 取 10 trials 平均 |
| **mAP** | 对每张 query 计算 AP（Average Precision），取平均 | 衡量整体排序质量 |
| **mINP** | mean Inverse Negative Penalty，penalizes false negatives at high ranks | 衡量最难匹配样本的检索能力 |

**CMC 计算细节**（`evaluation_metrics/ranking.py`）：
- 过滤同 camera 同 ID 的图像（避免 trivial match）
- 使用 `first_match_break=True`（market1501 标准）
- 每个 query 找到第一个正确匹配位置，计算累积命中率

### 4.10 采样器（RandomMultipleGallerySampler）

#### 4.10.1 作用

在训练时，每个 batch 需要包含：
- 固定数量的身份（p = batch_size / num_instances）
- 每个身份固定数量的样本（k = num_instances）

这就是经典的 **PK 采样**（P identities × K instances）。

#### 4.10.2 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_instances` | 16 | 每个身份的样本数 |
| `batch_size` | 128 | 总 batch size = 8 identities × 16 instances |

#### 4.10.3 变体

| 采样器 | 说明 |
|--------|------|
| `RandomMultipleGallerySampler` | 标准 PK 采样，考虑 camera 信息 |
| `RandomMultipleGallerySamplerNoCam` | 不考虑 camera 的 PK 采样（`no_cam=True` 时使用） |

### 4.11 优化器与学习率调度

| 配置 | 值 |
|------|-----|
| 优化器 | Adam |
| 初始学习率 | 0.00035 |
| 学习率调度 | StepLR，step_size=20，gamma=0.1 |
| 权重衰减 | 0.0005 |
| 训练轮数 | 50 epochs |
| 每轮迭代 | 200 iters |

**学习率衰减时机**：
- Epoch 0-20：lr = 3.5e-4
- Epoch 21-40：lr = 3.5e-5
- Epoch 41-50：lr = 3.5e-6

### 4.12 测试流程

| 搜索模式 | 说明 |
|----------|------|
| **All Search** | 使用所有 gallery 图像（cam1, cam2, cam4, cam5）|
| **Indoor Search** | 仅使用室内 gallery 图像（cam1, cam2）|

**测试协议**：
- 每个 trial 随机采样 gallery 子集
- 共 10 trials，取平均结果
- 评估指标：Rank-1, Rank-5, Rank-10, Rank-20, mAP, mINP

### 4.13 关键代码调用链

#### 4.13.1 Stage 1 完整调用链

```
main()
  └─ main_worker_stage1()
       ├─ create_model()           # 创建 AGW + DataParallel
       ├─ 循环 epoch 0-49:
       │    ├─ extract_features(get_single_model(model))  # 提取特征
       │    ├─ compute_jaccard_distance()                 # Jaccard 距离
       │    ├─ DBSCAN.fit_predict()                       # 聚类
       │    ├─ generate_cluster_features()                # 生成聚类中心
       │    ├─ ClusterMemory()                            # 初始化内存库
       │    ├─ ClusterContrastTrainer_DCL.train()         # 训练
       │    │    ├─ _forward() → encoder(x1, x2, modal=0)
       │    │    ├─ memory_ir(f_out_ir, labels_ir)        # IR 损失
       │    │    └─ memory_rgb(f_out_rgb, labels_rgb)     # RGB 损失
       │    └─ 测试（all + indoor，单 trial）
       └─ 保存最优模型
```

#### 4.13.2 Stage 2 完整调用链

```
main()
  └─ main_worker_stage2()
       ├─ 加载 Stage 1 最优模型
       ├─ 循环 epoch 0-49:
       │    ├─ extract_features() + DBSCAN()              # 分别聚类
       │    ├─ two_step_hungarian_matching()              # 跨模态匹配
       │    │    ├─ linear_sum_assignment(cost)           # 匈牙利算法
       │    │    └─ 生成 i2r, r2i, pseudo_label
       │    ├─ ClusterMemory()                            # 初始化内存库
       │    ├─ ClusterContrastTrainer_PCLMP.train()       # 训练
       │    │    ├─ _forward() → encoder()
       │    │    ├─ memory_ir() + memory_rgb()            # 模态内损失
       │    │    └─ cross_loss (交替学习)                  # 跨模态损失 × 0.25
       │    └─ 测试（all + indoor，10 trials 平均）
       └─ 保存最优模型
```

#### 4.13.3 测试调用链

```
test_sysu.py
  └─ main_worker()
       ├─ 加载 Stage 2 最优模型
       ├─ mode='all'
       │    ├─ extract_query_feat(model, query_loader)     # 提取 query 特征
       │    ├─ for trial in range(10):
       │    │     ├─ process_gallery_sysu(trial=trial)     # 随机采样 gallery
       │    │     ├─ extract_gall_feat(model, gall_loader) # 提取 gallery 特征
       │    │     └─ np.matmul(query_feat, gall_feat.T)    # 距离矩阵
       │    └─ 计算 CMC + mAP + mINP
       └─ mode='indoor'（同上，仅室内相机）
```

---

## 五、实验结果

> **重要说明**：`logs/sysu_s1/train_log.txt` 日志文件中，Stage 1 的原始训练记录（Epoch 0-49）末尾被追加混入了 Stage 2 的部分输出内容（从 Epoch 0 R1=37.3% 开始）。这是因为 `train_sysu.py` 连续执行两阶段时 stdout 重定向的缓冲延迟导致。以下数据已根据日志上下文进行了清洗和校正。

### 5.1 Stage 1 训练过程（`sysu_s1`）

Stage 1 完成 50 epochs，**最优结果出现在 Epoch 45**：

| Epoch | Rank-1 | mAP | 备注 |
|-------|--------|-----|------|
| 0 | 6.28% | 8.16% | 初始 |
| 5 | 20.56% | 19.69% | — |
| 10 | 27.45% | 26.39% | — |
| 15 | 27.37% | 26.34% | — |
| 20 | 35.97% | 34.29% | — |
| 25 | 39.86% | 37.76% | best_epoch=25 |
| 30 | 40.12% | 40.06% | best_epoch=30 |
| 35 | 40.62% | 40.68% | — |
| 40 | 40.71% | 40.68% | best_epoch=40 |
| **45** | **42.52%** | **41.02%** | **Stage 1 Best** |
| 48 | 41.78% | 40.94% | — |
| 49 | 41.76% | 40.73% | — |

**Stage 1 收敛趋势**：
- 前 10 epochs 提升较快（6% → 27%）
- 10-30 epochs 稳步提升（27% → 40%）
- 30-45 epochs 缓慢增长至最优 42.5%
- 45 epochs 后轻微过拟合，性能下降

### 5.2 Stage 2 训练过程（`sysu_s2`）

Stage 2 基于 Stage 1 最优模型，每 2 个 epoch 进行一次完整评估（实际共 25 个评估点），**最优结果出现在 Epoch 48**：

| Epoch | Rank-1 | mAP | mINP | 备注 |
|-------|--------|-----|------|------|
| 0 | 37.26% | 35.86% | 22.43% | 加载 Stage 1 模型 |
| 2 | 43.99% | 41.19% | 26.61% | — |
| 4 | 45.07% | 40.65% | 24.32% | — |
| 6 | 45.14% | 41.25% | 25.37% | — |
| 8 | 45.94% | 41.70% | 26.28% | — |
| 10 | 47.92% | 43.68% | 27.79% | — |
| 12 | 47.40% | 42.76% | 27.09% | — |
| 14 | 46.13% | 42.17% | 25.97% | — |
| 16 | 47.25% | 44.46% | 28.84% | — |
| 18 | 47.76% | 44.11% | 27.52% | — |
| 20 | 54.37% | 48.85% | 31.37% | 明显提升 |
| 22 | 56.02% | 50.46% | 33.34% | — |
| 24 | 57.13% | 51.52% | 34.36% | — |
| 26 | 56.42% | 51.82% | 34.87% | — |
| 28 | 56.66% | 51.52% | 34.41% | — |
| 30 | 57.61% | 52.22% | 35.22% | — |
| 32 | 57.21% | 52.04% | 35.22% | — |
| 34 | 56.83% | 52.32% | 35.52% | — |
| 36 | 57.32% | 52.51% | 35.73% | — |
| 38 | 56.93% | 52.41% | 35.57% | — |
| 40 | 57.66% | 52.11% | 35.09% | — |
| 42 | 57.71% | 52.30% | 35.42% | — |
| 44 | 57.68% | 52.55% | 35.85% | — |
| 46 | 57.58% | 52.43% | 35.60% | — |
| **48** | **58.08%** | **52.80%** | **36.07%** | **Stage 2 Best** |

**关键观察**：
- Stage 2 初始即具备较高性能（37% Rank-1），说明 Stage 1 预训练有效
- Epoch 10-20 之间提升相对平缓（47% → 54%）
- Epoch 20-30 快速提升（54% → 57%）
- Epoch 30-48 缓慢收敛至最优 58.08%

### 5.3 最终测试结果（`logs/sysu_s2/test_log.txt`）

使用 Stage 2 最优模型 `train_model_best.pth.tar`（Epoch 48）进行测试：

#### All Search（全场景搜索）

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 59.43% | 81.96% | 88.25% | 94.50% | 53.41% | 36.29% |
| 1 | 57.17% | 86.17% | 92.43% | 96.00% | 53.16% | 36.93% |
| 2 | 58.37% | 83.80% | 90.98% | 95.85% | 52.76% | 35.43% |
| 3 | 59.69% | 84.38% | 90.88% | 96.19% | 54.38% | 37.82% |
| 4 | 59.16% | 83.30% | 90.27% | 95.19% | 54.27% | 38.43% |
| 5 | 57.69% | 83.54% | 91.16% | 96.34% | 52.55% | 34.68% |
| 6 | 58.32% | 82.20% | 89.30% | 94.74% | 51.85% | 34.76% |
| 7 | 56.06% | 83.22% | 90.72% | 96.03% | 51.14% | 34.26% |
| 8 | 56.88% | 81.46% | 88.59% | 95.24% | 51.24% | 34.92% |
| 9 | 58.03% | 83.07% | 90.67% | 95.21% | 53.20% | 37.18% |
| **平均** | **58.08%** | **83.31%** | **90.32%** | **95.53%** | **52.80%** | **36.07%** |

#### Indoor Search（室内搜索）

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|-------|--------|--------|---------|---------|-----|------|
| 0 | 64.04% | 87.86% | 93.84% | 97.42% | 69.23% | 64.11% |
| 1 | 61.91% | 85.01% | 92.80% | 97.51% | 67.26% | 62.74% |
| 2 | 63.09% | 83.33% | 92.03% | 97.83% | 67.96% | 63.54% |
| 3 | 64.76% | 86.01% | 93.34% | 98.78% | 68.77% | 63.54% |
| 4 | 63.00% | 86.59% | 94.02% | 96.56% | 68.96% | 65.14% |
| 5 | 60.82% | 85.19% | 92.57% | 97.46% | 67.03% | 63.03% |
| 6 | 60.37% | 86.05% | 92.98% | 97.60% | 67.69% | 64.03% |
| 7 | 61.10% | 87.32% | 93.80% | 97.87% | 67.80% | 63.19% |
| 8 | 62.05% | 86.82% | 93.30% | 97.96% | 67.76% | 62.93% |
| 9 | 62.68% | 88.04% | 94.93% | 97.83% | 70.11% | 66.64% |
| **平均** | **62.38%** | **86.22%** | **93.36%** | **97.68%** | **68.26%** | **63.89%** |

### 5.4 训练结果与测试结果一致性校验

| 来源 | 模式 | Rank-1 | mAP | mINP |
|------|------|--------|-----|------|
| Stage 2 Epoch 48 (train_log) | All Search | 58.08% | 52.80% | 36.07% |
| test_log.txt | All Search | **58.08%** | **52.80%** | **36.07%** |
| Stage 2 Epoch 48 (train_log) | Indoor Search | 62.38% | 68.26% | 63.89% |
| test_log.txt | Indoor Search | **62.38%** | **68.26%** | **63.89%** |

**校验结论**：训练日志中的最优 epoch 结果与独立测试脚本的结果 **完全一致**，说明模型保存和加载正确。

### 5.5 结果汇总

| 搜索模式 | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|----------|--------|--------|---------|---------|-----|------|
| **All Search** | **58.08%** | **83.31%** | **90.32%** | **95.53%** | **52.80%** | **36.07%** |
| **Indoor Search** | **62.38%** | **86.22%** | **93.36%** | **97.68%** | **68.26%** | **63.89%** |

---

## 六、项目文件结构

```
PGM-SYSU/
├── train_sysu.py          # 主训练脚本（两阶段）
├── test_sysu.py           # 测试脚本
├── run_train_sysu.sh      # 训练启动脚本
├── run_test_sysu.sh       # 测试启动脚本
├── ChannelAug.py          # 通道级数据增强
├── requirements.txt       # 环境依赖清单
├── PGM-SYSU代码报告清单.md  # 本报告
├── clustercontrast/
│   ├── datasets/          # 数据集定义
│   ├── models/            # 模型定义（AGW, ResNet, CLIP, CM）
│   ├── evaluators.py      # 评估器
│   ├── trainers.py        # 训练器（DCL, PCLMP）
│   ├── evaluation_metrics/# CMC, mAP 计算
│   └── utils/             # 工具函数（rerank, faiss, 聚类匹配等）
├── examples/
│   └── pretrained/
│       └── resnet50-19c8e357.pth  # ImageNet 预训练权重
└── logs/
    ├── sysu_s1/           # Stage 1 训练日志与模型
    │   ├── train_log.txt
    │   ├── train_model_best.pth.tar
    │   └── checkpoint.pth.tar
    └── sysu_s2/           # Stage 2 训练日志与模型
        ├── train_log.txt
        ├── test_log.txt
        ├── train_model_best.pth.tar
        └── checkpoint.pth.tar
```

---

## 七、关键代码模块说明

| 模块 | 文件路径 | 说明 |
|------|----------|------|
| AGW 骨干 | `clustercontrast/models/agw.py` | ResNet50 + Non-local + GeM Pooling + BNNeck |
| Cluster Memory | `clustercontrast/models/cm.py` | 聚类对比学习内存库（CM/CMhard/CMhybrid） |
| 训练器 Stage 1 | `clustercontrast/trainers.py` | `ClusterContrastTrainer_DCL`（模态内对比学习） |
| 训练器 Stage 2 | `clustercontrast/trainers.py` | `ClusterContrastTrainer_PCLMP`（跨模态交替对比学习） |
| Jaccard 重排序 | `clustercontrast/utils/faiss_rerank.py` | k-倒数重排序距离计算 |
| 跨模态匹配 | `clustercontrast/utils/matching_and_clustering.py` | 最优传输 + 匈牙利匹配 |
| 通道增强 | `ChannelAug.py` | 跨模态通道级数据增强 |
| 评估器 | `clustercontrast/evaluators.py` | 特征提取 + flip 测试 + CMC/mAP/mINP 计算 |

---

## 八、训练时长与资源消耗

| 阶段 | 时长 | 模型大小 |
|------|------|----------|
| Stage 1 | 约 6 小时 | 291 MB (train_model_best.pth.tar) |
| Stage 2 | 约 6 小时 | 291 MB (train_model_best.pth.tar) |
| 测试 | 约 2 分钟 | — |

---

*报告结束*
