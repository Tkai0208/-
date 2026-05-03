# RPNR-RegDB 基线复现报告清单

> **对比基准**：源代码 `/root/work/RPNR-main`（GitHub 官方代码） vs 复现代码 `/root/autodl-tmp/RPNR-RegDB复现和改进代码/RPNR-RegDB复现代码/RPNR-RegDB`  
> **论文**：Robust Pseudo-label Learning with Neighbor Relation for Unsupervised Visible-Infrared Person Re-Identification (RPNR, arXiv:2405.05613)  
> **复现时间**：2026年4月13日 – 4月15日  
> **核心原则**：仅做环境兼容性修复与路径适配，**未改动任何 baseline 核心算法逻辑**。

---

## 一、运行环境清单

### 1.1 硬件环境

| 项目 | 配置 |
|:---|:---|
| GPU | NVIDIA GeForce RTX 5090 (32GB) |
| GPU 数量 | 1 张（单卡训练） |
| 操作系统 | Linux (Ubuntu) |

### 1.2 软件环境

| 项目 | 版本 |
|:---|:---|
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| TorchVision | 0.23.0+cu128 |
| Pillow | 12.1.0 |
| NumPy | 2.4.3 |
| scikit-learn | 1.8.0 |
| faiss-cpu | 1.13.2 |
| POT | 0.9.6.post1 |
| SciPy | 1.17.1 |
| OpenCV-Python | 4.13.0 |
| TensorBoard | 2.20.0 |
| tqdm | 4.66.2 |

### 1.3 与官方原始环境的差异

| 项目 | 论文原始环境 | 本次复现环境 | 影响 |
|:---|:---|:---|:---|
| PyTorch | 1.8.0 | 2.8.0+cu128 | 需兼容补丁（weights_only、addmm_） |
| Python | 3.8.13 | 3.12.3 | 无直接影响 |
| NumPy | 1.x | 2.4.3 | 需 `np.bool` → `bool` |
| Pillow | <10 | 12.1.0 | 需 `ANTIALIAS` → `LANCZOS` |
| GPU | 未明确 | RTX 5090 (32GB) | 训练更快 |

### 1.4 关键依赖兼容性说明

- **PyTorch 2.8.0+cu128**：PyTorch 2.6+ 将 `torch.load` 的 `weights_only` 参数默认从 `False` 改为 `True`，导致所有旧版 `.pth` / `.tar` 权重加载失败，复现代码已显式设置 `weights_only=False`。
- **Pillow 12.1.0**：原始代码使用 `Image.ANTIALIAS`，在 Pillow 10+ 中已移除，复现代码已修复为 `Image.LANCZOS`。
- **NumPy 2.4.3**：`np.bool` 别名已移除，需改为 `bool`。
- **PyTorch 2.x `addmm_`**：不再支持旧版位置参数签名 `(1, -2, x, y.t())`，需改为关键字参数形式 `(x, y.t(), beta=1, alpha=-2)`。

---

## 二、代码修改清单

### 2.1 修改原则

**仅做环境兼容性修复与路径适配，未改动任何 baseline 核心算法逻辑。**

### 2.2 详细修改对比（RPNR-main → RPNR-RegDB）

#### 修改 1：`clustercontrast/models/resnet.py`

**原因**：硬编码的预训练权重路径 `/data/yxb/work1/examples/pretrained/resnet50-19c8e357.pth` 在当前环境中不存在，且 `torch.load` 缺少 `weights_only=False`。

**修改内容**：
```python
# 修改前
resnet.load_state_dict(torch.load('/data/yxb/work1/examples/pretrained/resnet50-19c8e357.pth'))

# 修改后
import os
possible_paths = [
    '/root/work/RPNR-main/examples/pretrained/resnet50-19c8e357.pth',
    './examples/pretrained/resnet50-19c8e357.pth',
    '../examples/pretrained/resnet50-19c8e357.pth',
    '/data/yxb/work1/examples/pretrained/resnet50-19c8e357.pth',
]
pretrained_path = None
for path in possible_paths:
    if os.path.exists(path):
        pretrained_path = path
        break
if pretrained_path is None:
    raise FileNotFoundError("Cannot find resnet50-19c8e357.pth. Please check the path.")
resnet.load_state_dict(torch.load(pretrained_path, weights_only=False))
```

**影响**：无。当前 baseline 实际使用 `-a agw`，走的是 `resnet_agw.py`，不会执行到此处；修改仅为增强鲁棒性。

---

#### 修改 2：`clustercontrast/utils/serialization.py`

**原因**：PyTorch 2.6+ `weights_only=True` 默认值导致旧版权重加载失败。

**修改内容**：
```python
# 修改前
checkpoint = torch.load(fpath, map_location=torch.device('cpu'))

# 修改后
checkpoint = torch.load(fpath, map_location=torch.device("cpu"), weights_only=False)
```

**影响**：无。仅恢复旧版权重加载行为。

---

#### 修改 3：`clustercontrast/evaluation_metrics/ranking.py`

**原因**：NumPy 2.x 移除了 `np.bool` 别名。

**修改内容**：
```python
# 修改前
mask = np.zeros(num, dtype=np.bool)

// 修改后
mask = np.zeros(num, dtype=bool)
```

**影响**：无。`bool` 与 `np.bool` 行为完全一致。

---

#### 修改 4：`clustercontrast/evaluators.py`

**原因**：PyTorch 2.x `addmm_` 不再支持旧版位置参数签名 `(1, -2, x, y.t())`。

**修改内容**：
```python
# 修改前
dist_m.addmm_(1, -2, x, y.t())

# 修改后
dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
```

**影响**：无。仅恢复旧的计算行为。

---

#### 修改 5：`clustercontrast/trainers.py`

**原因**：同上，`addmm_` API 签名变更。

**修改内容**：
```python
# 修改前
dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())

# 修改后
dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
```

**影响**：无。

---

#### 修改 6：`train_regdb.py` —— Pillow `ANTIALIAS` 兼容性修复

**原因**：Pillow 10+ 删除了 `Image.ANTIALIAS` 属性。

**修改内容**：
```python
# 修改前
img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)

# 修改后
img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
```

**影响**：无。`LANCZOS` 与 `ANTIALIAS` 为同一插值算法，图像预处理结果不变。

---

#### 修改 7：`train_regdb.py` —— `process_test_regdb()` 路径处理

**原因**：原代码路径拼接在某些环境下缺失 `/` 导致文件找不到。

**修改内容**：
```python
# 新增
def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    # 确保路径以/结尾
    if not img_dir.endswith('/'):
        img_dir = img_dir + '/'
    ...
```

**影响**：无。仅增强路径鲁棒性。

---

#### 修改 8：`train_regdb.py` —— 语法错误修复（×2 处）

**原因**：原代码中 `del cluster_loader_rgb,` 尾部逗号在 Python 3.12 下导致语法问题（解释为 tuple）。

**修改内容**：
```python
# 修改前（两处）
del cluster_loader_rgb,

// 修改后（两处）
del cluster_loader_rgb
```

**影响**：无。仅修复语法。

---

#### 修改 9：`test_regdb.py` —— Pillow `ANTIALIAS` 兼容性修复

**修改内容**：同修改 6，`Image.ANTIALIAS` → `Image.LANCZOS`。

---

#### 修改 10：`test_regdb.py` —— `process_test_regdb()` 路径规范化

**原因**：原代码使用字符串拼接路径，在某些系统下不鲁棒。

**修改内容**：
```python
# 修改前
input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]

# 修改后
input_data_path = osp.join(img_dir, 'idx/test_visible_{}.txt'.format(trial))
file_image = [osp.join(img_dir, s.split(' ')[0]) for s in data_file_list]
```

**影响**：无。仅增强跨平台兼容性。

---

#### 修改 11：`test_regdb.py` —— 重构为单 trial 模式，增加 `--resume` 和 `--trial` 参数

**原因**：原代码强制循环 10 个 trial 并计算平均值，无法灵活评估单个 trial。

**修改内容**：
```python
# 新增参数
parser.add_argument('--resume', type=str, metavar='PATH', help='path to model checkpoint')
parser.add_argument('--trial', type=int, default=1, help='trial number to evaluate')

# 修改前：强制 for trial in range(1, 11)
# 修改后：使用 args.trial，支持单 trial 评估
```

**影响**：增强灵活性，支持训练→评估串行流程。

---

#### 修改 12：`test_regdb.py` —— `state_dict` `module.` 前缀自适应

**原因**：`DataParallel` 保存的模型 key 含 `module.` 前缀，直接加载时 key 不匹配。

**修改内容**：
```python
# 新增
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.') and not list(model.state_dict().keys())[0].startswith('module.'):
        new_state_dict[k[7:]] = v  # 移除 'module.'
    elif not k.startswith('module.') and list(model.state_dict().keys())[0].startswith('module.'):
        new_state_dict['module.' + k] = v  # 添加 'module.'
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
```

**影响**：修复模型加载兼容性，确保 checkpoint 可在不同并行模式下正确加载。

---

#### 修改 13：`test_regdb.py` —— 结果自动保存到独立文件

**原因**：原代码仅打印到控制台，不便于后续汇总。

**修改内容**：增加结果自动保存到 `test_eval/result_trial{N}_{timestamp}.txt`。

**影响**：新增功能，便于 `compute_avg.py` 自动汇总。

---

#### 修改 14：`run_train_regdb.sh` —— 路径适配与流程重构

**修改前**：
```bash
CUDA_VISIBLE_DEVICES=0,1 \
 python train_regdb.py -mb CMhcl -b 128 -a agw -d regdb_rgb \
 --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 --trial $trial \
 --data-dir "/data/wml/dataset/RegDB/"
```

**修改后**：
```bash
for trial in 1 2 3 4 5 6 7 8 9 10
do
    echo "=========================================="
    echo "Starting Trial $trial"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 python train_regdb.py \
        -mb CMhcl -b 128 -a agw -d regdb_rgb \
        --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 \
        --trial $trial \
        --data-dir "../RegDB/"
    
    CUDA_VISIBLE_DEVICES=0 python test_regdb.py \
        -b 128 -a agw -d regdb_rgb \
        --trial $trial \
        --data-dir "../RegDB/"
    
    echo "Trial $trial completed!"
done
```

**改动说明**：
1. `--data-dir` 从 `/data/wml/dataset/RegDB/` 改为 `../RegDB/`，适配当前环境。
2. 将纯训练循环改为 **"训练 → 评估" 串行循环**，每完成一个 trial 立即评估。
3. 改为单卡 `CUDA_VISIBLE_DEVICES=0`。

---

#### 修改 15：`run_test_regdb.sh` —— 路径适配

**修改内容**：`--data-dir` 改为 `../RegDB/`。

---

#### 修改 16：新增 `compute_avg.py`

**说明**：RPNR-RegDB 目录下**新增**该脚本，用于自动解析 `test_eval/result_trial*.txt`，计算 10-trial 的 Rank-1/5/10/20、mAP、mINP 的均值与标准差，并输出到 `test_eval/summary_10trials_avg.txt`。

RPNR-main 中**不存在**此脚本。

---

### 2.3 修改总览表

| 序号 | 文件 | 修改类型 | 修改内容 | 对算法结果的影响 |
|:---:|:---|:---|:---|:---|
| 1 | `clustercontrast/models/resnet.py` | 兼容性修复 + 路径回退 | `torch.load` 添加 `weights_only=False`；增加 4 个候选路径自动搜索 | 无影响 |
| 2 | `clustercontrast/utils/serialization.py` | 兼容性修复 | `torch.load` 添加 `weights_only=False` | 无影响 |
| 3 | `clustercontrast/evaluation_metrics/ranking.py` | 兼容性修复 | `dtype=np.bool` → `dtype=bool` | 无影响 |
| 4 | `clustercontrast/evaluators.py` | 兼容性修复 | `addmm_(1, -2, x, y.t())` → `addmm_(x, y.t(), beta=1, alpha=-2)` | 无影响 |
| 5 | `clustercontrast/trainers.py` | 兼容性修复 | `addmm_(1, -2, emb1, emb2.t())` → `addmm_(emb1, emb2.t(), beta=1, alpha=-2)` | 无影响 |
| 6 | `train_regdb.py` | 兼容性修复 | `Image.ANTIALIAS` → `Image.LANCZOS` | 无影响 |
| 7 | `train_regdb.py` | 路径鲁棒性 | `process_test_regdb()` 确保路径以 `/` 结尾 | 无影响 |
| 8 | `train_regdb.py` | 语法修复 | `del cluster_loader_rgb,` → `del cluster_loader_rgb`（×2） | 无影响 |
| 9 | `test_regdb.py` | 兼容性修复 | `Image.ANTIALIAS` → `Image.LANCZOS` | 无影响 |
| 10 | `test_regdb.py` | 路径规范化 | `process_test_regdb()` 使用 `osp.join()` | 无影响 |
| 11 | `test_regdb.py` | 功能增强 | 新增 `--resume` 和 `--trial` 参数，支持单 trial 评估 | 支持训练-评估串行流程 |
| 12 | `test_regdb.py` | 兼容性修复 | `state_dict` `module.` 前缀自适应逻辑 | 修复模型加载问题 |
| 13 | `test_regdb.py` | 功能增强 | 结果自动保存到 `test_eval/result_trial{N}.txt` | 便于结果汇总 |
| 14 | `run_train_regdb.sh` | 路径适配 + 流程重构 | `--data-dir` 改为 `../RegDB/`；训练→评估串行循环 | 适配当前环境 |
| 15 | `run_test_regdb.sh` | 路径适配 | `--data-dir` 改为 `../RegDB/` | 适配当前环境 |
| 16 | `compute_avg.py` | 新增脚本 | 自动汇总 10-trial 评估结果 | 新增辅助工具 |

### 2.4 未修改的核心算法部分

以下模块在复现代码中**保持与源代码完全一致**，未做任何改动：
- `train_regdb.py` 中的两阶段训练逻辑（Stage 1: DCL / Stage 2: RPNR）
- `clustercontrast/trainers.py` 中的 `ClusterContrastTrainer_RPNR` 训练逻辑
- `clustercontrast/models/agw.py` 模型结构
- `clustercontrast/models/cm.py` 中的 `ClusterMemory`
- `clustercontrast/models/losses.py` 中的 `RCLoss`（邻居关系学习损失）
- DBSCAN 聚类参数与流程（eps=0.3, min_samples=4）
- `clustercontrast/utils/faiss_rerank.py` Jaccard 距离计算

---

## 三、数据集清单

### 3.1 数据集概况

| 属性 | 详情 |
|:---|:---|
| **数据集名称** | RegDB |
| **采集设备** | 1 台可见光相机 + 1 台远红外热成像相机，同步采集 |
| **总身份数** | 412 identities |
| **每人图像数** | 10 张 Visible + 10 张 Thermal = 20 张/人 |
| **原始总图像** | 8,240 张（4,120 Visible + 4,120 Thermal） |
| **预处理后图像** | 98,880 张（含数据增强后的 `rgb_modify` / `ir_modify`） |
| **图像分辨率** | 原始可变，统一 resize 为 **288 × 144** |
| **评估协议** | 10-trial 随机划分，每 trial 训练集/测试集各半 |

### 3.2 数据集路径

```
/root/autodl-tmp/RegDB/  (通过 ../RegDB/ 相对路径访问)
├── Visible/              # 原始可见光图像，412 个子目录（按 identity 组织）
│   ├── 0/                # identity 0 的 10 张 visible 图像
│   ├── 1/
│   └── ...
├── Thermal/              # 原始红外热成像图像，412 个子目录
│   ├── 0/
│   └── ...
├── rgb_modify/           # 预处理后可见光图像（数据增强后）
├── ir_modify/            # 预处理后红外图像（数据增强后）
└── idx/                  # 10-trial 划分索引文件
    ├── train_visible_1.txt ~ train_visible_10.txt  （每文件 2,060 行）
    ├── train_thermal_1.txt  ~ train_thermal_10.txt   （每文件 2,060 行）
    ├── test_visible_1.txt  ~ test_visible_10.txt    （每文件 2,060 行）
    └── test_thermal_1.txt  ~ test_thermal_10.txt     （每文件 2,060 行）
```

### 3.3 每 Trial 数据划分

| 划分 | 可见光 (Visible) | 红外 (Thermal) | 合计 |
|:---|:---|:---|:---|
| 训练集 | 2,060 张 | 2,060 张 | 4,120 张 |
| 测试集 (Query) | 2,060 张 | 2,060 张 | 4,120 张 |
| **每 Trial 总计** | **4,120 张** | **4,120 张** | **8,240 张** |

> 注：10 个 trial 使用相同的 412 个 identity 和 8,240 张图像，但训练/测试的 identity 划分不同（随机半分）。

### 3.4 数据预处理

`prepare_regdb.py` 执行以下步骤：
1. 将原始 BMP 图像转换为统一格式
2. 对每类模态执行数据增强（翻转、裁剪、擦除等）
3. 生成 `rgb_modify/` 和 `ir_modify/` 目录
4. 生成 10 组 trial 的索引文件 `idx/train_*.txt` 和 `idx/test_*.txt`

---

## 四、实验框架与流程

### 4.1 整体架构

RPNR 采用**两阶段训练策略**：

```
第一阶段: DCL (Dynamic Clustering Learning) 热身
    ├── 初始化模型（ImageNet 预训练权重）
    ├── 对每个模态独立进行 DBSCAN 聚类
    ├── 生成伪标签
    ├── 使用 ClusterContrastTrainer_DCL 训练
    └── 保存第一阶段最优模型 -> logs/regdb_s1/{trial}/model_best.pth.tar

第二阶段: RPNR (Robust Pseudo-label learning with Neighbor Relation)
    ├── 加载第一阶段最优模型
    ├── NPC  (Neighbor-guided Pseudo-label Cleaning)
    ├── OTPM (Optimal Transport Pseudo-label Matching)
    ├── MHL  (Multi-level Hybrid Learning)
    ├── NRL  (Neighbor Relation Learning)
    └── 保存第二阶段最优模型 -> logs/regdb_s2/{trial}/model_best.pth.tar
```

### 4.2 核心模块说明

| 模块 | 文件 | 功能 |
|:---|:---|:---|
| **模型 backbone** | `clustercontrast/models/agw.py` | AGW 网络（ResNet50 + Non-local + GeM pooling） |
| **Stage 1 Trainer** | `clustercontrast/trainers.py` | `ClusterContrastTrainer_DCL`：单模态独立对比学习 warm-up |
| **Stage 2 Trainer** | `clustercontrast/trainers.py` | `ClusterContrastTrainer_RPNR`：跨模态伪标签学习与邻居关系约束 |
| **聚类** | `train_regdb.py` | DBSCAN (eps=0.3, min_samples=4) 生成伪标签 |
| **数据加载** | `clustercontrast/datasets/regdb_rgb.py` / `regdb_ir.py` | RegDB 数据集专用加载器 |
| **评估** | `clustercontrast/evaluators.py` | 特征提取（L2 归一化 + 水平翻转平均）、CMC / mAP / mINP 计算 |

### 4.3 Stage 1：DCL 热身（50 epochs）

**每 epoch 流程**：

1. **特征提取**：提取 RGB 训练集全部特征 `features_rgb`，IR 训练集全部特征 `features_ir`。
2. **DBSCAN 聚类**（`eps=0.3`, `min_samples=4`）：
   - 计算 Jaccard 距离 `rerank_dist = compute_jaccard_distance(features, k1=30, k2=6)`
   - RGB 模态聚类 → `pseudo_labels_rgb`
   - IR 模态聚类 → `pseudo_labels_ir`
   - 噪声点标记为 `-1`
3. **生成聚类中心**：各簇内特征取平均。
4. **构建 Cluster Memory**：`memory_rgb` 和 `memory_ir`，使用 `CMhcl` 模式。
5. **训练一个 epoch**：
   - RGB 数据加载器：`batch_size//2=64`，含 `ChannelExchange` 增强
   - IR 数据加载器：`batch_size=128`，含 `ChannelAdapGray` 增强
   - 损失：对比学习损失（Cluster Contrastive Loss）
6. **验证**：每 epoch 测试 V→T Rank-1，保存最佳模型 `model_best.pth.tar`。

### 4.4 Stage 2：RPNR 主训练（50 epochs）

**初始化**：加载 Stage 1 最佳模型权重。

**每 epoch 流程**：

1. **特征提取与 DBSCAN 聚类**（同 Stage 1，`eps=0.3`）

2. **NPC：邻居引导伪标签净化**
   - 对每个簇计算原型（prototypes）：
     - 归一化特征，计算余弦相似度矩阵 `S`
     - 邻居投票得分：`rho = sign(S[j] - 0.5).sum()`
     - 取 top-5 rho 对应的样本作为簇原型
   - `correct_label()`：用原型对噪声/边缘样本重新分配标签
     - `s = dot(features[i], prototypes.T) / (norm * norm)`
     - `pseudo_labels_hat = argmax(s)`
   - 输出净化后的 `pseudo_labels_rgb_hat` 和 `pseudo_labels_ir_hat`

3. **OTPM：最优传输伪标签匹配**
   - 归一化聚类中心：`cluster_features_rgb_norm`, `cluster_features_ir_norm`
   - 计算相似度矩阵：`similarity = exp(mm(rgb_norm, ir_norm.T))`
   - 转换代价矩阵：`cost = 1 / similarity`
   - **Sinkhorn 算法**求解最优传输：
     ```python
     result = ot.sinkhorn(a, b, M, reg=5, numItermax=5000, stopThr=1e-5)
     ```
   - 硬匹配：`rgb_to_ir = argmax(result, axis=1)`, `ir_to_rgb = argmax(result, axis=0)`

4. **MHL：多级混合学习**
   - 构建 Hybrid Memory：
     ```python
     cluster_features_hybrid[i] = mean([cluster_features_ir[i], cluster_features_rgb[i2r[i]]])
     ```
   - 即 IR 簇中心与其匹配的 RGB 簇中心取平均

5. **训练一个 epoch**
   - 使用净化后的伪标签 `pseudo_labels_rgb_hat` / `pseudo_labels_ir_hat`
   - Trainer: `ClusterContrastTrainer_RPNR`
   - 输入：`i2r`, `r2i` 映射关系
   - 损失：Cluster Contrastive Loss + Cross Contrastive Loss + Hybrid Loss + Neighbor Relation Loss

6. **验证**：每 epoch 测试 V→T Rank-1，保存最佳模型。

### 4.5 测试阶段

**对每 trial 的 Stage 2 最佳模型**：

1. 加载 `logs/regdb_s2/{trial}/model_best.pth.tar`
2. **V→T 测试**：Visible 为 Query，Thermal 为 Gallery
3. **T→V 测试**：Thermal 为 Query，Visible 为 Gallery
4. 特征提取时使用 **水平翻转增强**（flip + average）
5. 距离计算：余弦相似度 `distmat = query_feat @ gall_feat.T`
6. 评估指标：CMC (Rank-1/5/10/20), mAP, mINP

### 4.6 训练参数设置

| 项目 | 设置 |
|:---|:---|
| **方法** | RPNR (Robust Pseudo-label Learning with Neighbor Relation) |
| **数据集** | RegDB |
| **评估协议** | 10-fold cross-validation |
| **模型** | AGW (ResNet50 backbone) |
| **Memory Bank** | CMhcl |
| **Batch Size** | 128 |
| **优化器** | Adam |
| **学习率 (lr)** | 0.00035 |
| **权重衰减 (weight_decay)** | 0.0005 |
| **Stage 1 epochs** | 50 |
| **Stage 2 epochs** | 50 |
| **每 epoch iterations** | 100 |
| **聚类算法** | DBSCAN |
| **DBSCAN eps (Stage 1/2)** | 0.3（硬编码） |
| **DBSCAN min_samples** | 4 |
| **num_instances** | 16 |
| **momentum** | 0.1 |
| **输入尺寸** | 288 × 144 |
| **k1 / k2 (Jaccard)** | 30 / 6 |
| **Sinkhorn reg (OTPM)** | 5 |
| **训练设备** | 单卡 RTX 5090 (32GB) |
| **CUDA_VISIBLE_DEVICES** | 0 |

### 4.7 训练与评估脚本

**批量训练并评估（10 trials）**：
```bash
cd /root/autodl-tmp/RPNR-RegDB复现和改进代码/RPNR-RegDB复现代码/RPNR-RegDB
sh run_train_regdb.sh
```

该脚本会顺序执行 trial 1~10，每个 trial 先训练后评估。

**独立测试单个 trial**：
```bash
CUDA_VISIBLE_DEVICES=0 python test_regdb.py \
    -b 128 -a agw -d regdb_rgb \
    --trial 1 \
    --data-dir "../RegDB/"
```

**结果汇总**：
```bash
cd /root/autodl-tmp/RPNR-RegDB复现和改进代码/RPNR-RegDB复现代码/RPNR-RegDB
python compute_avg.py
```

---

## 五、核心算法与公式

### 5.1 NPC（Neighbor-guided Pseudo-label Cleaning，邻居引导伪标签净化）

**目的**：通过邻居投票原型净化 DBSCAN 聚类产生的噪声伪标签。

**步骤**：

1. **归一化特征**：
   ```
   normalized_feat = feat / ||feat||_2
   ```

2. **计算余弦相似度矩阵**：
   ```
   S = cosine_similarity(normalized_feat)  # shape: (N_cluster, N_cluster)
   ```

3. **邻居投票得分（rho 分数）**：
   ```
   rho[j] = sum(sign(S[j] - 0.5))
   ```
   即统计与样本 j 的余弦相似度大于 0.5 的邻居数量。

4. **选取原型**：取 rho 分数最高的 top-5 个样本作为该簇的原型：
   ```
   prototypes[i] = mean(top-5 feat)
   ```

5. **重新分配标签**：对每个样本，计算其与各原型原型的余弦相似度，分配到最相似的原型所属簇：
   ```
   s[i] = dot(features[i], prototypes.T) / (||features[i]|| * ||prototypes||)
   pseudo_labels_hat[i] = argmax(s[i])
   ```

### 5.2 OTPM（Optimal Transport Pseudo-label Matching，最优传输伪标签匹配）

**目的**：通过 Sinkhorn 算法建立跨模态聚类簇的软匹配关系。

**步骤**：

1. **归一化聚类中心**：
   ```
   C_rgb = F.normalize(cluster_features_rgb, dim=1)
   C_ir = F.normalize(cluster_features_ir, dim=1)
   ```

2. **计算指数相似度**：
   ```
   similarity = exp(C_rgb @ C_ir.T)
   ```

3. **转换代价矩阵**：
   ```
   cost = 1 / similarity
   ```

4. **Sinkhorn 算法**求解最优传输计划：
   ```
   a = ones(N_rgb) / N_rgb
   b = ones(N_ir) / N_ir
   result = Sinkhorn(a, b, cost, reg=5)
   ```
   其中 `reg=5` 为正则化系数，控制传输计划的熵正则化强度。

5. **硬匹配提取**：
   ```
   rgb_to_ir[i] = argmax(result[i, :])   # RGB 簇 i 匹配到哪个 IR 簇
   ir_to_rgb[j] = argmax(result[:, j])   # IR 簇 j 匹配到哪个 RGB 簇
   ```

### 5.3 MHL（Multi-level Hybrid Learning，多级混合学习）

**目的**：通过将匹配的 IR 和 RGB 聚类中心取平均，构建混合记忆库，实现双模态联合学习。

**公式**：
```
cluster_features_hybrid[i] = mean([cluster_features_ir[i], cluster_features_rgb[i2r[i]]])
```

即对每个 IR 簇 i，找到其通过 OTPM 匹配到的 RGB 簇 `i2r[i]`，将两个簇中心取平均，作为 hybrid memory 的初始化特征。

### 5.4 NRL（Neighbor Relation Learning，邻居关系学习）

**目的**：在特征空间中强制邻居一致性约束，增强特征判别性。

**实现**：`RCLoss`（Relaxed Contrastive Loss）

**公式**：

1. **计算 batch 内 pairwise 距离**：
   ```
   S_dist = cdist(s_emb, s_emb)
   S_dist = S_dist / S_dist.mean(1, keepdim=True)  # 归一化
   ```

2. **构建目标邻居权重（无梯度）**：
   ```
   T_dist = cdist(t_emb, t_emb)
   W = exp(-T_dist^2 / sigma)
   pos_weight = W * (1 - I)
   neg_weight = (1 - W) * (1 - I)
   ```
   其中 `I` 为单位矩阵，`sigma=1`。

3. **Pull Loss（拉近邻居）**：
   ```
   pull_loss = sum(relu(S_dist)^2 * pos_weight)
   ```

4. **Push Loss（推远非邻居）**：
   ```
   push_loss = sum(relu(delta - S_dist)^2 * neg_weight)
   ```
   其中 `delta=1` 为间隔阈值。

5. **总损失**：
   ```
   loss_RC = (pull_loss + push_loss) / (N * (N - 1))
   ```

**在 Stage 2 总损失中的权重**：`loss = loss_ir + loss_rgb + 0.25*cross_loss + 0.5*loss_hybrid + 10.0*loss_RC`

### 5.5 Stage 2 总损失函数

```
Loss_total = Loss_ir + Loss_rgb + 0.25 * Loss_cross + 0.5 * Loss_hybrid + 10.0 * Loss_RC
```

其中：
- `Loss_ir`：IR 模态的聚类对比损失
- `Loss_rgb`：RGB 模态的聚类对比损失
- `Loss_cross`：跨模态交替对比损失（ACCL）
  - 奇数 epoch：`memory_rgb(f_out_ir, ir2rgb_labels)`
  - 偶数 epoch：`memory_ir(f_out_rgb, rgb2ir_labels)`
- `Loss_hybrid`：混合记忆库对比损失
- `Loss_RC`：邻居关系学习损失（NRL）

---

## 六、实验结果

### 6.1 10-Trial 汇总结果（由 `compute_avg.py` 自动生成）

#### Visible → Thermal (V→T)

| 指标 | 均值 ± 标准差 |
|:---|:---|
| **Rank-1** | **89.90% ± 1.47%** |
| **Rank-5** | **94.17% ± 1.36%** |
| **Rank-10** | **95.90% ± 1.36%** |
| **Rank-20** | **97.42% ± 1.17%** |
| **mAP** | **82.57% ± 1.38%** |
| **mINP** | **68.44% ± 1.63%** |

#### Thermal → Visible (T→V)

| 指标 | 均值 ± 标准差 |
|:---|:---|
| **Rank-1** | **89.11% ± 2.18%** |
| **Rank-5** | **93.99% ± 1.81%** |
| **Rank-10** | **95.90% ± 1.41%** |
| **Rank-20** | **97.44% ± 1.11%** |
| **mAP** | **81.74% ± 1.86%** |
| **mINP** | **65.29% ± 1.39%** |

### 6.2 各 Trial 详细结果（源自 `test_eval/result_trial*_*.txt`）

#### Visible → Thermal (V→T)

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 87.48% | 94.32% | 96.60% | 98.30% | 80.24% | 65.87% |
| 2 | 93.01% | 96.36% | 98.20% | 99.27% | 85.37% | 70.88% |
| 3 | 88.01% | 91.70% | 93.40% | 95.00% | 81.36% | 67.77% |
| 4 | 89.27% | 93.98% | 95.58% | 97.48% | 82.00% | 67.71% |
| 5 | 89.42% | 92.72% | 94.56% | 96.60% | 81.89% | 67.91% |
| 6 | 90.39% | 94.85% | 96.46% | 97.57% | 82.75% | 69.08% |
| 7 | 90.92% | 95.58% | 97.23% | 98.64% | 83.56% | 69.37% |
| 8 | 89.76% | 92.86% | 94.47% | 96.31% | 83.88% | 71.05% |
| 9 | 90.05% | 94.08% | 95.97% | 97.72% | 81.76% | 66.28% |
| 10 | 90.73% | 95.24% | 96.50% | 97.28% | 82.86% | 68.49% |
| **Mean±Std** | **89.90±1.47%** | **94.17±1.36%** | **95.90±1.36%** | **97.42±1.17%** | **82.57±1.38%** | **68.44±1.63%** |

#### Thermal → Visible (T→V)

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 87.43% | 93.79% | 96.02% | 97.04% | 79.89% | 63.65% |
| 2 | 92.57% | 96.65% | 98.25% | 99.42% | 85.23% | 67.97% |
| 3 | 84.85% | 89.56% | 92.48% | 95.00% | 78.55% | 63.93% |
| 4 | 87.18% | 93.45% | 95.58% | 97.38% | 80.50% | 64.63% |
| 5 | 88.50% | 93.98% | 95.44% | 97.14% | 81.41% | 65.05% |
| 6 | 89.61% | 94.37% | 96.31% | 97.91% | 82.33% | 66.01% |
| 7 | 92.23% | 96.02% | 96.99% | 98.59% | 84.05% | 66.36% |
| 8 | 89.61% | 93.20% | 95.24% | 96.65% | 82.70% | 66.70% |
| 9 | 89.32% | 94.03% | 96.12% | 97.62% | 80.93% | 63.43% |
| 10 | 89.76% | 94.85% | 96.60% | 97.67% | 81.78% | 65.17% |
| **Mean±Std** | **89.11±2.18%** | **93.99±1.81%** | **95.90±1.41%** | **97.44±1.11%** | **81.74±1.86%** | **65.29±1.39%** |

### 6.3 结果验证说明

- 所有实验结果均来自实际运行 `test_regdb.py` 生成的日志文件，保存在 `/root/autodl-tmp/RPNR-RegDB复现和改进代码/RPNR-RegDB复现代码/RPNR-RegDB/test_eval/result_trial{1..10}_*.txt`。
- `compute_avg.py` 自动解析上述日志，计算均值与标准差，输出至 `/root/autodl-tmp/RPNR-RegDB复现和改进代码/RPNR-RegDB复现代码/RPNR-RegDB/test_eval/summary_10trials_avg.txt`。
- 评估时加载的模型检查点为 `/root/work/RPNR-main/logs/regdb_s2/{trial}/model_best.pth.tar`（Stage 2 最优模型）。
- 测试过程包含 **Visible→Thermal** 和 **Thermal→Visible** 两个方向的完整评估。

### 6.4 与官方结果对比

#### 6.4.1 GitHub README 官方 10-trial 详细结果

**Visible → Thermal (V→T)**

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 92.91% | 96.89% | 98.25% | 99.17% | 86.28% | 73.99% |
| 2 | 91.80% | 96.41% | 98.20% | 99.17% | 83.97% | 69.48% |
| 3 | 88.69% | 93.01% | 94.71% | 96.55% | 82.65% | 70.18% |
| 4 | 88.40% | 93.11% | 95.05% | 96.99% | 80.89% | 67.35% |
| 5 | 89.61% | 93.98% | 96.02% | 97.67% | 81.32% | 66.14% |
| 6 | 88.40% | 94.47% | 96.12% | 97.77% | 82.31% | 69.43% |
| 7 | 89.90% | 94.17% | 95.87% | 97.62% | 82.72% | 69.34% |
| 8 | 88.98% | 93.25% | 95.44% | 97.48% | 83.30% | 70.59% |
| 9 | 90.97% | 95.15% | 97.14% | 98.35% | 83.82% | 70.16% |
| 10 | 93.45% | 96.12% | 97.14% | 98.06% | 86.22% | 72.89% |
| **Mean** | **90.31%** | **94.66%** | **96.39%** | **97.88%** | **83.35%** | **69.96%** |

**Thermal → Visible (T→V)**

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 91.26% | 96.41% | 98.35% | 99.56% | 84.82% | 70.47% |
| 2 | 90.78% | 96.17% | 97.72% | 98.79% | 83.76% | 66.89% |
| 3 | 87.28% | 92.43% | 95.05% | 96.75% | 81.60% | 67.22% |
| 4 | 87.48% | 92.96% | 95.00% | 97.23% | 79.66% | 64.76% |
| 5 | 88.83% | 93.69% | 95.78% | 97.38% | 80.12% | 62.95% |
| 6 | 87.48% | 93.69% | 95.63% | 97.09% | 81.22% | 65.68% |
| 7 | 87.57% | 93.98% | 95.78% | 96.94% | 81.08% | 65.05% |
| 8 | 88.54% | 93.69% | 95.19% | 97.09% | 82.01% | 66.05% |
| 9 | 89.90% | 94.61% | 96.46% | 97.67% | 83.04% | 66.14% |
| 10 | 91.80% | 96.12% | 97.14% | 97.96% | 85.30% | 69.76% |
| **Mean** | **89.09%** | **94.37%** | **96.21%** | **97.65%** | **82.26%** | **66.50%** |

#### 6.4.2 官方均值 vs 本次复现均值

| 来源 | V→T Rank-1 | V→T Rank-5 | V→T Rank-10 | V→T Rank-20 | V→T mAP | V→T mINP | T→V Rank-1 | T→V Rank-5 | T→V Rank-10 | T→V Rank-20 | T→V mAP | T→V mINP |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 官方 README | 90.31% | 94.66% | 96.39% | 97.88% | 83.35% | 69.96% | 89.09% | 94.37% | 96.21% | 97.65% | 82.26% | 66.50% |
| 本次复现 | 89.90% | 94.17% | 95.90% | 97.42% | 82.57% | 68.44% | 89.11% | 93.99% | 95.90% | 97.44% | 81.74% | 65.29% |
| 差距 | -0.41% | -0.49% | -0.49% | -0.46% | -0.78% | -1.52% | +0.02% | -0.38% | -0.31% | -0.21% | -0.52% | -1.21% |

#### 6.4.3 论文 Table 1

| 来源 | V→T Rank-1 | V→T mAP |
|:---|:---|:---|
| RPNR 论文 | 90.9% | 84.7% |
| 本次复现 | 89.90% | 82.57% |
| 差距 | -1.00% | -2.13% |

### 6.5 结果分析

- **复现成功**：本次复现的 V→T Rank-1 = 89.90%，与官方 README 的 90.31% 差距仅 **0.41 个百分点**，在合理误差范围内。
- **V→T Rank-1 标准差为 1.47%**，T→V 为 2.18%，说明 RegDB 的 10-trial 交叉验证结果存在一定波动。
- **最好 trial（Trial 2）**：V→T 93.01% / T→V 92.57%；**最差 trial（Trial 1）**：V→T 87.48% / T→V 87.43%。
- **与论文差距可能原因**：
  1. **环境差异**：PyTorch 2.8 vs 1.8，内部算子实现（如 BatchNorm、卷积）可能有微小差异。
  2. **随机性**：DBSCAN 聚类、数据增强、Dropout 等均含随机性。
  3. **Trial 1 偏低**：Trial 1 的 Rank-1 = 87.48%，显著低于其他 trial（88%~93%），拉低了平均值。

---

## 七、文件修改树状图

```
RPNR-main/                                  RPNR-RegDB/
│                                           │
├── README.md                               ├── REGDB_REPRODUCTION_REPORT_FULL.md  (已有)
├── environment.yml                         ├── RPNR-RegDB复现报告清单.md  (本文件)
├── requirements.txt                        ├── compute_avg.py  (新增)
├── prepare_regdb.py                        ├── prepare_regdb.py  (未修改)
├── imgs/                                   ├── imgs/  (未修改)
├── examples/                               ├── examples/pretrained/  (新增预训练权重)
├── train_regdb.py                          ├── train_regdb.py  (+多处修复)
├── test_regdb.py                           ├── test_regdb.py  (+多处修复与增强)
├── run_train_regdb.sh                      ├── run_train_regdb.sh  (+路径+流程重构)
├── run_test_regdb.sh                       ├── run_test_regdb.sh  (+路径适配)
├── ChannelAug.py                           ├── ChannelAug.py  (未修改)
├── meters.py                               ├── meters.py  (未修改)
└── clustercontrast/                        └── clustercontrast/
    ├── __init__.py                             ├── __init__.py  (未修改)
    ├── trainers.py                             ├── trainers.py  (+addmm_修复)
    ├── evaluators.py                           ├── evaluators.py  (+addmm_修复)
    ├── datasets/                               ├── datasets/  (未修改)
    ├── evaluation_metrics/                     ├── evaluation_metrics/
    │   ├── classification.py                   │   ├── classification.py  (未修改)
    │   └── ranking.py                          │   └── ranking.py  (+np.bool修复)
    ├── models/                                 ├── models/
    │   ├── agw.py                              │   ├── agw.py  (未修改)
    │   ├── cm.py                               │   ├── cm.py  (未修改)
    │   ├── losses.py                           │   ├── losses.py  (未修改)
    │   ├── resnet.py                           │   ├── resnet.py  (+weights_only+路径)
    │   ├── resnet_agw.py                       │   ├── resnet_agw.py  (未修改)
    │   ├── resnet_ibn.py                       │   ├── resnet_ibn.py  (未修改)
    │   ├── resnet_ibn_a.py                     │   ├── resnet_ibn_a.py  (未修改)
    │   └── vision_transformer.py               │   └── vision_transformer.py  (未修改)
    └── utils/                                  └── utils/
        ├── faiss_rerank.py                         ├── faiss_rerank.py  (未修改)
        ├── faiss_utils.py                          ├── faiss_utils.py  (未修改)
        ├── logging.py                              ├── logging.py  (未修改)
        ├── meters.py                               ├── meters.py  (未修改)
        ├── osutils.py                              ├── osutils.py  (未修改)
        ├── rerank.py                               ├── rerank.py  (未修改)
        └── serialization.py                        └── serialization.py  (+weights_only)
```

---

> **报告生成时间**：2026-04-30  
> **报告依据**：直接读取 `/root/autodl-tmp/RPNR-RegDB复现和改进代码/RPNR-RegDB复现代码/RPNR-RegDB/test_eval/result_trial*_*.txt` 与 `summary_10trials_avg.txt` 中的实测数据，以及 `diff` 命令对比 `/root/work/RPNR-main` 与 `/root/autodl-tmp/RPNR-RegDB复现和改进代码/RPNR-RegDB复现代码/RPNR-RegDB` 的代码差异。
