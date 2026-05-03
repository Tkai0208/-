# PGM-RegDB 基线复现报告清单

> **对比基准**：源代码 `/root/work/PGM-new` vs 复现代码 `/root/work/PGM-RegDB`  
> **方法来源**：Wu 等, "Unsupervised Visible-Infrared Person Re-Identification via Progressive Graph Matching and Alternate Cross Contrastive Learning", CVPR 2023.  
> **实验时间**：2026年4月  
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
| SciPy | 1.17.1 |
| OpenCV-Python | 4.13.0 |
| TensorBoard | 2.20.0 |
| tqdm | 4.66.2 |

### 1.3 关键依赖兼容性说明

- **PyTorch 2.8.0+cu128**：PyTorch 2.6+ 将 `torch.load` 的 `weights_only` 参数默认从 `False` 改为 `True`，导致所有旧版 `.pth` / `.tar` 权重加载失败，复现代码已显式设置 `weights_only=False` 进行兼容。
- **Pillow 12.1.0**：原始代码使用 `Image.ANTIALIAS`，在 Pillow 10+ 中已移除，复现代码已修复为 `Image.Resampling.LANCZOS`。

---

## 二、代码修改清单

### 2.1 修改原则

**仅做环境兼容性修复与路径适配，未改动任何 baseline 算法逻辑。**

具体包括：
- PyTorch 2.6+ `weights_only` 兼容性修复
- Pillow 10+ `ANTIALIAS` 移除修复
- 错误导入修复
- 预训练权重路径回退
- 数据集路径适配
- 训练-评估流程重构
- 测试脚本 bug 修复与功能增强

### 2.2 详细修改对比（PGM-new → PGM-RegDB）

#### 修改 1：`clustercontrast/models/resnet.py`

**原因**：硬编码的预训练权重路径 `/dat01/.../resnet50-19c8e357.pth` 在当前环境中不存在，且 `torch.load` 缺少 `weights_only=False`。

**修改内容**：
```python
# 修改前
resnet = ResNet.__factory[depth](pretrained=False)
resnet.load_state_dict(torch.load('/dat01/yangbin/cluster-contrast-reid-main/examples/pretrained/resnet50-19c8e357.pth'))

# 修改后
try:
    resnet = ResNet.__factory[depth](pretrained=False)
    resnet.load_state_dict(torch.load('/dat01/yangbin/cluster-contrast-reid-main/examples/pretrained/resnet50-19c8e357.pth', weights_only=False))
except Exception:
    resnet = ResNet.__factory[depth](pretrained=True)
```

**影响**：无。当前 baseline 实际使用 `-a agw`，走的是 `resnet_agw.py`，不会执行到此处；修改仅为增强鲁棒性。

---

#### 修改 2：`clustercontrast/models/resnet_agw.py`

**原因**：PyTorch 2.6+ `weights_only=True` 默认值导致旧版权重加载失败。

**修改内容**：
```python
# 修改前
model.load_state_dict(remove_fc(torch.load(model_urls['resnet50'])))

# 修改后
model.load_state_dict(remove_fc(torch.load(model_urls['resnet50'], weights_only=False)))
```

**影响**：无。仅恢复旧版权重加载行为。

---

#### 修改 3：`clustercontrast/models/resnet_ibn_a.py`

**原因**：同上，两处 `torch.load` 均需修复。

**修改内容**：
```python
# 修改前（两处）
state_dict = torch.load(model_urls['ibn_resnet50a'], map_location=torch.device('cpu'))['state_dict']
state_dict = torch.load(model_urls['ibn_resnet101a'], map_location=torch.device('cpu'))['state_dict']

# 修改后（两处）
state_dict = torch.load(model_urls['ibn_resnet50a'], map_location=torch.device('cpu'), weights_only=False)['state_dict']
state_dict = torch.load(model_urls['ibn_resnet101a'], map_location=torch.device('cpu'), weights_only=False)['state_dict']
```

**影响**：无。

---

#### 修改 4：`clustercontrast/models/clip/clip.py`

**原因**：同上。

**修改内容**：
```python
# 修改前
state_dict = torch.load(model_path, map_location="cpu")

# 修改后
state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
```

**影响**：无。

---

#### 修改 5：`clustercontrast/utils/serialization.py`

**原因**：同上。

**修改内容**：
```python
# 修改前
checkpoint = torch.load(fpath, map_location=torch.device('cpu'))

# 修改后
checkpoint = torch.load(fpath, map_location=torch.device('cpu'), weights_only=False)
```

**影响**：无。仅恢复旧版权重加载行为。

---

#### 修改 6：`train_regdb.py` —— Pillow `ANTIALIAS` 兼容性修复

**原因**：Pillow 10+ 删除了 `Image.ANTIALIAS` 属性。

**修改内容**：
```python
# 修改前
img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)

# 修改后
img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
```

**影响**：无。`LANCZOS` 与 `ANTIALIAS` 为同一插值算法，图像预处理结果不变。

---

#### 修改 7：`test_regdb.py` —— 多处修复与增强

**修改 7a：Pillow `ANTIALIAS` 兼容性修复**
```python
# 修改前
img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)

# 修改后
img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
```

**修改 7b：错误导入修复**
```python
# 修改前
from clustercontrast.utils.data.preprocessor import Preprocessor, Preprocessor_aug

# 修改后
from clustercontrast.utils.data.preprocessor import Preprocessor
```
`Preprocessor_aug` 在 `preprocessor.py` 中不存在，且 `test_regdb.py` 中从未实际使用该类。

**修改 7c：checkpoint 文件名修正**
```python
# 修改前
checkpoint = load_checkpoint(osp.join(logs_dir_root+'/'+str(trial),
    'full_model_with_unified_loss_and_soft_label_loss_and_global_clustering_model_best.pth.tar'))

// 修改后
checkpoint = load_checkpoint(osp.join(logs_dir_root+'/'+str(trial), 'train_model_best.pth.tar'))
```
原始文件名与实际训练保存的 `train_model_best.pth.tar`（由 `save_name='train_'` + `'model_best.pth.tar'` 拼接）不一致。

**修改 7d：新增 `--trial` 参数，支持单 trial 评估**
```python
# 新增
parser.add_argument('--trial', type=int, default=0,
                    help="evaluate specific trial only (1-10); default 0 means all trials")

trial_list = range(1, 11) if args.trial == 0 else [args.trial]
```
原始代码只能一次性评估全部 10 个 trials，修改后支持训练 1 个 trial → 立即评估 1 个 trial 的串行流程。

**修改 7e：评估日志独立目录**
```python
# 新增
eval_log_name = 'regdb_eval'
eval_logs_dir_root = osp.join(args.logs_dir+'/'+eval_log_name)

os.makedirs(osp.join(eval_logs_dir_root, str(trial)), exist_ok=True)
sys.stdout = Logger(osp.join(eval_logs_dir_root, str(trial), 'test_log.txt'))
```
每个 trial 的评估日志独立保存到 `logs/regdb_eval/$trial/test_log.txt`，便于后续汇总分析。

---

#### 修改 8：`run_train_regdb.sh` —— 路径适配与流程重构

**修改前**：
```bash
# 10 Trials
for trial in 1 2 3 4 5 6 7 8 9 10
do
CUDA_VISIBLE_DEVICES=0,1 \
 python train_regdb.py -mb CMhybrid -b 128 -a agw -d regdb_rgb \
 --iters 100 --momentum 0.1 --eps 0.3 --num-instances 16 --trial $trial \
 --data-dir "/data/wml/dataset/RegDB/"
done
```

**修改后**：
```bash
for trial in 1 2 3 4 5 6 7 8 9 10
do
    echo "Starting Trial ${trial}"

    echo "  Training..."
    CUDA_VISIBLE_DEVICES=0,1 \
     python train_regdb.py -mb CMhybrid -b 128 -a agw -d regdb_rgb \
     --iters 100 --momentum 0.1 --eps 0.3 --num-instances 16 --trial ${trial} \
     --data-dir "/root/autodl-tmp/RegDB/"

    echo "  Evaluating..."
    CUDA_VISIBLE_DEVICES=0,1 \
     python test_regdb.py -b 128 -a agw -d regdb_rgb \
     --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 \
     --data-dir "/root/autodl-tmp/RegDB/" --trial ${trial}
done

echo "All 10 trials completed"
```

**改动说明**：
1. `--data-dir` 从 `/data/wml/dataset/RegDB/` 改为 `/root/autodl-tmp/RegDB/`，适配当前环境。
2. 将纯训练循环改为 **"训练 → 评估" 串行循环**，每完成一个 trial 立即评估。

---

#### 修改 9：`run_test_regdb.sh` —— 路径适配

**修改内容**：`--data-dir` 从 `/data/wml/dataset/RegDB/` 改为 `/root/autodl-tmp/RegDB/`。

---

#### 修改 10：`summarize_results.py` —— 新增结果汇总脚本

**说明**：PGM-RegDB 目录下**新增**该脚本，用于自动解析 `logs/regdb_eval/{trial}/test_log.txt`，计算 10-trial 的 Rank-1/5/10/20、mAP、mINP 的均值与标准差，并输出到 `logs/regdb_eval/summary.txt`。

PGM-new 中**不存在**此脚本。

---

### 2.3 修改总览表

| 序号 | 文件 | 修改类型 | 修改内容 | 对算法结果的影响 |
|:---:|:---|:---|:---|:---|
| 1 | `clustercontrast/models/resnet.py` | 兼容性修复 + 路径回退 | `torch.load` 添加 `weights_only=False`；增加 `try-except` 回退到 `torchvision` | 无影响 |
| 2 | `clustercontrast/models/resnet_agw.py` | 兼容性修复 | `torch.load` 添加 `weights_only=False` | 无影响 |
| 3 | `clustercontrast/models/resnet_ibn_a.py` | 兼容性修复 | 两处 `torch.load` 添加 `weights_only=False` | 无影响 |
| 4 | `clustercontrast/models/clip/clip.py` | 兼容性修复 | `torch.load` 添加 `weights_only=False` | 无影响 |
| 5 | `clustercontrast/utils/serialization.py` | 兼容性修复 | `torch.load` 添加 `weights_only=False` | 无影响 |
| 6 | `train_regdb.py` | 兼容性修复 | `Image.ANTIALIAS` → `Image.Resampling.LANCZOS` | 无影响 |
| 7a | `test_regdb.py` | 兼容性修复 | `Image.ANTIALIAS` → `Image.Resampling.LANCZOS` | 无影响 |
| 7b | `test_regdb.py` | 错误导入修复 | 删除 `Preprocessor_aug` 导入 | 无影响 |
| 7c | `test_regdb.py` | Bug 修复 | checkpoint 文件名修正为 `train_model_best.pth.tar` | 修复测试无法加载模型的问题 |
| 7d | `test_regdb.py` | 功能增强 | 新增 `--trial` 参数，支持单 trial 评估 | 支持训练-评估串行流程 |
| 7e | `test_regdb.py` | 功能增强 | 评估日志独立目录 (`logs/regdb_eval/$trial/`) | 便于结果汇总与分析 |
| 8 | `run_train_regdb.sh` | 路径适配 + 流程重构 | `--data-dir` 改为 `/root/autodl-tmp/RegDB/`；改为训练→评估串行循环 | 适配当前环境 |
| 9 | `run_test_regdb.sh` | 路径适配 | `--data-dir` 改为 `/root/autodl-tmp/RegDB/` | 适配当前环境 |
| 10 | `summarize_results.py` | 新增脚本 | 自动汇总 10-trial 评估结果 | 新增辅助工具 |

### 2.4 未修改的核心算法部分

以下模块在复现代码中**保持与源代码完全一致**，未做任何改动：
- `train_regdb.py` 中的两阶段训练逻辑（Stage 1: DCL / Stage 2: PCLMP）
- `clustercontrast/trainers.py` 中的 `ClusterContrastTrainer_DCL` 和 `ClusterContrastTrainer_PCLMP`
- `clustercontrast/utils/matching_and_clustering.py` 中的 `two_step_hungarian_matching` 跨模态匹配逻辑
- DBSCAN 聚类参数与流程（eps=0.3, min_samples=4）
- `clustercontrast/models/agw.py` 模型结构
- `clustercontrast/models/cm.py` 中的 `ClusterMemory`
- `clustercontrast/models/losses.py` 中的所有 loss 函数
- `clustercontrast/evaluators.py` 中的特征提取与评估逻辑

---

## 三、数据集清单

### 3.1 数据集概况

| 属性 | 详情 |
|:---|:---|
| **数据集名称** | RegDB |
| **采集设备** | 双相机系统（Visible + Thermal） |
| **总身份数** | 412 人（每个 trial 随机划分为 206 人训练 / 206 人测试） |
| **图像分辨率** | 原始可变，统一 resize 为 **288 × 144** |
| **评估协议** | 10-fold cross-validation |
| **训练/测试划分** | 每个 trial：约 206 IDs 训练，约 206 IDs 测试 |
| **训练数据/模态** | Visible 约 2,060 张 / Thermal 约 2,060 张 |
| **测试数据/模态** | Visible 约 2,060 张 Query + 约 2,060 张 Gallery |

### 3.2 数据集路径

```
/root/autodl-tmp/RegDB/
├── Visible/          # 可见光图像（约 4,120 张，10 个 trial 共用）
├── Thermal/          # 红外图像（约 4,120 张，10 个 trial 共用）
├── idx/              # 10-trial 划分索引文件
│   ├── train_visible_1.txt ~ train_visible_10.txt  （每文件 2,060 行）
│   ├── train_thermal_1.txt  ~ train_thermal_10.txt   （每文件 2,060 行）
│   ├── test_visible_1.txt  ~ test_visible_10.txt    （每文件 2,060 行）
│   └── test_thermal_1.txt  ~ test_thermal_10.txt     （每文件 2,060 行）
├── rgb_modify/       # 数据增强相关
└── ir_modify/        # 数据增强相关
```

### 3.3 数据集特点

- **单相机场景**：每个身份仅由一对 Visible-Thermal 相机采集，场景相对简单。
- **模态差异大**：Visible 图像包含丰富的颜色和纹理信息，Thermal 图像仅保留温度分布，两者在像素空间差异显著。
- **数据规模小**：仅约 4,000 张图像，对无监督方法的聚类质量要求更高。
- **10-trial 评估**：每次 trial 随机划分训练/测试集，10 次试验取平均，结果更稳定但方差较大。

---

## 四、实验框架与流程

### 4.1 整体架构

PGM 采用**两阶段训练策略**：

```
Stage 1: DCL (Deep Cluster Learning)
    ├── 初始化模型（ImageNet 预训练权重）
    ├── 对每个模态独立进行 DBSCAN 聚类
    ├── 生成伪标签
    ├── 使用 ClusterContrastTrainer_DCL 训练
    └── 保存 Stage 1 最优模型

Stage 2: PCLMP (Progressive Cross-modality Learning with Matching and Pseudo-labeling)
    ├── 加载 Stage 1 模型
    ├── 提取 RGB 和 IR 特征
    ├── DBSCAN 聚类生成伪标签
    ├── Progressive Graph Matching (PGM) 挖掘跨模态对应关系
    ├── 生成 cross-modality pseudo-labels
    ├── 使用 ClusterContrastTrainer_PCLMP 训练（含 ACCL）
    └── 保存 Stage 2 最优模型
```

### 4.2 核心模块说明

| 模块 | 文件 | 功能 |
|:---|:---|:---|
| **模型 backbone** | `clustercontrast/models/agw.py` | AGW (Attribute-Guided Weighting) 网络，含独立 visible_module / thermal_module，共享 base_resnet，使用 GeM pooling |
| **Stage 1 Trainer** | `clustercontrast/trainers.py` | `ClusterContrastTrainer_DCL`：单模态独立对比学习 warm-up |
| **Stage 2 Trainer** | `clustercontrast/trainers.py` | `ClusterContrastTrainer_PCLMP`：跨模态交替对比学习 |
| **PGM 匹配** | `clustercontrast/utils/matching_and_clustering.py` | `two_step_hungarian_matching`：两步匈牙利图匹配，解决聚类数量不平衡 |
| **聚类** | `train_regdb.py` | DBSCAN (eps=0.3, min_samples=4) 生成伪标签 |
| **数据加载** | `clustercontrast/datasets/regdb_rgb.py` / `regdb_ir.py` | RegDB 数据集专用加载器 |
| **评估** | `clustercontrast/evaluators.py` | 特征提取（L2 归一化 + 水平翻转平均）、pairwise distance、CMC / mAP / mINP 计算 |

### 4.3 Stage 2 关键流程（PGM）

```
1. 提取特征
   features_rgb ← model(RGB_images, modal=1)
   features_ir  ← model(IR_images, modal=2)

2. 独立聚类
   pseudo_labels_rgb ← DBSCAN(rerank_dist_rgb)
   pseudo_labels_ir  ← DBSCAN(rerank_dist_ir)

3. 生成聚类中心
   cluster_features_rgb ← generate_cluster_features(pseudo_labels_rgb, features_rgb)
   cluster_features_ir  ← generate_cluster_features(pseudo_labels_ir, features_ir)

4. Progressive Graph Matching
   i2r, r2i ← two_step_hungarian_matching(cluster_features_rgb, cluster_features_ir)
   （解决聚类数量不平衡，全局最优匹配）

5. 生成 cross-modality pseudo-labels
   rgb2ir_labels ← [i2r[label] for label in rgb_labels]
   ir2rgb_labels ← [r2i[label] for label in ir_labels]

6. 训练（PCLMP + ACCL）
   loss_ir   ← memory_ir(f_out_ir, labels_ir)
   loss_rgb  ← memory_rgb(f_out_rgb, labels_rgb)
   cross_loss ← alternate(memory_rgb(f_out_ir, ir2rgb_labels), memory_ir(f_out_rgb, rgb2ir_labels))
   loss_hybrid ← memory_hybrid(...)
   total_loss ← loss_ir + loss_rgb + λ1*cross_loss + λ2*loss_hybrid + λ3*loss_RC
```

### 4.4 训练参数设置

| 项目 | 设置 |
|:---|:---|
| **方法** | PGM (Progressive Graph Matching) |
| **数据集** | RegDB |
| **评估协议** | 10-fold cross-validation |
| **模型** | AGW (ResNet50 backbone) |
| **Memory Bank** | CMhybrid |
| **Batch Size** | 128 |
| **优化器** | Adam |
| **学习率 (lr)** | 0.00035 |
| **权重衰减 (weight_decay)** | 0.0005 |
| **Stage 1 epochs** | 50 |
| **Stage 2 epochs** | 50 |
| **每 epoch iterations** | 100 |
| **聚类算法** | DBSCAN |
| **DBSCAN eps (Stage 1)** | 0.3 |
| **DBSCAN eps (Stage 2)** | 0.6（测试时） |
| **DBSCAN min_samples** | 4 |
| **num_instances** | 16 |
| **momentum** | 0.1 |
| **输入尺寸** | 288 × 144 |
| **训练设备** | 单卡 RTX 5090 (32GB) |
| **CUDA_VISIBLE_DEVICES** | 0,1 |

### 4.5 训练与评估脚本

**批量训练并评估（10 trials）**：
```bash
cd /root/work/PGM-RegDB
sh run_train_regdb.sh
```

该脚本会顺序执行 trial 1~10，每个 trial 先训练后评估。

**独立测试单个 trial**：
```bash
CUDA_VISIBLE_DEVICES=0,1 python test_regdb.py \
    -b 128 -a agw -d regdb_rgb \
    --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 \
    --data-dir "/root/autodl-tmp/RegDB/" --trial 1
```

**结果汇总**：
```bash
cd /root/work/PGM-RegDB
python summarize_results.py
```

---

### 4.6 核心算法与公式

以下梳理 PGM 基线方法中关键模块的数学描述，公式与代码实现一一对应。

#### (1) 特征提取与 GeM Pooling

AGW 网络对 RGB 与 IR 分别使用独立的 `visible_module` / `thermal_module`，再共享 `base_resnet`。最后一层特征图 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$ 经过 Generalized Mean (GeM) Pooling：

$$
\mathbf{x}_{\text{pool}} = \left( \frac{1}{HW}\sum_{i=1}^{HW} x_i^{\,p} + \varepsilon \right)^{\!\frac{1}{p}}, \qquad p=3.0, \; \varepsilon=10^{-12}
$$

推理时对原始图像与水平翻转图像分别提取特征并取平均：

$$
\mathbf{f}_{\text{final}} = \frac{\mathbf{f} + \mathbf{f}_{\text{flip}}}{2}
$$

最终做 L2 归一化：$\mathbf{f} \leftarrow \mathbf{f} / \|\mathbf{f}\|_2$。

#### (2) Jaccard 距离与 DBSCAN 聚类

对每个样本 $i$，通过 FAISS 搜索得到前向 $k$ 近邻 $\mathcal{N}_k(i)$，再双向验证得到 **k-互惠邻居**：

$$
\mathcal{R}_k(i)=\{j\in\mathcal{N}_k(i)\mid i\in\mathcal{N}_k(j)\}
$$

经过 k-互惠扩展后，构造局部权重向量 $\mathbf{V}_i$（仅在扩展邻居位置非零）：

$$
V_{i,j}=\frac{\exp(-d_{ij})}{\sum_{t\in\mathcal{R}_k^{\text{exp}}(i)}\exp(-d_{it})},\qquad d_{ij}=2-2\mathbf{f}_i^{\top}\mathbf{f}_j
$$

Query Expansion（$k_2=6$）：$\mathbf{V}_i^{\text{QE}}=\frac{1}{k_2}\sum_{t\in\mathcal{N}_{k_2}(i)}\mathbf{V}_t$。

Jaccard 距离：

$$
\mathbf{D}_{\text{Jaccard}}(i,j)=1-\frac{\sum_{t}\min(V_{i,t},V_{j,t})}{2-\sum_{t}\min(V_{i,t},V_{j,t})}
$$

DBSCAN 在预计算的 Jaccard 距离矩阵上聚类（`eps=0.3, min_samples=4`），得到伪标签 $\hat{y}_i$。

#### (3) Cluster Memory（CMhybrid）

维护 $2C$ 个代理特征（前 $C$ 个为 mean proxy $\mathbf{p}_c^{(\mu)}$，后 $C$ 个为 hard proxy $\mathbf{p}_c^{(h)}$）。输入特征 $\mathbf{x}_i$ 与所有代理计算内积并经温度缩放：

$$
\mathbf{s}_i=\frac{\mathbf{x}_i\mathbf{P}^{\top}}{\tau},\qquad \tau=0.05
$$

拆分为 $\mathbf{s}^{(\mu)}$ 与 $\mathbf{s}^{(h)}$ 后，CMhybrid 损失为：

$$
\mathcal{L}_{\text{hybrid}}=\frac{1}{2}\Big[\mathcal{L}_{\text{CE}}(\mathbf{s}^{(h)},\hat{y})+\text{ReLU}\big(\mathcal{L}_{\text{CE}}(\mathbf{s}^{(\mu)},\hat{y})-r\big)\Big],\qquad r=0.2
$$

反向传播时动量更新代理（$m=0.1$）：

$$
\mathbf{p}_c\leftarrow m\cdot\mathbf{p}_c+(1-m)\cdot\mathbf{x}_i^{*},\quad \text{然后 } \mathbf{p}_c\leftarrow\frac{\mathbf{p}_c}{\|\mathbf{p}_c\|_2}
$$

其中 mean proxy 使用 batch 内该类样本均值，hard proxy 使用与当前 proxy 内积最小的样本。

#### (4) 两步匈牙利匹配（Progressive Graph Matching）

设 RGB 聚类中心 $\{\mathbf{c}_i^{\text{rgb}}\}_{i=1}^{N_r}$，IR 聚类中心 $\{\mathbf{c}_j^{\text{ir}}\}_{j=1}^{N_i}$（$N_r\ge N_i$）。代价矩阵：

$$
\mathbf{C}_{ij}=\frac{1}{\exp\big((\mathbf{c}_i^{\text{rgb}})^{\top}\mathbf{c}_j^{\text{ir}}\big)}
$$

补零为方阵后调用 `linear_sum_assignment` 求解最小代价完美匹配。Step-1 后未匹配的 RGB 簇在 Step-2/Step-3 中继续与全部 IR 簇进行级联匹配，最终得到映射 $r2i$（RGB$\to$IR）和 $i2r$（IR$\to$RGB）。

#### (5) Stage 2 总损失（PCLMP + ACCL）

$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{IR}}+\mathcal{L}_{\text{RGB}}+0.25\cdot\mathcal{L}_{\text{cross}}
$$

交替跨模态对比（ACCL）：奇数 epoch 用 RGB memory 约束 IR 特征（标签为 $i2r[\hat{y}_{\text{ir}}]$），偶数 epoch 用 IR memory 约束 RGB 特征（标签为 $r2i[\hat{y}_{\text{rgb}}]$）。

#### (6) 评估指标

**CMC@$k$**：前 $k$ 个返回结果中至少命中一个正确匹配的概率。

**mAP**：

$$
\text{AP}_i=\frac{1}{N_i^{\text{rel}}}\sum_{t:\,m_{i,t}=1}\frac{\sum_{j=1}^{t}m_{i,j}}{t},\qquad \text{mAP}=\frac{1}{N_q}\sum_i\text{AP}_i
$$

**mINP**：设最后一个正确匹配的位置为 $p_i^{\text{max}}$，则

$$
\text{INP}_i=\frac{N_i^{\text{rel}}}{p_i^{\text{max}}},\qquad \text{mINP}=\frac{1}{N_q}\sum_i\text{INP}_i
$$

---

## 五、实验结果

### 5.1 10-Trial 汇总结果（由 `summarize_results.py` 自动生成）

#### Visible → Thermal (V→T)

| 指标 | 均值 ± 标准差 |
|:---|:---|
| **Rank-1** | **83.19% ± 2.51%** |
| **Rank-5** | **89.02% ± 1.94%** |
| **Rank-10** | **91.98% ± 1.33%** |
| **Rank-20** | **94.85% ± 1.03%** |
| **mAP** | **77.77% ± 2.52%** |
| **mINP** | **65.53% ± 3.19%** |

#### Thermal → Visible (T→V)

| 指标 | 均值 ± 标准差 |
|:---|:---|
| **Rank-1** | **82.77% ± 1.94%** |
| **Rank-5** | **89.50% ± 1.62%** |
| **Rank-10** | **92.42% ± 1.46%** |
| **Rank-20** | **95.03% ± 1.10%** |
| **mAP** | **76.89% ± 2.07%** |
| **mINP** | **62.29% ± 2.52%** |

### 5.2 各 Trial 详细结果（源自 `logs/regdb_eval/*/test_log.txt`）

#### Visible → Thermal (V→T)

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 87.23% | 91.26% | 93.50% | 95.53% | 82.30% | 70.82% |
| 2 | 81.26% | 88.16% | 90.97% | 93.45% | 75.44% | 62.14% |
| 3 | 83.06% | 89.17% | 92.91% | 96.36% | 78.72% | 67.15% |
| 4 | 82.38% | 89.13% | 91.50% | 94.22% | 75.70% | 63.81% |
| 5 | 81.55% | 86.94% | 91.17% | 94.42% | 76.00% | 63.08% |
| 6 | 80.92% | 87.91% | 91.41% | 94.66% | 75.54% | 62.22% |
| 7 | 87.04% | 91.36% | 93.45% | 96.21% | 81.27% | 69.80% |
| 8 | 82.82% | 88.50% | 90.68% | 93.79% | 78.26% | 67.23% |
| 9 | 79.71% | 85.68% | 90.05% | 93.83% | 74.87% | 61.63% |
| 10 | 85.92% | 92.09% | 94.13% | 96.02% | 79.58% | 67.40% |
| **Mean±Std** | **83.19±2.51%** | **89.02±1.94%** | **91.98±1.33%** | **94.85±1.03%** | **77.77±2.52%** | **65.53±3.19%** |

#### Thermal → Visible (T→V)

| Trial | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | mINP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 85.83% | 91.99% | 94.37% | 96.55% | 80.39% | 67.39% |
| 2 | 81.84% | 88.88% | 92.09% | 94.56% | 75.24% | 60.42% |
| 3 | 82.62% | 90.29% | 92.62% | 96.12% | 77.18% | 62.47% |
| 4 | 81.94% | 88.74% | 92.14% | 94.71% | 75.71% | 61.82% |
| 5 | 81.02% | 88.06% | 90.68% | 93.54% | 75.42% | 59.75% |
| 6 | 81.80% | 88.50% | 92.28% | 94.76% | 75.14% | 59.50% |
| 7 | 85.73% | 92.52% | 95.29% | 96.84% | 79.20% | 64.41% |
| 8 | 83.50% | 88.69% | 90.78% | 93.74% | 77.69% | 63.27% |
| 9 | 79.37% | 87.23% | 90.78% | 94.08% | 73.71% | 59.30% |
| 10 | 84.08% | 90.10% | 93.20% | 95.44% | 79.17% | 64.54% |
| **Mean±Std** | **82.77±1.94%** | **89.50±1.62%** | **92.42±1.46%** | **95.03±1.10%** | **76.89±2.07%** | **62.29±2.52%** |

### 5.3 结果验证说明

- 所有实验结果均来自实际运行 `test_regdb.py` 生成的日志文件，保存在 `/root/work/PGM-RegDB/logs/regdb_eval/{1..10}/test_log.txt`。
- `summarize_results.py` 自动解析上述日志，计算均值与标准差，输出至 `/root/work/PGM-RegDB/logs/regdb_eval/summary.txt`。
- 评估时加载的模型检查点为 `/root/work/PGM-new/logs/regdb_s2/{trial}/train_model_best.pth.tar`（Stage 2 最优模型）。
- 测试过程包含 **Visible→Thermal** 和 **Thermal→Visible** 两个方向的完整评估。

### 5.4 结果分析

- **V→T Rank-1 标准差为 2.51%**，T→V 为 1.94%，说明 RegDB 的 10-trial 交叉验证结果存在较大波动。这是因为 RegDB 数据量小（仅 412 人），不同 trial 的随机划分对聚类质量影响显著。
- **最好 trial（Trial 1）**：V→T 87.23% / T→V 85.83%；**最差 trial（Trial 9）**：V→T 79.71% / T→V 79.37%，差距约 7~8 个百分点。
- **Rank-1 与 mAP 强正相关**：Rank-1 的提升通常伴随 mAP 的提升。
- **Rank-5/10/20 趋于饱和**：所有 trial 的 Rank-5 均超过 85%，Rank-10 超过 90%，Rank-20 超过 93%，说明 top-5/top-10 检索已较为成熟，主要提升空间在 Rank-1。

---

## 六、文件修改树状图

```
PGM-new/                                    PGM-RegDB/
│                                           │
├── train_regdb.py                          ├── train_regdb.py  (+ANTIALIAS修复)
├── test_regdb.py                           ├── test_regdb.py  (+多处修复与增强)
├── run_train_regdb.sh                      ├── run_train_regdb.sh  (+路径+流程重构)
├── run_test_regdb.sh                       ├── run_test_regdb.sh  (+路径适配)
│                                           ├── summarize_results.py  (新增)
│                                           ├── PGM-RegDB_基线复现报告.md  (新增)
│                                           └── RegDB复现报告清单.md  (本文件)
├── ChannelAug.py                           ├── ChannelAug.py  (未修改)
└── clustercontrast/                        └── clustercontrast/
    ├── models/                                 ├── models/
    │   ├── agw.py  (未修改)                    │   ├── agw.py  (未修改)
    │   ├── cm.py  (未修改)                     │   ├── cm.py  (未修改)
    │   ├── losses.py  (未修改)                 │   ├── losses.py  (未修改)
    │   ├── resnet.py                           │   ├── resnet.py  (+weights_only +路径回退)
    │   ├── resnet_agw.py                       │   ├── resnet_agw.py  (+weights_only)
    │   ├── resnet_ibn_a.py                     │   ├── resnet_ibn_a.py  (+weights_only×2)
    │   └── clip/clip.py                        │   └── clip/clip.py  (+weights_only)
    ├── trainers.py  (未修改)                   ├── trainers.py  (未修改)
    ├── evaluators.py  (未修改)                 ├── evaluators.py  (未修改)
    ├── datasets/  (未修改)                     ├── datasets/  (未修改)
    └── utils/                                  └── utils/
        ├── matching_and_clustering.py              ├── matching_and_clustering.py  (未修改)
        ├── faiss_rerank.py                         ├── faiss_rerank.py  (未修改)
        ├── serialization.py                        ├── serialization.py  (+weights_only)
        ├── logging.py                              ├── logging.py  (未修改)
        └── data/                                   └── data/  (未修改)
```

---

> **报告生成时间**：2026-04-30  
> **报告依据**：直接读取 `/root/work/PGM-RegDB/logs/regdb_eval/*/test_log.txt` 与 `/root/work/PGM-RegDB/logs/regdb_eval/summary.txt` 中的实测数据，以及 `diff` 命令对比 `/root/work/PGM-new` 与 `/root/work/PGM-RegDB` 的代码差异。
