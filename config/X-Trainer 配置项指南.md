# X-Trainer 配置项指南

> by: Junhao Xiao - https://github.com/Codecode-X/X-Trainer.git



## 全局配置（Global）

| 配置项             | 类型 | 示例值     | 说明                                       |
| ------------------ | ---- | ---------- | ------------------------------------------ |
| **cfg.VERBOSE**    | bool | True       | 启用详细日志输出（如训练进度、指标等）     |
| **cfg.SEED**       | int  | 42         | 全局随机种子，确保实验可复现性             |
| **cfg.USE_CUDA**   | bool | True       | 自动检测并使用可用 GPU 加速计算              |
| **cfg.OUTPUT_DIR** | str  | "./output" | 输出目录（存储日志/模型/评估结果）         |
| **cfg.RESUME**     | str  | ""         | 断点续训路径（需包含`checkpoint.pth`文件） |

---



## 训练引擎配置（Engine）

### 基础配置
| 配置项                        | 类型 | 示例          | 说明                                                         |
| ----------------------------- | ---- | ------------- | ------------------------------------------------------------ |
| **cfg.TRAINER.NAME**          | str  | "TrainerClip" | 训练器类名（如`TrainerClip`）                                |
| **cfg.TRAINER.PREC**          | str  | "amp"         | 训练精度：<br>`fp32`-全精度，`fp16`-半精度，`amp`-自动混合精度（显存优化） |
| **cfg.TRAINER.FROZEN_LAYERS** | bool | False         | 冻结基础网络层（仅训练分类头）                               |

### 训练流程
| 配置项                        | 类型 | 示例  | 说明                                |
| ----------------------------- | ---- | ----- | ----------------------------------- |
| **cfg.TRAIN.DO_EVAL**         | bool | True  | 每个 epoch 后验证模型（保存最佳模型） |
| **cfg.TRAIN.NO_TEST**         | bool | False | 训练完成后跳过测试阶段              |
| **cfg.TRAIN.CHECKPOINT_FREQ** | int  | 5     | 模型保存频率（单位：epoch）         |
| **cfg.TRAIN.PRINT_FREQ**      | int  | 50    | 训练日志打印间隔（单位：batch）     |
| **cfg.TRAIN.MAX_EPOCH**       | int  | 100   | 最大训练轮次（单位：epoch）         |

### 测试流程
| 配置项                   | 类型 | 示例       | 说明                                                         |
| ------------------------ | ---- | ---------- | ------------------------------------------------------------ |
| **cfg.TEST.FINAL_MODEL** | str  | "best_val" | 测试模型选择：<br>`best_val`-验证集最佳，`last_step`-最终权重 |
| **cfg.TEST.SPLIT**       | str  | "test"     | 测试数据集选择：<br>`val`-验证集 `test`-测试集               |

---



## 数据管理配置（Data）

### 数据集（dataset）

#### 基础配置

| 配置项                      | 类型 | 示例                | 说明                                                   |
| --------------------------- | ---- | ------------------- | ------------------------------------------------------ |
| **cfg.DATASET.NAME**        | str  | "Caltech101"        | 数据集类名                                             |
| **cfg.DATASET.DATASET_DIR** | str  | "/root/caltech-101" | 数据集根目录路径                                       |
| **cfg.DATASET.SPLIT**       | list | [0.7, 0.1, 0.2]     | 训练/验证/测试集划分比例（总和≤1）                     |
| **cfg.DATASET.NUM_SHOTS**   | int  | -1                  | 每类样本数：<br>`-1`-全量，`0`-零样本，`≥1`-小样本学习 |

#### 分类数据集基础配置

| 配置项                    | 类型 | 示例                                     | 说明           |
| ------------------------- | ---- | ---------------------------------------- | -------------- |
| **cfg.DATASET.IMAGE_DIR** | str  | "/root/caltech-101/101_ObjectCategories" | 数据集图像目录 |

#### 回归数据集基础配置

*暂未实现*

#### 特定数据集配置

| 数据集         | 配置项 | 类型 | 示例 | 说明       |
| -------------- | ------ | ---- | ---- | ---------- |
| **Caltech101** | -      | -    | -    | 无额外配置 |

-----

### 数据加载（dataloader）
| 配置项                          | 类型 | 示例  | 说明                                           |
| ------------------------------- | ---- | ----- | ---------------------------------------------- |
| **cfg.DATALOADER.BATCH_SIZE**   | int  | 32    | 训练批大小（影响显存占用）                     |
| **cfg.DATALOADER.NUM_WORKERS**  | int  | 4     | 数据加载并行进程数（建议=CPU 核心数）           |
| **cfg.DATALOADER.K_TRANSFORMS** | int  | 1     | 每种增强在原始图像上（横向）重复应用次数       |
| **cfg.DATALOADER.RETURN_IMG0**  | bool | False | 是否返回原始未增强图像（用于可视化或对比学习） |

---

### 数据采样（samplers）

| 配置项                    | 类型 | 示例                | 说明               |
| ------------------------- | ---- | ------------------- | ------------------ |
| **cfg.SAMPLER.TRAIN_SP ** | str  | "RandomSampler"     | 训练数据采样器类名 |
| **cfg.SAMPLER.TEST_SP**   | str  | "SequentialSampler" | 测试数据采样器类名 |

----

### 图像增强（transforms）

#### 基础配置
| 配置项                                   | 类型 | 示例                                   | 说明                                                   |
| ---------------------------------------- | ---- | -------------------------------------- | ------------------------------------------------------ |
| **cfg.INPUT.SIZE**                       | int  | 224                                    | 输入图像统一尺寸（需匹配模型）                         |
| **cfg.INPUT.INTERPOLATION**              | str  | "bilinear"                             | 图像缩放插值方法：<br>`bilinear`, `bicubic`, `nearest` |
| **cfg.INPUT.BEFORE_TOTENSOR_TRANSFORMS** | list | `["RandomResizedCrop", "ColorJitter"]` | 在转换为张量之前的数据增强方法列表                     |
| **cfg.INPUT.AFTER_TOTENSOR_TRANSFORMS**  | list | `["Normalize"]`                        | 在转换为张量之后的数据增强方法列表                     |

#### 特定图像增强策略配置表

| 增强策略                                  | 配置项                                         | 示例                        | 说明                                                         |
| ----------------------------------------- | ---------------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| **AutoAugment**                           |                                                |                             | 从 25 个最佳子策略中随机选择。<br />适用于不同数据集（ImageNet，CIFAR10，SVHN） |
| ├─ **ImageNetPolicy**                     | `cfg.INPUT.ImageNetPolicy.fillcolor`           | (128,128,128)               | 图像填充颜色（RGB 值）                                        |
| ├─ **CIFAR10Policy**                      | `cfg.INPUT.CIFAR10Policy.fillcolor`            | (128,128,128)               |                                                              |
| └─ **SVHNPolicy**                         | `cfg.INPUT.SVHNPolicy.fillcolor`               | (128,128,128)               |                                                              |
| **RandomAugment**                         |                                                |                             | 随机组合增强操作                                             |
| ├─ **RandomIntensityAugment**             | `cfg.INPUT.RandomIntensityAugment.n`           | 2                           | 随机选择 n 个增强操作                                          |
|                                           | `cfg.INPUT.RandomIntensityAugment.m`           | 10                          | 增强强度（0-30，值越大效果越强）                             |
| └─ **ProbabilisticAugment**               | `cfg.INPUT.ProbabilisticAugment.n`             | 2                           | 随机选择 n 个增强操作                                          |
|                                           | `cfg.INPUT.ProbabilisticAugment.p`             | 0.6                         | 每个操作的应用概率                                           |
| **Cutout**                                |                                                |                             | 随机遮挡图像区域                                             |
|                                           | `cfg.INPUT.Cutout.n_holes`                     | 1                           | 每张图像的遮挡区域数量                                       |
|                                           | `cfg.INPUT.Cutout.length`                      | 16                          | 每个方形遮挡区域的边长（像素）                               |
| **GaussianNoise**                         |                                                |                             | 添加高斯噪声                                                 |
|                                           | `cfg.INPUT.GaussianNoise.mean`                 | 0                           | 噪声均值（通常保持 0 不变）                                    |
|                                           | `cfg.INPUT.GaussianNoise.std`                  | 0.15                        | 噪声强度（值越大噪声越明显）                                 |
|                                           | `cfg.INPUT.GaussianNoise.p`                    | 0.5                         | 应用概率（0-1 之间）                                          |
| **Random2DTranslation**                   |                                                |                             | 缩放后随机裁剪                                               |
|                                           | `cfg.INPUT.Random2DTranslation.p`              | 0.5                         | 执行概率（0=禁用，1=始终应用）                               |
| **Normalize**                             | `cfg.INPUT.PIXEL_MEAN`                         | [0.485,0.456,0.406]         | 图像归一化均值（需与预训练模型一致）                         |
|                                           | `cfg.INPUT.PIXEL_STD`                          | [0.229,0.224,0.225]         | 图像归一化标准差                                             |
| **InstanceNormalization**                 | -                                              | -                           | 实例归一化（无配置参数）                                     |
| ---------**特定模型增强策略**------------ | ---------------------------------------------- | --------------------------- | -------------------------------------------------------      |
| **TransformClipVisual**（CLIP）           | -                                              | -                           | CLIP 专用预处理（无配置参数）                                 |

---



## 模型配置（Model）

### 基础配置

| 配置项                          | 类型 | 示例                              | 说明                 |
| ------------------------------- | ---- | --------------------------------- | -------------------- |
| **cfg.MODEL.NAME**              | str  | "Clip"                            | 模型类名（如`Clip`） |
| **cfg.MODEL.INIT_WEIGHTS_PATH** | str  | "log/my_model/model-best.pth.tar" | 预训练权重路径       |

### 特定模型配置

| 特定模型 | 配置项                      | 类型 | 示例            | 说明                       |
| -------- | --------------------------- | ---- | --------------- | -------------------------- |
| **Clip** |                             |      |                 | 经典对比学习模型           |
| ├─       | **cfg.MODEL.pretrained**    | str  | "ViT-B/16"      | Clip 的预训练模型名         |
| └─       | **cfg.MODEL.download_root** | str  | "~/.cache/clip" | Clip 预训练权重下载保存目录 |

---



## 学习率策略配置（LR）

### 调度器（lr_scheduler）

#### 基础配置

| 配置项                    | 类型 | 示例          | 说明                                                         |
| ------------------------- | ---- | ------------- | ------------------------------------------------------------ |
| **cfg.LR_SCHEDULER.NAME** | str  | "MultiStepLR" | `MultiStepLR`（阶梯下降）, `CosineLR`（余弦退火）, `SingleStepLrScheduler`（单步下降） |

#### 特定调度器配置

| 特定调度器                | 配置项                      | 类型      | 示例         | 说明                            |
| ------------------------- | --------------------------- | --------- | ------------ | ------------------------------- |
| **MultiStepLrScheduler**  |                             |           |              | 多步学习率调度器                |
| ├─                        | cfg.LR_SCHEDULER.MILESTONES | list[int] | [30, 60, 90] | 学习率下降的周期数列表          |
| └─                        | cfg.LR_SCHEDULER.GAMMA      | float     | 0.1          | 学习率衰减系数                  |
| **SingleStepLrScheduler** |                             |           |              | 单步学习率调度器                |
| ├─                        | cfg.LR_SCHEDULER.STEP_SIZE  | int       | 50           | 步长，多少个 epoch 后降低学习率 |
| └─                        | cfg.LR_SCHEDULER.GAMMA      | float     | 0.1          | 学习率衰减系数                  |
| **CosineLrScheduler**     | -                           | -         |              | 余弦学习率调度器                |

### 预热器（lr_scheduler/warmup）

#### 基础配置

| 配置项                                     | 类型 | 示例                    | 说明                     |
| ------------------------------------------ | ---- | ----------------------- | ------------------------ |
| **cfg.LR_SCHEDULER.WARMUP.NAME**           | str  | "LinearWarmupScheduler" | 预热器类名               |
| **cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT** | bool | True                    | 是否在预热结束后重置周期 |
| **cfg.LR_SCHEDULER.WARMUP.EPOCHS**         | int  | 5                       | 预热周期                 |

#### 特定预热器配置

| 特定预热器                  | 配置项                              | 类型  | 示例  | 说明       |
| --------------------------- | ----------------------------------- | ----- | ----- | ---------- |
| **ConstantWarmupScheduler** | **cfg.LR_SCHEDULER.WARMUP.CONS_LR** | float | 0.001 | 常数学习率 |
| **LinearWarmupScheduler**   | **cfg.LR_SCHEDULER.WARMUP.MIN_LR**  | float | 1e-6  | 最小学习率 |

---



## 评估器配置（evaluator）

### 基础配置

| 配置项                 | 类型 | 示例                      | 说明       |
| ---------------------- | ---- | ------------------------- | ---------- |
| **cfg.EVALUATOR.NAME** | str  | "EvaluatorClassification" | 评估器类名 |

### 特定评估器配置

| 优化器                  | 配置项                  | 类型 | 示例 | 说明                   |
| ----------------------- | ----------------------- | ---- | ---- | ---------------------- |
| EvaluatorClassification |                         |      |      | 分类任务评估器         |
| ├─                      | cfg.EVALUATOR.per_class | bool | True | 是否评估每个类别的结果 |
| └─                      | cfg.EVALUATOR.calc_cmat | bool | True | 是否计算混淆矩阵       |

----



## 优化器配置（optimizer）

### 基础配置

| 配置项                            | 类型      | 示例                 | 说明                                |
| --------------------------------- | --------- | -------------------- | ----------------------------------- |
| **cfg.OPTIMIZER.NAME**            | str       | "AdamW"              | 优化器类名，如 `"Adam"`、`"SGD"` 等 |
| **cfg.OPTIMIZER.LR**              | float     | 0.001                | 全局学习率                          |
| **cfg.OPTIMIZER.STAGED_LR**       | bool      | True                 | 是否使用分阶段学习率                |
| ├─ **cfg.OPTIMIZER.NEW_LAYERS**   | list[str] | ["layer1", "layer2"] | 新增的网络层（通常用于适配特定任务  |
| └─ **cfg.OPTIMIZER.BASE_LR_MULT** | float     | 0.1                  | 基础层学习率缩放系数（一般 <1）     |


### 特定优化器配置
| 优化器 | 配置项 | 类型 | 示例 | 说明                                   |
|------------|----------|--------|------|------------|
| **Adam**   |          |        |        | 适合大多数深度学习任务 |
| ├─ | **cfg.OPTIMIZER.BETAS**              | Tuple[float, float] | `(0.9, 0.999)` | Adam 的 beta 参数                      |
| ├─          | **cfg.OPTIMIZER.EPS**                | float               | `1e-8`         | 除数中的常数，避免除零错误             |
| ├─          | **cfg.OPTIMIZER.WEIGHT_DECAY**       | float               | `0.01`         | 权重衰减                               |
| └─          | **cfg.OPTIMIZER.AMSGRAD**            | bool                | `False`        | 是否使用 AMSGrad                       |
| **SGD**    |          |        |        | 需要精细调参但可能获得更好结果 |
| ├─ | **cfg.OPTIMIZER.MOMENTUM**           | float               | `0.9`          | 动量                                   |
| ├─          | **cfg.OPTIMIZER.WEIGHT_DECAY**       | float               | `0.0005`       | 权重衰减                               |
| ├─          | **cfg.OPTIMIZER.DAMPENING**          | float               | `0.0`          | 阻尼                                   |
| └─          | **cfg.OPTIMIZER.NESTEROV**           | bool                | `True`         | 是否使用 Nesterov 动量 |
| **RMSprop**|          |        |        | 适合非平稳目标函数 |
| ├─ | **cfg.OPTIMIZER.ALPHA**              | float               | `0.99`         | 移动平均系数                           |
| ├─          | **cfg.OPTIMIZER.EPS**                | float               | `1e-8`         | 除数中的常数，避免除零错误             |
| ├─          | **cfg.OPTIMIZER.WEIGHT_DECAY**       | float               | `0.0001`       | 权重衰减                               |
| ├─          | **cfg.OPTIMIZER.MOMENTUM**           | float               | `0.9`          | 动量                                   |
| └─          | **cfg.OPTIMIZER.CENTERED**           | bool                | `False`        | 是否使用中心化的 RMSprop |
| **RAdam**  |          |        |        | 自适应学习率的鲁棒版本 |
| ├─ | **cfg.OPTIMIZER.BETAS**              | Tuple[float, float] | `(0.9, 0.999)` | RAdam 的 beta 参数                     |
| ├─          | **cfg.OPTIMIZER.EPS**                | float               | `1e-8`         | 除数中的常数，避免除零错误             |
| ├─          | **cfg.OPTIMIZER.WEIGHT_DECAY**       | float               | `0.01`         | 权重衰减                               |
| └─          | **cfg.OPTIMIZER.DEGENERATED_TO_SGD** | bool                | `False`        | 是否将 RAdam 退化为 SGD（无自适应性） |
| **AdamW**  |          |        |        | 改进的 Adam+ 正确权重衰减 |
| ├─ | **cfg.OPTIMIZER.BETAS**              | Tuple[float, float] | `(0.9, 0.999)` | AdamW 的 beta 参数                     |
| ├─          | **cfg.OPTIMIZER.EPS**                | float               | `1e-8`         | 除数中的常数，避免除零错误             |
| ├─          | **cfg.OPTIMIZER.WEIGHT_DECAY**       | float               | `0.01`         | 权重衰减                               |
| └─          | **cfg.OPTIMIZER.WARMUP**             | int                 | `1000`         | 预热步数（在一定步数内逐步增加学习率） |

