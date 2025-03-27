## X-Trainer 配置文件

#### **global :**

cfg.VERBOSE： 是否打印日志

cfg.SEED：随机种子

cfg.USE_CUDA： 是否使用cuda

cfg.OUTPUT_DIR：输出路径(日志、模型权重等)

cfg.RESUME： RESUME 目录，用以加载继续训练



#### **engine :**

调度其他所有模块，完成模型训练和测试的所有过程

cfg.TRAINER.NAME： 训练器类名，例如 TrainerClip

cfg.TRAINER.COOP.PREC： 采用训练精度 | ["fp16", "fp32", "amp"] "amp"精度混合训练减少现存占用

cfg.TRAINER.COOP.FROZEN_LAYERS：是否冻结模型某些蹭 | True，False

**训练**

cfg.TRAIN.DO_EVAL：每个 epoch 后是否需要进行验证 | 为True时：进行验证，并保存验证性能最好的模型

cfg.TRAIN.CHECKPOINT_FREQ：保存检查点的频率，即多少个epoch保存一次

cfg.TRAIN.PRINT_FREQ == 0： 训练期间打印日志的频率，即每训练多少个batch打印一次

cfg.TRAIN.MAX_EPOCH：最大优化周期



**测试**

cfg.TEST.NO_TEST： 训练完毕后是否进行测试

cfg.TEST.FINAL_MODEL：测试哪个模型 | 'best_val'-保存eval性能最好的模型；'last_step'-保存最后一个step得到的模型

cfg.TEST.SPLIT：测试哪个数据子集 | 'val'验证集 'test'测试集





#### data/data_manager :

cfg.DATALOADER.TRAIN_X.BATCH_SIZE: 批大小

cfg.DATALOADER.NUM_WORKERS: 用于数据加载的子进程数量

cfg.DATALOADER.K_TRANSFORMS：数据增强重复次数 | 训练模式生效；测试模式下默认为 1

cfg.DATALOADER.RETURN_IMG0：是否记录未增强的原始图像 | 默认为 False

#### **data/dataset :**

cfg.DATASET.NAME：数据集类名，例如 Caltech101

cfg.DATASET.DATASET_DIR: 数据集的存放目录路径，例如：/root/autodl-tmp/caltech-101

cfg.DATASET.IMAGE_DIR: 图像目录，例如：/root/autodl-tmp/caltech-101/101_ObjectCategories

cfg.DATASET.SPLIT：[训练集比例，验证集比例，测试集比例]，例如[0.6, 0.2, 0.2]

cfg.DATASET.NUM_SHOTS：训练集每个类别的样本数量，-1表示进行全采样，0表示zero-shot，>=1表示few-shot



#### **data/samplers :**

cfg.DATALOADER.TRAIN.SAMPLER ：训练数据采样器类名，例如 RandomSampler（随机采样）

cfg.DATALOADER.TEST.SAMPLER：测试数据采样器类名，例如 SequentialSampler（顺序采样）



#### **data/transforms :**

cfg.INPUT.BEFORE_TOTENSOR_TRANSFORMS：在转换为张量之前的数据增强方法列表

cfg.INPUT.AFTER_TOTENSOR_TRANSFORMS：在转换为张量之后的数据增强方法列表

cfg.INPUT.PIXEL_MEAN: 均值，[0.485, 0.456, 0.406]，默认ImageNet

cfg.INPUT.PIXEL_STD: 标准差，[0.229, 0.224, 0.225]，默认ImageNet

cfg.INPUT.SIZE：模型统一输入图片尺寸，默认224

cfg.INPUT.INTERPOLATION：resize输入图片时用的插值方法，包括：bilinear 双线性插值，bicubic 双三次插值，nearest 最近邻插值

-----

###### AutoAugment / ImageNetPolicy, CIFAR10Policy, SVHNPolicy：

从 ImageNet, CIFAR10Policy, SVHNPolicy 上的 25 个最佳子策略中随机选择一个。

cfg.INPUT.{增强类名}.fillcolor：填充颜色，默认(128, 128, 128)。 例如cfg.ImageNetPolicy.fillcolor, cfg.INPUT.CIFAR10Policy.fillcolor，cfg.INPUT.SVHNPolicy.fillcolor

###### Cutout:

随机从图像中遮盖一个或多个补丁。

cfg.INPUT.Cutout.n_holes: 每张图像中要遮盖的补丁数量 | 默认值为 1

cfg.INPUT.Cutout.length: 每个方形补丁的边长（以像素为单位） | 默认值为 16

###### GaussianNoise:

高斯噪声增强

cfg.INPUT.GaussianNoise.mean: 高斯噪声的均值 | 默认值为 0。

cfg.INPUT.GaussianNoise.std: 高斯噪声的标准差 | 默认值为 0.15。

cfg.INPUT.GaussianNoise.p: 应用高斯噪声的概率 | 默认值为 0.5。

###### InstanceNormalization:

实例归一化

None

###### Normalize:

图像归一化

cfg.INPUT.PIXEL_MEAN: 均值，float

cfg.INPUT.PIXEL_STD: 标准差，float

###### Random2DTranslation：

 将给定的图像从 (height, width) 尺寸调整为 (height\*1.125, width\*1.125)，然后进行随机裁剪。

cfg.INPUT.Random2DTranslation.p：执行此操作的概率 | 默认值为 0.5。

###### RandomAugment / RandomIntensityAugment：

从预定义的增强操作列表中随机选择数量 n 的操作，并且随机选择其强度。  

cfg.INPUT.RandomIntensityAugment.n：操作数量 | 默认值：2

cfg.INPUT.RandomIntensityAugment.m：操作强度 | 默认值：10

###### RandomAugment / ProbabilisticAugment：

从预定义的增强操作列表中随机选择数量 n 的操作，每个操作以概率 p 应用。

cfg.INPUT.ProbabilisticAugment.n: 操作数量 | 默认值：2

cfg.INPUT.ProbabilisticAugment.p: 操作概率 | 默认值：0.6

###### TransformClipVisual：

None



#### **evaluation :**

cfg.EVALUATOR.NAME：评估器类名

###### EvaluatorClassification: 

cfg.EVALUATOR.per_class：是否评估每个类别的结果 | True,False

cfg.EVALUATOR.calc_cmat：是否计算混淆矩阵 | True,False



#### **model :**

cfg.MODEL.NAME： 模型类名 | 例如 Clip

cfg.MODEL.INIT_WEIGHTS_PATH： 初始（预训练）权重的所在路径（如果有，则载入进模型

###### model / Clip：

cfg.Clip.download_root： Clip预训练权重下载保存目录（和模型名拼接





#### lr_scheduler:

cfg.LR_SCHEDULER.NAME： 学习率调度器类名  (例如：MultiStepLrScheduler)

###### MultiStepLrScheduler: 

cfg.LR_SCHEDULER.MILESTONES：学习率下降的周期数列表

cfg.LR_SCHEDULER.GAMMA：学习率衰减系数。

###### SingleStepLrScheduler

cfg.LR_SCHEDULER.STEP_SIZE: 步长，多少个 epoch 后降低学习率。

cfg.LR_SCHEDULER.GAMMA: 学习率衰减系数。

###### CosineLrScheduler

None



#### lr_scheduler / warmup: 

cfg.LR_SCHEDULER.WARMUP.NAME:  预热调度器类名

cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT: 是否在预热结束后重置周期

cfg.LR_SCHEDULER.WARMUP.EPOCHS: 预热周期

###### ConstantWarmupScheduler:

cfg.LR_SCHEDULER.WARMUP.CONS_LR: 常数学习率

###### LinearWarmupScheduler:

cfg.LR_SCHEDULER.WARMUP.MIN_LR: 最小学习率



#### **optim：**

cfg.OPTIMIZER.NAME：优化器类名

cfg.OPTIMIZER.LR：学习率

cfg.OPTIMIZER.STAGED_LR：是否分阶段学习率（bool）

* cfg.OPTIMIZER.NEW_LAYERS：新层
* cfg.OPTIMIZER.BASE_LR_MULT：基础层学习率缩放系数（一般<1）

###### Adam

cfg.OPTIMIZER.BETAS (Tuple[float, float]): Adam 的 beta 参数

cfg.OPTIMIZER.EPS (float): 除数中的常数，避免除零错误

cfg.OPTIMIZER.WEIGHT_DECAY (float): 权重衰减

cfg.OPTIMIZER.AMSGRAD (bool): 是否使用 AMSGrad

###### Sgd

cfg.OPTIMIZER.MOMENTUM (float): 动量

cfg.OPTIMIZER.WEIGHT_DECAY (float): 权重衰减

cfg.OPTIMIZER.DAMPENING (float): 阻尼

cfg.OPTIMIZER.NESTEROV (bool): 是否使用 Nesterov 动量

###### Rmsprop

cfg.OPTIMIZER.ALPHA (float): 移动平均系数

cfg.OPTIMIZER.EPS (float): 除数中的常数，避免除零错误

cfg.OPTIMIZER.WEIGHT_DECAY (float): 权重衰减

cfg.OPTIMIZER.MOMENTUM (float): 动量

cfg.OPTIMIZER.CENTERED (bool): 是否使用中心化的 RMSprop

###### RAdam

cfg.OPTIMIZER.BETAS (Tuple[float, float]): Adam 的 beta 参数

cfg.OPTIMIZER.EPS (float): 除数中的常数，避免除零错误

cfg.OPTIMIZER.WEIGHT_DECAY (float): 权重衰减

OPTIMIZER.DEGENERATED_TO_SGD (bool): 是否将 RAdam 退化为 SGD

###### AdamW

cfg.OPTIMIZER.BETAS (Tuple[float, float]): Adam 的 beta 参数

cfg.OPTIMIZER.EPS (float): 除数中的常数，避免除零错误

cfg.OPTIMIZER.WEIGHT_DECAY (float): 权重衰减

cfg.OPTIMIZER.WARMUP (int): warmup 预热步数