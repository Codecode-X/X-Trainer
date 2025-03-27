# X-Trainer 框架



==配置文件最后需要检查代码中用到的所有配置，筛选出需要的配置。==

> by: Junhao Xiao

DataManager

* build_data_loader

  - build_dataset

  - build_sampler

  - build_transform







### 准备工作

```python
def __init__(self, cfg):
        """
        初始化训练器。
        主要包括：
        * 初始化相关属性，读取配置信息
        * 构建数据加载器
        * 构建并注册模型，优化器，学习率调度器；并初始化模型
        * 构建评估器
        """
```

**构建数据加载器**

```python
dm = DataManager(self.cfg) # 通过配置创建数据管理器
self.dm = dm  # 保存数据管理器

self.train_loader_x = dm.train_loader_x # 有标签训练数据加载器
self.train_loader_u = dm.train_loader_u  # 无标签训练数据加载器 (可选，可以为 None

self.val_loader = dm.val_loader  # 验证数据加载器 (可选，可以为 None
self.test_loader = dm.test_loader # 测试数据加载器

self.num_classes = dm.num_classes # 类别数
self.lab2cname = dm.lab2cname  # 类别名称字典 {label: classname}

```

**构建模型，优化器，学习率调度器**

```python
self.model, self.optim, self.sched = self.init_model(cfg)

```

**构建评估器**

```python
self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname) 

```

---

### 训练

```python
    def before_train(self):
        """
        训练前的操作。 (可选子类实现)
        
        主要包括：
        * 设置输出目录
        * 如果输出目录存在检查点，则恢复检查点
        * 初始化 summary writer
        * 记录开始时间（用于计算经过的时间）
        """
```



```python
      def before_epoch(self):
        """
        每个 epoch 前的操作。 (可选子类实现)  
        未实现
        """
        pass
```



```python
   def run_epoch(self):
        """
        执行每个 epoch 的训练。 (子类可重写)
        此处采用标准有标签数据的训练模式

        主要包括：
        * 设置模型为训练模式
        * 初始化度量器：损失度量器、批次时间度量器、数据加载时间度量器
        * 开始迭代
            — 遍历有标签数据集 train_loader_x
            — 前向和反向传播，获取损失
            — 打印日志 (epoch、batch、时间、数据加载时间、损失、学习率、剩余时间)
        """
```

  

```python
def after_epoch(self):
        """
        每个 epoch 后的操作。 (可选子类实现)
        
        主要包括：
        * 判断模型保存条件：是否是最后一个 epoch、是否需要验证、是否满足保存检查点的频率
        * 根据条件保存模型
        """
```



```python
    def after_train(self):
        """
        训练后的操作。 (可选子类实现)
        
        主要包括：
        * 如果训练后需要测试，则测试，并保存最佳模型；否则保存最后一个 epoch 的模型
        * 打印经过的时间
        * 关闭 writer
        """
```

**外部调用进行训练**

```python
# 如果不是仅评估模式，则进行训练
if not args.no_train:
    trainer.train()
```



### 测试



```python
```

**外部调用进行测试**

```python
# 如果仅评估模式，则加载模型并测试
if args.eval_only:
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer.test()
    return
```

