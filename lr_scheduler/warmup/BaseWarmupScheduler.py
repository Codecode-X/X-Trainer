from torch.optim.lr_scheduler import _LRScheduler

class BaseWarmupScheduler(_LRScheduler):
    """
    基类 学习率预热包装器
    该类继承自 _LRScheduler，提供学习率预热包装器通用结构。
    
    子类需要实现以下方法：
        - __init__()：初始化方法
        - get_lr()：计算下一个学习率
    """
    def __init__(self, cfg, successor, last_epoch=-1):
        """ 初始化方法 
        参数:
            - optimizer (Optimizer): 优化器 
            - successor (LRScheduler): 后续的学习率调度器
            - last_epoch (int): 上一个周期 (当前结束的周期)
                - 手动恢复训练时，可以传入上一个周期的值，确保学习率从正确的位置开始衰减。
                - 默认值：-1，表示还未开始训练。

        配置:
            - warmup_recount (bool): cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT: 是否在预热结束后重置周期
            - warmup_epoch (int): cfg.LR_SCHEDULER.WARMUP.EPOCHS: 预热周期
            - verbose (bool): cfg.VERBOSE: 是否打印信息
        """
        warmup_epoch = cfg.LR_SCHEDULER.WARMUP.EPOCHS  # 预热周期
        verbose = cfg.VERBOSE  # 是否打印日志
        self.warmup_recount = cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT  # 是否在预热结束后重置周期


        self.successor = successor  # 后续的学习率调度器
        self.warmup_epoch = warmup_epoch  # 预热周期

        optimizer = successor.optimizer # 优化器

        super().__init__(optimizer, last_epoch, verbose)  # 调用父类的初始化方法

    def get_lr(self):
        """
        用于计算下一个学习率的逻辑 (实现 _LRScheduler 的 get_lr())
        当需要自定义调度器时，需要实现 _LRScheduler 的 get_lr() 来定义学习率的计算方式。

        返回:
            - list: 包含每个参数组的计算后学习率的列表。

        其他:
            - 和 get_last_lr() 方法的区别是，get_last_lr() 返回的是上一个周期的学习率，而 get_lr() 返回的是下一个周期的学习率。
        """
        raise NotImplementedError

    def step(self, epoch=None):
        """
        重写 _LRScheduler 更新学习率的方法

        参数:
            - epoch (int): 当前周期数
                - 如果训练中断后恢复，传入当前的 epoch 可以确保学习率从正确的位置开始衰减。
                - 默认值：None，表示当前周期数为上一个周期数 + 1。
        """
        if self.last_epoch >= self.warmup_epoch:  # 如果预热周期结束
            # 如果需要在预热结束后重置周期
            if self.warmup_recount:
                print("由于设置了 warmup_recount，且现在预热结束，因此重置 epoch 为 -1")
                self.last_epoch = -1  # 重置周期数

            self.successor.step(epoch)  # 使用后续的学习率调度器更新学习率
            self._last_lr = self.successor.get_last_lr()  # 获取最新的学习率
        else:
            super().step(epoch)  # 否则使用父类的方法更新学习率
