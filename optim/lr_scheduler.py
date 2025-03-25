"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
提供对 AVAI_SCHEDS 中的调度器进行预热的功能：常数预热调度器 和 线性预热调度器
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler

AVAI_SCHEDS = ["single_step", "multi_step", "cosine"] # 可用的学习率调度器列表

class _BaseWarmupScheduler(_LRScheduler):
    """基类 学习率预热调度器"""
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1, 
        verbose=False
    ):
        """ 初始化方法 
        参数:
            optimizer (Optimizer): 优化器 
            successor (_LRScheduler): 后续的学习率调度器
            warmup_epoch (int): 预热周期
            last_epoch (int): 上一个周期 (当前结束的周期)
            verbose (bool): 是否打印信息
        """
        super().__init__(optimizer, last_epoch, verbose)  # 调用父类的初始化方法
        self.successor = successor  # 后续的学习率调度器
        self.warmup_epoch = warmup_epoch  # 预热周期

    def get_lr(self):
        """获取学习率的方法（需要子类实现）"""
        raise NotImplementedError

    def step(self, epoch=None):
        """ 更新学习率的方法 """
        if self.last_epoch >= self.warmup_epoch:  # 如果当前周期大于等于预热周期
            self.successor.step(epoch)  # 使用后续的学习率调度器更新学习率
            self._last_lr = self.successor.get_last_lr()  # 获取最新的学习率
        else:
            super().step(epoch)  # 否则使用父类的方法更新学习率


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    """
    常数 学习率预热调度器
    在预热期间，学习率保持不变，等于 cons_lr
    预热结束后，学习率由后续调度器 successor 控制
    """
    # 初始化方法
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        """ 初始化方法 
        参数:
            optimizer (Optimizer): 优化器 
            successor (_LRScheduler): 后续的学习率调度器
            warmup_epoch (int): 预热周期
            cons_lr (float): 常数学习率
            last_epoch (int): 上一个周期 (当前结束的周期)
            verbose (bool): 是否打印信息
        """
        super().__init__(optimizer, successor, warmup_epoch, last_epoch, verbose)
        self.cons_lr = cons_lr  # 预热期间的常数学习率

    def get_lr(self):
        """(实现父类的抽象方法) 获取学习率的方法"""
        if self.last_epoch >= self.warmup_epoch:  # 如果当前周期大于等于 (结束) 预热周期
            return self.successor.get_last_lr()  # 返回后续调度器的学习率
        else: # 如果当前周期还在预热周期内
            return [self.cons_lr for _ in self.base_lrs]  # 返回常数学习率


class LinearWarmupScheduler(_BaseWarmupScheduler):
    """
    线性 学习率预热调度器
    在预热期间，学习率线性变化
    预热结束后，学习率由后续调度器 successor 控制
    """
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        """ 初始化方法
        参数:
            optimizer (Optimizer): 优化器 
            successor (_LRScheduler): 后续的学习率调度器
            warmup_epoch (int): 预热周期
            min_lr (float): 最小学习率
            last_epoch (int): 上一个周期 (当前结束的周期)
            verbose (bool): 是否打印信息
        """
        super().__init__(optimizer, successor, warmup_epoch, last_epoch, verbose)
        self.min_lr = min_lr  # 预热期间的最小学习率

    def get_lr(self):
        """(实现父类的抽象方法) 获取学习率的方法"""
        if self.last_epoch >= self.warmup_epoch:  # 如果当前周期大于等于 (不在) 预热周期
            return self.successor.get_last_lr()  # 返回后续调度器的学习率
        
        # 如果当前周期还在预热周期内
        if self.last_epoch == 0:  # 如果是第一个周期
            return [self.min_lr for _ in self.base_lrs]  # 返回最小学习率
        return [
            lr * self.last_epoch/self.warmup_epoch for lr in self.base_lrs
        ]  # 否则返回线性变化的学习率


def build_lr_scheduler(optimizer, optim_cfg):
    """构建学习率调度器的函数包装器

    参数:
        optimizer (Optimizer): 优化器
        optim_cfg (CfgNode): 优化配置
    """
    lr_scheduler = optim_cfg.LR_SCHEDULER  # 学习率调度器类型
    stepsize = optim_cfg.STEPSIZE  # 步长
    gamma = optim_cfg.GAMMA  # 衰减率
    max_epoch = optim_cfg.MAX_EPOCH  # 最大周期

    if lr_scheduler not in AVAI_SCHEDS:  # 如果调度器类型不在可用调度器列表中
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "single_step": # 如果是单步调度器
        """ 单步调度器：每隔 stepsize 个周期，学习率乘以 gamma """
        if isinstance(stepsize, (list, tuple)): # 如果步长是列表或元组，取最后一个步长
            stepsize = stepsize[-1] 

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )

        if stepsize <= 0: # 如果步长小于等于 0，取 max_epoch 作为步长
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(  # 创建单步调度器
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step": # 如果是多步调度器
        """ 多步调度器：在 milestones(stepsize) 中的 epoch，学习率乘以 gamma """
        milestones = stepsize
        if not isinstance(milestones, (list, tuple)): # 如果步长不是列表或元组，抛出异常
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(milestones)}"
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR( # 创建多步调度器
            optimizer, milestones=milestones, gamma=gamma
        )

    elif lr_scheduler == "cosine": # 如果是余弦调度器
        """ 余弦调度器：学习率按照余弦函数变化 """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( # 创建余弦调度器
            optimizer, float(max_epoch)
        )

    if optim_cfg.WARMUP_EPOCH > 0: # 如果预热周期大于 0
        """ 
        预热调度器：用于对其他调度器（如单步、多步、余弦调度器）进行预热
        规则：
            预热结束后，学习率由后续调度器控制
            如果不重置周期，则将上一个 (当前结束) 周期设置为预热周期
            如果是常数预热调度器，学习率保持不变，等于 cons_lr
            如果是线性预热调度器，学习率线性变化
        """
        if not optim_cfg.WARMUP_RECOUNT: # 如果不重置周期，则将上一个 (当前结束) 周期设置为预热周期
            scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

        if optim_cfg.WARMUP_TYPE == "constant": # 如果是常数预热调度器
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_CONS_LR
            )

        elif optim_cfg.WARMUP_TYPE == "linear": # 如果是线性预热调
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_MIN_LR
            )

        else:
            raise ValueError

    return scheduler
