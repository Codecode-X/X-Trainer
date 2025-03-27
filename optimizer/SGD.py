from .build import OPTIMIZER_REGISTRY
from torch.optim import SGD as TorchSGD

@OPTIMIZER_REGISTRY.register()
class Sgd(TorchSGD):
    """ Sgd 优化器 """
    def __init__(self, cfg, params=None):
        """
        初始化 Sgd 优化器

        参数:
            - cfg (CfgNode): 配置
            - params (iterable): 模型参数

        配置:
            - 优化器默认参数
                - OPTIMIZER.LR (float): 学习率
                - OPTIMIZER.MOMENTUM (float): 动量
                - OPTIMIZER.WEIGHT_DECAY (float): 权重衰减
                - OPTIMIZER.DAMPENING (float): 阻尼
                - OPTIMIZER.NESTEROV (bool): 是否使用 Nesterov 动量
        """
        
        # ---读取配置---
        # 读取优化器的默认参数
        lr = cfg.OPTIMIZER.LR
        momentum = cfg.OPTIMIZER.MOMENTUM
        weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
        dampening = cfg.OPTIMIZER.DAMPENING
        nesterov = cfg.OPTIMIZER.NESTEROV

        # ---检查参数有效性---
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= dampening:
            raise ValueError("Invalid dampening value: {}".format(dampening))
        
        # ---传入优化器的默认参数给父类---
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov  # 是否使用 Nesterov 动量
        )