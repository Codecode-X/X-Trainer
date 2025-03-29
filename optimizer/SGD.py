from .build import OPTIMIZER_REGISTRY
from torch.optim import SGD as TorchSGD

@OPTIMIZER_REGISTRY.register()
class SGD(TorchSGD):
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
                - OPTIMIZER.momentum (float): 动量
                - OPTIMIZER.weight_decay (float): 权重衰减
                - OPTIMIZER.dampening (float): 阻尼
                - OPTIMIZER.nesterov (bool): 是否使用 Nesterov 动量
        """
        
        # ---读取配置---
        # 读取优化器的默认参数
        lr = float(cfg.OPTIMIZER.LR)
        momentum = float(cfg.OPTIMIZER.momentum)
        weight_decay = float(cfg.OPTIMIZER.weight_decay)
        dampening = float(cfg.OPTIMIZER.dampening)
        nesterov = bool(cfg.OPTIMIZER.nesterov)

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