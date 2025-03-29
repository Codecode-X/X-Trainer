from .build import OPTIMIZER_REGISTRY
from torch.optim import RMSprop as TorchRMSprop

@OPTIMIZER_REGISTRY.register()
class RMSprop(TorchRMSprop):
    """ RMSprop 优化器 """
    def __init__(self, cfg, params=None):
        """
        初始化 RMSprop 优化器

        参数:
            - cfg (CfgNode): 配置
            - params (iterable): 模型参数

        配置:
            - 优化器默认参数
                - OPTIMIZER.LR (float): 学习率
                - OPTIMIZER.alpha (float): 移动平均系数
                - OPTIMIZER.eps (float): 除数中的常数，避免除零错误
                - OPTIMIZER.weight_decay (float): 权重衰减
                - OPTIMIZER.momentum (float): 动量
                - OPTIMIZER.centered (bool): 是否使用中心化的 RMSprop
        """
        
        # ---读取配置---
        # 读取优化器的默认参数
        lr = float(cfg.OPTIMIZER.LR)
        alpha = float(cfg.OPTIMIZER.alpha)
        eps = float(cfg.OPTIMIZER.eps)
        weight_decay = float(cfg.OPTIMIZER.weight_decay)
        momentum = float(cfg.OPTIMIZER.momentum)
        centered = bool(cfg.OPTIMIZER.centered)

        # ---检查参数有效性---
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        
        # ---传入优化器的默认参数给父类---
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=alpha,
            eps=eps,
            centered=centered  # 是否使用中心化的 RMSprop
        )