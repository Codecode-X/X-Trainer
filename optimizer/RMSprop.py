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
                - OPTIMIZER.ALPHA (float): 移动平均系数
                - OPTIMIZER.EPS (float): 除数中的常数，避免除零错误
                - OPTIMIZER.WEIGHT_DECAY (float): 权重衰减
                - OPTIMIZER.MOMENTUM (float): 动量
                - OPTIMIZER.CENTERED (bool): 是否使用中心化的 RMSprop
        """
        
        # ---读取配置---
        # 读取优化器的默认参数
        lr = cfg.OPTIMIZER.LR
        alpha = cfg.OPTIMIZER.ALPHA
        eps = cfg.OPTIMIZER.EPS
        weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
        momentum = cfg.OPTIMIZER.MOMENTUM
        centered = cfg.OPTIMIZER.CENTERED

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