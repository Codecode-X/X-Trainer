from .build import OPTIMIZER_REGISTRY
from torch.optim import Adam as TorchAdam


@OPTIMIZER_REGISTRY.register()
class Adam(TorchAdam):
    """ Adam 优化器 """
    def __init__(self, cfg, params=None):
        """
        初始化 Adam 优化器

        参数:
            - cfg (CfgNode): 配置
            - params (iterable): 模型参数

        配置:
            - 优化器默认参数
                - OPTIMIZER.LR (float): 学习率
                - OPTIMIZER.betas (Tuple[float, float]): Adam 的 beta 参数
                - OPTIMIZER.eps (float): 除数中的常数，避免除零错误
                - OPTIMIZER.weight_decay (float): 权重衰减
            - 其他配置
                - OPTIMIZER.amsgrad (bool): 是否使用 AMSGrad

        主要步骤:
            - 读取配置
            - 检查参数有效性
            - 传入优化器的默认参数给父类

        """
        
        # ---读取配置---
        # 读取优化器的默认参数
        lr = float(cfg.OPTIMIZER.LR)
        betas = tuple(map(float, cfg.OPTIMIZER.betas))
        eps = float(cfg.OPTIMIZER.eps)
        weight_decay = float(cfg.OPTIMIZER.weight_decay)

        # 相关设置
        amsgrad = cfg.OPTIMIZER.get("amsgrad", False) # 是否使用 AMSGrad

        # ---检查参数有效性---
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        # ---传入优化器的默认参数给父类---
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=amsgrad  # 是否使用 AMSGrad
        )
