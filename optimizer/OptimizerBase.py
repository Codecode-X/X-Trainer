from torch.optim import Optimizer

class OptimizerBase(Optimizer):
    """
    基类 优化器
    继承自 torch.optim.Optimizer，提供优化器通用结构。

    子类需要实现以下方法：
        - __init__()：初始化方法
        - step()：执行参数更新
    """

    def __init__(self, params, defaults):
        """
        初始化优化器

        参数:
            - params (iterable): 待优化的参数组（parameter groups）列表。
            每个参数组是一个字典，包含一组模型参数及其对应的超参数（如学习率、权重衰减等）。
            
            - defaults (dict): 优化器的默认参数。
            如果某个参数组未显式指定某个超参数，则使用 defaults 中的值。
        
        """
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """
        执行参数更新，子类必须实现具体的 step() 逻辑
        (继承自 torch.optim.Optimizer)

        参数:
            - closure (callable, optional): 可选的闭包函数，用于计算 loss 并反向传播
        
        返回:
            - loss: 计算的 loss 值
        """
        raise NotImplementedError
