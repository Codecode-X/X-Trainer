from .build import EVALUATOR_REGISTRY

@ EVALUATOR_REGISTRY.register()
class EvaluatorBase:
    """
    接口类 评估器。
    
    子类需要实现以下方法：
        - __init__：初始化评估器。
        - reset：重置评估器状态。
        - process：处理模型输出和真实标签。
        - evaluate：计算评估结果并返回。
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        """重置评估器状态。"""
        raise NotImplementedError

    def process(self, model_output, gt):
        """处理模型输出和真实标签。
        参数：
            model_output (torch.Tensor): 模型输出 [batch, num_classes]
            gt (torch.LongTensor): 真实标签 [batch]
        """
        raise NotImplementedError

    def evaluate(self):
        """计算评估结果并返回。"""
        raise NotImplementedError