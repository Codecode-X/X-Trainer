from .build import EVALUATOR_REGISTRY

@ EVALUATOR_REGISTRY.register()
class EvaluatorBase:
    """接口类 评估器。"""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        """重置评估器状态。(需要子类实现)"""
        raise NotImplementedError

    def process(self, model_output, gt):
        """处理模型输出和真实标签。(需要子类实现)
        参数：
            model_output (torch.Tensor): 模型输出 [batch, num_classes]
            gt (torch.LongTensor): 真实标签 [batch]
        """
        raise NotImplementedError

    def evaluate(self):
        """计算评估结果并返回。(需要子类实现)"""
        raise NotImplementedError