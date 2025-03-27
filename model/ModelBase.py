import torch.nn as nn


class ModelBase(nn.Module):
    """
    接口类 模型。
    继承自 torch.nn.Module，提供模型通用结构。

    子类需要实现以下方法：
        - __init__()：初始化方法
        - forward()：前向传播
        - (可选) build_model()：构建模型（例如：加载预训练模型）
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, x, return_feature=False):
        """
        前向传播。
        参数：
            x (torch.Tensor): 输入数据 [batch, ...]
            return_feature (bool): 是否返回特征
        返回：
            torch.Tensor: 输出结果 [batch, num_classes]
            (可选) torch.Tensor: 特征 [batch, ...]
        """
        raise NotImplementedError
    
    def build_model(self):
        """构建模型。(可选)"""
        pass