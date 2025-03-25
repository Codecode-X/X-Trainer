import torch.nn as nn


class ModelBase(nn.Module):
    """接口类 模型。"""

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