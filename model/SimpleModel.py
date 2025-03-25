import torch.nn as nn
from .build import MODEL_REGISTRY
from .ModelBase import ModelBase

@MODEL_REGISTRY.register()
class SimpleNet(ModelBase):
    """一个简单的神经网络，由一个 CNN 骨干网络 和一个 可选的头部（如用于分类的 MLP）组成。"""

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        self.num_classes = num_classes
        super().__init__()
        # 构建骨干网络
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # 全局最大池化
            nn.AdaptiveMaxPool2d((1, 1)), # 形状变为 [batch, 128, 1, 1]
            # 输出 10 类别数的全连接层
            nn.Flatten(), # 形状变为 [batch, 128]
            nn.Linear(128, num_classes) # 输出 [batch, num_classes]
        )

    def forward(self, x, return_feature=False):
        """前向传播。"""
        # 前向传播，通过骨干网络
        y = self.net(x)
        # 返回输出
        return y
    
    def build_model(self):
        """构建模型。"""
        super().build_model() # 直接调用父类的方法