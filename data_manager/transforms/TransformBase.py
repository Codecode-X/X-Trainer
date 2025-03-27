from data_manager.transforms import TRANSFORM_REGISTRY

@TRANSFORM_REGISTRY.register()
class TransformBase:
    """
    数据变换的基类
    此处只实现了恒等变换，即不对数据进行任何变换
    可以通过继承这个类来实现自定义的数据变换

    子类需要实现以下方法：
        - __init__()：初始化方法
        - __call__()：对数据进行变换的方法
    """

    def __init__(self, cfg):
        self.transform = lambda x: x # 恒等变换

    def __call__(self, img):
        return self.transform(img)