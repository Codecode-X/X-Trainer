from collections import defaultdict
import torch

__all__ = [
    "AverageMeter",  # 计算并存储平均值和当前值
    "MetricMeter" # 存储一组指标值
]


class AverageMeter:
    """计算并存储平均值和当前值。

    示例::
        >>> # 1. 初始化一个记录损失的计量器
        >>> losses = AverageMeter()
        >>> # 2. 在每次小批量更新后更新计量器
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        参数:
            ema (bool, optional): 是否应用指数移动平均（对新数据更敏感，更快反映数据变化）。
        """
        self.ema = ema # 是否使用指数移动平均
        self.reset()

    def reset(self):
        """重置所有值：val(当前损失值), avg, sum, count = 0。"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """更新计量器。
        参数:
            val (float): batch_mean_loss(当前批次计算得到损失均值)。
            n (int): 样本数。
        """
        # 如果 val 是 torch.Tensor 类型，则转换为 Python 标量
        if isinstance(val, torch.Tensor):
            val = val.item()

        # 更新当前值、总和和计数
        self.val = val  
        self.sum += val * n # 当前批次计算得到损失均值 * 当前批次的样本数
        self.count += n

        # 根据是否使用指数移动平均更新平均值
        # 指数移动平均（EMA）是一种加权的移动平均方法，它在计算过程中赋予最近的数据更高的权重，而较早的数据权重逐渐减小。
        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


class MetricMeter:
    """存储一组指标值。

    示例::
        >>> # 1. 创建 MetricMeter 实例
        >>> metric = MetricMeter()
        >>> # 2. 使用字典作为输入进行更新
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. 转换为字符串并打印
        >>> print(str(metric))
    """

    def __init__(self, delimiter=" "):
        """
        参数:
            delimiter (str): 指标之间的分隔符。
        """
        self.meters = defaultdict(AverageMeter) # 用于存储指标的平均值和当前值的字典
        self.delimiter = delimiter # 分隔符

    def update(self, input_dict):
        """ 更新指标值。
        参数:
            input_dict (dict): 包含各个指标的名称和值的字典，用于更新指标。
        """
        # 如果输入字典为空，则返回
        if input_dict is None:
            return

        # 如果输入不是字典类型，则抛出类型错误
        if not isinstance(input_dict, dict):
            raise TypeError("MetricMeter.update() 的输入必须是字典")

        # 更新每个指标的值
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item() # 将 torch.Tensor 转换为 Python 标量
            self.meters[k].update(v) # 更新指标的值

    def __str__(self):
        """将所有指标转换为字符串并连接"""
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(f"{name} {meter.val:.4f} ({meter.avg:.4f})")
        return self.delimiter.join(output_str)
