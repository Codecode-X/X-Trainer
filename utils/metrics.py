import torch
from torch.nn import functional as F
import numpy as np

__all__ = [
    "compute_distance_matrix",  # 计算距离矩阵的函数
    "compute_accuracy",  # 计算准确率的函数
    "compute_ci95"  # 计算 95% 置信区间的函数
]

def compute_distance_matrix(input1, input2, metric):
    """计算距离矩阵的函数。

    每个输入矩阵的形状为 (n_data, feature_dim)。

    参数:
        input1 (torch.Tensor): 2-D 特征矩阵。
        input2 (torch.Tensor): 2-D 特征矩阵。
        metric (str, optional): "euclidean" 或 "cosine"。

    返回:
        torch.Tensor: 距离矩阵。
    """
    # 检查输入
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, "期望 2-D 张量，但得到 {}-D".format(input1.dim())
    assert input2.dim() == 2, "期望 2-D 张量，但得到 {}-D".format(input2.dim())
    assert input1.size(1) == input2.size(1)

    if metric == "euclidean":
        distmat = _euclidean_squared_distance(input1, input2) # 计算欧氏距离
    elif metric == "cosine":
        distmat = _cosine_distance(input1, input2) # 计算余弦距离
    else:
        raise ValueError(
            "未知的距离度量：{}。"
            '请选择 "euclidean" 或 "cosine"'.format(metric)
        )

    return distmat


def compute_accuracy(output, target, topks=(1, )):
    """计算指定 k 值的前 k 个预测的准确率 (Top K)。

    参数:
        output (torch.Tensor): 预测矩阵，形状为 (batch_size, num_classes)。
        target (torch.LongTensor): 真实标签，形状为 (batch_size)。
        topks (tuple, optional): 将计算 top-k 的准确率。例如，topk=(1, 5) 表示将计算 top-1 和 top-5 的准确率。

    返回:
        list: top-k 的准确率。
    """
    maxk = max(topks)  # 获取 topk 中的最大值
    batch_size = target.size(0)  # 获取批量大小

    if isinstance(output, (tuple, list)):  # 如果输出是元组或列表
        output = output[0]  # 取第一个元素，通常是 (batch_size, num_classes) 的预测矩阵

    _, pred = output.topk(maxk, 1, True, True)  # 按第 1 维（即列方向）获取前 maxk 个置信度最高的预测结果
    pred = pred.t()  # 转置预测结果
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # pred 和 target 比较，得到一个布尔值矩阵

    res = []  # 初始化结果列表
    for k in topks:  # 遍历 topk 中的每个值
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)  # 计算 top-k 的正确预测数
        acc = correct_k.mul_(100.0 / batch_size)  # 计算准确率
        res.append(acc)  # 将准确率添加到结果列表中

    return res  # 返回 TopK 准确率列表


# ------辅助函数------

def _euclidean_squared_distance(input1, input2):
    """计算欧氏平方距离，即 L2 范数的平方。

    参数:
        input1 (torch.Tensor): 2-D 特征矩阵。
        input2 (torch.Tensor): 2-D 特征矩阵。

    返回:
        torch.Tensor: 距离矩阵。
    """
    m, n = input1.size(0), input2.size(0)  # 获取输入矩阵的行数
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)  # 计算 input1 的平方
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()  # 计算 input2 的平方
    distmat = mat1 + mat2  # 计算距离矩阵的初始值
    distmat.addmm_(1, -2, input1, input2.t())  # 计算最终的欧氏距离矩阵
    return distmat


def _cosine_distance(input1, input2):
    """计算余弦距离。
    （余弦相似度越小，余弦距离越大）

    参数:
        input1 (torch.Tensor): 2-D 特征矩阵。
        input2 (torch.Tensor): 2-D 特征矩阵。

    返回:
        torch.Tensor: 距离矩阵。
    """
    input1_normed = F.normalize(input1, p=2, dim=1)  # 对 input1 进行归一化
    input2_normed = F.normalize(input2, p=2, dim=1)  # 对 input2 进行归一化
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())  # 计算余弦距离矩阵
    return distmat

def compute_ci95(results):
    """ 
    计算 95% 置信区间。

    参数：
        - res (list): 包含多个数值的列表。
    返回：
        - float: 95% 置信区间。
    """
    return 1.96 * np.std(results) / np.sqrt(len(results))