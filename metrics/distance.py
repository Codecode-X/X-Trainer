"""
Source: https://github.com/KaiyangZhou/deep-person-reid
"""
import torch
from torch.nn import functional as F


def compute_distance_matrix(input1, input2, metric="euclidean"):
    """计算距离矩阵的包装函数。

    每个输入矩阵的形状为 (n_data, feature_dim)。

    参数:
        input1 (torch.Tensor): 2-D 特征矩阵。
        input2 (torch.Tensor): 2-D 特征矩阵。
        metric (str, optional): "euclidean" 或 "cosine"。
            默认是 "euclidean"。

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
        distmat = euclidean_squared_distance(input1, input2) # 计算欧氏距离
    elif metric == "cosine":
        distmat = cosine_distance(input1, input2) # 计算余弦距离
    else:
        raise ValueError(
            "未知的距离度量：{}。"
            '请选择 "euclidean" 或 "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
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


def cosine_distance(input1, input2):
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
