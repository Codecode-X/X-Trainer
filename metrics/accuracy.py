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
