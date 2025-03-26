import math
import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):
    """ RAdam 优化器 """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        """ 初始化方法 
        参数:
            params (iterable): 模型参数
            lr (float): 学习率
            betas (Tuple[float, float]): 用于计算梯度和平方梯度的移动平均值的系数
            eps (float): 用于数值稳定性的小值
            weight_decay (float): 权重衰减
            degenerated_to_sgd (bool): 是否将 RAdam 退化为 SGD
        """
        # 检查输入数值有效性
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd # 是否将 RAdam 退化为 SGD
        self.buffer = [[None, None, None] for _ in range(10)] # 缓存
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) # 优化器的优化选项的默认值
        # 调用父类的初始化方法，params：指定需要优化的张量；defaults：优化器的默认参数，用于在某个参数组没有明确指定这些选项时提供默认设置
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        """ 将 state 设置为优化器的状态 (例如优化器参数等) """
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        """ 优化器的一步更新方法 
        参数:
            closure (callable, optional): 一个计算损失的闭包函数
        """
        loss = None
        if closure is not None:
            loss = closure()  # 计算损失

        # 遍历参数组（基础层参数 和 新层参数）| 在一些复杂的训练场景中，还可能划分为更多的参数组
        for group in self.param_groups:
            for p in group["params"]: # 遍历组中的参数
                
                if p.grad is None:
                    continue # 不处理没有梯度的参数
                
                grad = p.grad.data.float()  # 获取梯度并转换为浮点型
                if grad.is_sparse: # 不支持稀疏梯度
                    raise RuntimeError("RAdam does not support sparse gradients")  

                # 读取参数的 数据 和 状态
                p_data_fp32 = p.data.float()  # 获取参数的数据（神经网络的权重或偏置）
                state = self.state[p]  # 获取参数的状态（优化器维护的辅助信息，用于参数更新的辅助计算）

                # 初始化参数的状态
                if len(state) == 0:
                    state["step"] = 0  # 初始化 step 计数器
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)  # 初始化一阶矩估计
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)  # 初始化二阶矩估计
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)  # 转换一阶矩估计的数据类型
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)  # 转换二阶矩估计的数据类型

                # 读取优化器的参数
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # 更新一阶矩估计和二阶矩估计
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)  # 更新二阶矩估计
                exp_avg.mul_(beta1).add_(1 - beta1, grad)  # 更新一阶矩估计

                # 更新 step 计数器
                state["step"] += 1  

                # 读取缓存
                buffered = self.buffer[int(state["step"] % 10)]  # 获取缓存
                if state["step"] == buffered[0]: # 如果当前步数等于缓存中的步数
                    N_sma, step_size = buffered[1], buffered[2]  # 从缓存中直接读取 N_sma 和 step_size
                else: # 如果当前步数不等于缓存中的步数，重新计算 N_sma 和 step_size
                    buffered[0] = state["step"]
                    beta2_t = beta2**state["step"]
                    
                    # 计算 N_sma
                    N_sma_max = 2 / (1-beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1-beta2_t)
                    buffered[1] = N_sma
                    
                    # 计算 step_size
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1-beta2_t) * (N_sma-4) / (N_sma_max-4) *
                            (N_sma-2) / N_sma * N_sma_max / (N_sma_max-2)
                        ) / (1 - beta1**state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1**state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size
                
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)  # 权重衰减
                    denom = exp_avg_sq.sqrt().add_(group["eps"])  # 计算分母
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)  # 更新参数
                    p.data.copy_(p_data_fp32)  # 将更新后的参数复制回原参数
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)  # 权重衰减
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)  # 更新参数
                    p.data.copy_(p_data_fp32)  # 将更新后的参数复制回原参数

        return loss  # 返回损失值


class PlainRAdam(Optimizer):
    # 初始化函数，定义优化器的参数
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        # 检查学习率是否有效
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        # 检查 epsilon 值是否有效
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        # 检查 beta 参数是否有效
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    # 执行一步优化
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "RAdam does not support sparse gradients"
                    )

                p_data_fp32 = p.data.float()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(
                        p_data_fp32
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                beta2_t = beta2**state["step"]
                N_sma_max = 2 / (1-beta2) - 1
                N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1-beta2_t)

                # 更保守，因为这是一个近似值
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    step_size = (
                        group["lr"] * math.sqrt(
                            (1-beta2_t) * (N_sma-4) / (N_sma_max-4) *
                            (N_sma-2) / N_sma * N_sma_max / (N_sma_max-2)
                        ) / (1 - beta1**state["step"])
                    )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    step_size = group["lr"] / (1 - beta1**state["step"])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup=0
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup=warmup
        )
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(
                        p_data_fp32
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                bias_correction1 = 1 - beta1**state["step"]
                bias_correction2 = 1 - beta2**state["step"]

                if group["warmup"] > state["step"]:
                    scheduled_lr = 1e-8 + state["step"] * group["lr"] / group[
                        "warmup"]
                else:
                    scheduled_lr = group["lr"]

                step_size = (
                    scheduled_lr * math.sqrt(bias_correction2) /
                    bias_correction1
                )

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        -group["weight_decay"] * scheduled_lr, p_data_fp32
                    )

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss
