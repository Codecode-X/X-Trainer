import math
import torch
from .OptimizerBase import OptimizerBase
from .build import OPTIMIZER_REGISTRY

@OPTIMIZER_REGISTRY.register()
class RAdam(OptimizerBase):
    """ RAdam 优化器 """
    def __init__(self, cfg, params=None):
        """ 初始化方法 

        参数:
            - cfg (CfgNode): 配置
            - params (iterable): 模型参数

        配置:
            - 优化器默认参数
                - OPTIMIZER.LR (float): 学习率
                - OPTIMIZER.BETAS (Tuple[float, float]): Adam 的 beta 参数
                - OPTIMIZER.EPS (float): 除数中的常数，避免除零错误
                - OPTIMIZER.WEIGHT_DECAY (float): 权重衰减
            - 其他配置
                - OPTIMIZER.DEGENERATED_TO_SGD (bool): 是否将 RAdam 退化为 SGD

        主要步骤:
            - 初始化缓存 self.buffer = [[None, None, None] for _ in range(10)]
            - 读取配置
            - 检查参数有效性
            - 传入优化器的默认参数给父类
        """
        # ----初始化缓存-----
        self.buffer = [[None, None, None] for _ in range(10)]

        # ----读取配置-----

        # 读取优化器的默认参数
        lr = cfg.OPTIMIZER.LR
        betas = cfg.OPTIMIZER.BETAS
        eps = cfg.OPTIMIZER.EPS
        weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY

        # 相关设置
        self.degenerated_to_sgd=cfg.OPTIMIZER.DEGENERATED_TO_SGD # 是否将 RAdam 退化为 SGD

        # ----检查参数有效性-----
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        # ---传入优化器的默认参数给父类---
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) # 优化器的优化选项的默认值
        super(RAdam, self).__init__(params, defaults)


    def step(self, closure=None):
        """ 
        优化器的一步更新方法 

        参数:
            - closure (callable, optional): 一个计算损失的闭包函数

        返回:
            - loss: 计算的 loss 值
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