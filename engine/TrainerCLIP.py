from TrainerBase import TrainerBase
from model import build_model
from utils import count_num_param
from optim import build_optimizer, build_lr_scheduler
from torch.cuda.amp import GradScaler
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from metrics import compute_accuracy
from utils import load_checkpoint
from optim import build_optimizer, build_lr_scheduler
import os.path as osp

class TrainerCLIP(TrainerBase):

    def check_cfg(self, cfg): # 检查配置文件中的 PREC 字段是否为合法值
        """ (实现父类的方法) 检查配置文件中的 PREC 字段是否为合法值。"""
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def init_model(self, cfg):
        """
        (实现父类的方法) 初始化模型。
        -> 只训练图像编码器示例

        参数：
            cfg (CfgNode): 配置。

        返回：
            model (nn.Module): 模型。
            optim (Optimizer): 优化器。
            sched (LRScheduler): 学习率调度器。

        主要步骤：
        1. 构建模型
        2. 冻结模型的文本编码器，只训练图像编码器
        3. 将模型移动到设备
        4. 将模型调整为精度混合训练
        5. 多 GPU 并行训练情况，则将模型部署到多个 GPU 上
        6. 构建优化器和调度器，只优化图像编码器，并注册
        7. 返回模型、优化器和调度器
        """
        # 构建模型
        self.model = build_model(cfg) # 构建模型 (此处 CLIP 模型提供了预训练模型的载入)
        print("模型参数数量：", count_num_param(self.model))

        # 冻结模型某些层 -> 示例：冻结 CLIP 的文本编码器，只训练图像编码器
        if cfg.TRAINER.COOP.FROZEN_LAYERS:
            for name, param in self.model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False

        # 将模型移动到设备
        self.model.to(self.device)

        # 将模型调整为精度混合训练，以减少显存占用 (如果配置了精度混合训练)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # 多 GPU 并行训练情况，则将模型部署到多个 GPU 上 (如果有多个 GPU)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        # 构建优化器和调度器并注册 -> 示例：优化器只优化 CLIP 的图像编码器
        image_encoder = self.model.visual
        self.optim = build_optimizer(image_encoder, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP_image_encoder", image_encoder, self.optim, self.sched)

        return self.model, self.optim, self.sched
    
    def forward_backward(self, batch): 
        """
        (实现父类的方法) 前向传播和反向传播。
        """

        image, label = self.parse_batch_train(batch)  # 解析训练批次数据，获取图像和标签
        
        prec = self.cfg.TRAINER.COOP.PREC  # 配置的精度
        if prec == "amp":  # 自动混合精度训练
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:  # 默认 fp16
            output = self.model(image) # 模型预测
            loss = F.cross_entropy(output, label)  # 计算损失
            self.model_backward_and_update(loss)  # 反向传播

        # 需要记录的 loss 日志
        loss_summary = {  
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        # 到阶段自动更新学习率
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary