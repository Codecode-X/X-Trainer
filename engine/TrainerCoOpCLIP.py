from .TrainerCLIP import TrainerClip
from model import build_model
from utils import count_num_param
from torch.cuda.amp import GradScaler
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from utils.metrics import compute_accuracy
from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler
from .build import TRAINER_REGISTRY

@TRAINER_REGISTRY.register()
class TrainerCoOpCLIP(TrainerClip):


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
        assert cfg.MODEL.NAME == "CoOpClip", f"TrainerCoOpClip 只支持 CoOpClip 模型，但 cfg.MODEL.NAME = {cfg.MODEL.NAME}"
        self.CoOp_model = build_model(cfg) # 构建模型 (此处 CLIP 模型提供了预训练模型的载入)
        print("模型参数数量：", count_num_param(self.CoOp_model))

        # 冻结模型某些层 -> 示例：冻结 CLIP 的文本编码器和图像编码器，只训练 CoOp 的 PromptLearner
        if cfg.TRAINER.FROZEN:
            for name, param in self.CoOp_model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad = False

        # 将模型移动到设备
        self.CoOp_model.to(self.device)

        # 设置模型的文本标签，让模型提前提取好每个类别的文本特征
        sorted_labels = sorted(self.lab2cname.items(), key=lambda x: x[0]) # 将文本标签按照 label 从小到大排序，方便模型预测结果与 label 进行对齐
        label_texts = [item[1] for item in sorted_labels]  # 文本标签 tensor | [num_classes]
        print("从小到大排序后的数据集文本标签：", label_texts)
        self.CoOp_model.init_promptLearner(label_texts) # 初始化提示学习器

        # 将模型调整为精度混合训练，以减少显存占用 (如果配置了精度混合训练)
        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None

        # 多 GPU 并行训练情况，则将模型部署到多个 GPU 上 (如果有多个 GPU)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.CoOp_model = nn.DataParallel(self.CoOp_model)

        # 构建 PromptLearner 并注册 -> 示例：优化器只优化 CLIP 的 PromptLearner
        promptLearner = self.CoOp_model.pptLearner
        self.optim = build_optimizer(promptLearner, cfg)
        self.sched = build_lr_scheduler(cfg, self.optim)
        self.register_model("CLIP_promptLearner", promptLearner, self.optim, self.sched)

        return self.CoOp_model, self.optim, self.sched
    
    def forward_backward(self, batch): 
        """
        (实现父类的方法) 前向传播和反向传播。
        """
        image, label = self.parse_batch_train(batch)  # 解析训练批次数据，获取图像和标签
        assert image is not None and label is not None, "forward_backward() 中 parse_batch_train 解析到的图像和标签不能为空"

        prec = self.cfg.TRAINER.PREC  # 配置的精度
        if prec == "amp":  # 自动混合精度训练
            with autocast():
                # Clip 需要传入 图像 和 文本 (初始化模型时已经加载了每个类别的文本特征)。
                # 图像-image: [batch, 3, 224, 224]
                output = self.CoOp_model(image) # 模型预测 -> output: [batch, num_classes]
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:  # 默认 fp16
            output = self.CoOp_model(image) # 模型预测
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