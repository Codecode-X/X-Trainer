import os.path as osp
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import (tolist_if_not, load_checkpoint, save_checkpoint, resume_from_checkpoint)
import time
import numpy as np
import os.path as osp
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from data import DataManager
from torch.cuda.amp import GradScaler
from optim import build_optimizer
from lr_scheduler import build_lr_scheduler
from model import build_model
from utils import (count_num_param, mkdir_if_missing, load_pretrained_weights)
from evaluation import build_evaluator
from utils import (MetricMeter, AverageMeter)


class TrainerBase:
    """
    迭代训练器的基类。
    
    包含的方法：

    -------工具方法-------
    * init_writer: 初始化 TensorBoard。
    * close_writer: 关闭 TensorBoard。
    * write_scalar: 写入标量值到 TensorBoard。

    * register_model: 注册模型、优化器和学习率调度器。
    * get_model_names: 获取所有已注册的模型名称。

    * save_model: 保存模型，包括模型状态、epoch、优化器状态、学习率调度器状态、验证结果。
    * load_model: 加载模型，包括模型状态、epoch、验证结果。
    * resume_model_if_exist: 如果存在检查点，则恢复模型，包括模型状态、优化器状态、学习率调度器状态。

    * set_model_mode: 设置模型的模式 (train/eval)。

    * model_backward_and_update: 模型反向传播和更新，包括清零梯度、反向传播、更新模型参数。
    * update_lr: 调用学习率调度器的 step() 方法，更新 names 模型列表中的模型的学习率。
    * get_current_lr: 获取当前学习率。 

    * train: 通用训练循环，但里面包含的子方法 (before_train、after_train、before_epoch、
             after_epoch、run_epoch(必实现)) 需由子类实现。
    
    -------子类可重写的方法（可选）-------
    * check_cfg: 检查配置中的某些变量是否正确设置。 (未实现)

    * before_train: 训练前的操作。
    * after_train: 训练后的操作。
    * before_epoch: 每个 epoch 前的操作。 (未实现) 
    * after_epoch: 每个 epoch 后的操作。
    * run_epoch: 执行每个 epoch 的训练。
    * test: 测试方法。
    * parse_batch_train: 解析训练批次。
    * parse_batch_test: 解析测试批次。
    * model_inference: 模型推理。

    -------需要子类重写的方法（必选）-------
    * init_model: 初始化模型，如冻结模型的某些层，加载预训练权重等。 (未实现 - 冻结模型某些层)
    * forward_backward: 前向传播和反向传播。
    """

    def __init__(self, cfg):
        """
        初始化训练器。
        主要包括：
        * 初始化相关属性，读取配置信息
        * 构建数据加载器
        * 构建并注册模型，优化器，学习率调度器；并初始化模型
        * 构建评估器
        """
        # 检查输入合法性
        assert isinstance(cfg, dict)  # 确保配置是字典

        # 检查配置中的某些变量是否正确设置（可选）
        self.check_cfg(cfg)  

        # 初始化相关属性
        self._models = OrderedDict()  # 存储模型的有序字典
        self._optims = OrderedDict()  # 存储优化器的有序字典
        self._scheds = OrderedDict()  # 存储学习率调度器的有序字典
        self._writer = None  # TensorBoard 的 SummaryWriter
        self.best_result = -np.inf  # 初始化最佳结果
        self.start_epoch = self.epoch = 0
        # 读取配置信息
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg
        # 设置设备（GPU 或 CPU）
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 构建数据加载器
        print("构建数据加载器...")
        dm = DataManager(self.cfg) # 通过配置创建数据管理器
        self.dm = dm  # 保存数据管理器

        self.train_loader_x = dm.train_loader # 有标签训练数据加载器
        
        self.val_loader = dm.val_loader  # 验证数据加载器 (可选，可以为 None
        self.test_loader = dm.test_loader # 测试数据加载器

        self.num_classes = dm.num_classes # 类别数
        self.lab2cname = dm.lab2cname  # 类别名称字典 {label: classname}

        # 构建并注册模型，优化器，学习率调度器；并初始化模型
        print("构建模型，优化器，学习率调度器...")
        self.model, self.optim, self.sched = self.init_model(cfg)

        # 构建评估器
        print("构建评估器...")
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)  # 构建评估器
        
        

    def init_writer(self, log_dir):
        """
        工具方法：初始化 TensorBoard。
        
        参数：
            * log_dir: 日志目录

        返回：
            * None
        """
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)  # 初始化 TensorBoard

    def close_writer(self):
        """工具方法：关闭 TensorBoard。"""
        if self._writer is not None:
            self._writer.close()  # 关闭 TensorBoard

    def write_scalar(self, tag, scalar_value, global_step=None):
        """
        工具方法：写入标量值到 TensorBoard。
        
        参数：
            * tag: 标签
            * scalar_value: 标量值
            * global_step: 全局步数
            
        返回：
            * None
        """
        if self._writer is None:
            pass # 如果 writer 未初始化，则不执行任何操作
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)  # 写入标量值
    

    def register_model(self, name="model", model=None, optim=None, sched=None):
        """
        工具方法：注册模型、优化器和学习率调度器。
        self._models[name] = model  # 注册模型
        self._optims[name] = optim  # 注册优化器
        self._scheds[name] = sched  # 注册学习率调度器

        参数：
            * name: 模型名称
            * model: 模型
            * optim: 优化器
            * sched: 学习率调度器

        返回：
            * None
        """
        
        # 确保在调用 super().__init__() 之后才能注册模型
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )
        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )
        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )
        assert name not in self._models, "Found duplicate model names"  # 确保模型名称不重复

        # 注册
        self._models[name] = model  # 注册模型
        self._optims[name] = optim  # 注册优化器
        self._scheds[name] = sched  # 注册学习率调度器

    def get_model_names(self, names=None):
        """
        工具方法：获取所有已注册的模型名称。
        self._models.keys()
        """
        names_real = list(self._models.keys())  # 获取所有模型名称
        if names is not None:
            names = tolist_if_not(names)  # 如果 names 不是列表，将其转换为列表
            for name in names:
                assert name in names_real  # 确保名称在已注册的模型名称中
            return names
        else:
            return names_real


    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        """
        工具方法：保存模型。
        参数：
            * epoch: 当前 epoch
            * directory: 保存目录
            * is_best: 是否是最佳模型
            * val_result: 验证结果
            * model_name: 模型名称

        返回：
            * None

        保存内容：
            * 模型状态字典
            * epoch
            * 优化器状态字典
            * 学习率调度器状态字典
            * 验证结果
        """
        names = self.get_model_names()  # 获取所有模型名称
        for name in names:
            model_dict = self._models[name].state_dict()  # 获取模型状态字典

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()  # 获取优化器状态字典

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()  # 获取学习率调度器状态字典

            # 保存 checkpoint
            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def load_model(self, directory, epoch=None):
        """
        工具方法：将 dictionary 中的模型文件载入到模型中。
        
        参数：
            * directory: 模型目录
            * epoch: epoch
        
        返回：
            * None

        加载内容：
            * 模型状态字典
            * epoch
            * 验证结果
        """
        if not directory: # 如果目录不存在，直接返回
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()  # 获取所有模型名称

        model_file = "model-best.pth.tar" # 默认情况下，加载最佳模型
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch) # 如果指定 epoch，加载指定 epoch 的模型

        # 遍历所有模型名称，加载模型
        for name in names: 
            model_path = osp.join(directory, name, model_file) # 模型路径

            if not osp.exists(model_path): # 如果模型路径不存在，抛出异常
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path) # 加载检查点
            state_dict = checkpoint["state_dict"] # 获取状态字典
            epoch = checkpoint["epoch"] # 获取 epoch
            val_result = checkpoint["val_result"] # 获取验证结果
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict) # 加载模型状态字典


    def resume_model_if_exist(self, directory):
        """
        工具方法：如果存在检查点，则恢复模型。
        
        参数：
            * directory: 检查点目录
            
        返回：
            * start_epoch: 开始的 epoch

        恢复内容：
            * 模型状态字典
            * 优化器状态字典
            * 学习率调度器状态字典
        """
        names = self.get_model_names()  # 获取所有模型名称
        
        # 遍历所有模型名称，检查是否存在检查点文件
        file_missing = False
        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True # 文件缺失
                break
        if file_missing: # 如果文件缺失，返回 0
            print("No checkpoint found, train from scratch")
            return 0
        print(f"Found checkpoint at {directory} (will resume training)")

        # 恢复模型
        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint( # 从检查点恢复
                path, self._models[name], self._optims[name], # 恢复模型、优化器
                self._scheds[name] # 恢复学习率调度器
            )
        return start_epoch # 返回开始的 epoch

    
    def set_model_mode(self, mode="train", names=None):
        """
        工具方法：设置模型的模式 (train/eval)。
        如果 names 为 None，则设置所有模型的模式。

        参数：
            * mode: 模式
            * names: 需要设置的模型名称列表
        """
        names = self.get_model_names(names)  # 获取所有模型名称
        
        # 遍历所有模型名称，设置模型模式
        for name in names:
            if mode == "train":
                self._models[name].train()  # 设置模型为训练模式
            elif mode in ["test", "eval"]:
                self._models[name].eval()  # 设置模型为评估模式
            else:
                raise KeyError



    def model_backward_and_update(self, loss, names=None):
        """
        工具方法：模型反向传播和更新。
        
        参数：
            * loss: 损失
            * names: 需要更新的模型名称列表
            
        返回：
            * None
            
        流程：
            1. 清零梯度 (遍历需要更新的模型名称列表，调用优化器的 zero_grad() 方法 清零梯度)
            2. 反向传播 (检查 loss 是否为有限值，如果不是有限值，抛出异常)
            3. 更新模型参数 (调用优化器的 step() 方法)
        """
        # ------清零梯度------
        names = self.get_model_names(names)  # 获取所有模型名称
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()  # 清零梯度

        # ------反向传播------
        # 检测损失是否为有限值
        if not torch.isfinite(loss).all(): # 如果损失不是有限，抛出异常
            raise FloatingPointError("Loss is infinite or NaN!") 
        loss.backward()  # 反向传播
        
        # ------更新模型参数------
        names = self.get_model_names(names)  # 获取所有模型名称
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()  # 更新模型参数


    def update_lr(self, names=None):
        """
        工具方法：调用学习率调度器的 step() 方法，更新 names 模型列表中的模型的学习率。
        
        参数：
            * names: 需要更新学习率的模型名称列表
        返回：
            * None

        方式：
            * 调用学习率调度器的 step() 方法   
        """
        names = self.get_model_names(names)  # 获取所有模型名称

        # 遍历所有模型名称，更新学习率
        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()  # 更新学习率

    
    def get_current_lr(self, names=None):
        """
        工具方法：获取当前学习率。
        
        参数：
            * names: 需要获取学习率的模型名称列表
            
        返回：
            * 模型名称列表中第一个模型的学习率
        """
        names = self.get_model_names(names)
        name = names[0] # 只获取第一个模型的学习率
        return self._optims[name].param_groups[0]["lr"]

    
    def train(self, start_epoch, max_epoch):
        """
        工具方法：通用训练循环。
            
        流程：
        1. 执行训练前的操作 before_train()
        2. 开始训练
            * 执行每个 epoch 前的操作 before_epoch()
            * 执行每个 epoch 的训练 run_epoch()
            * 执行每个 epoch 后的操作 after_epoch()
        3. 执行训练后的操作 after_train()
        """
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        # 执行训练前的操作
        self.before_train()
        
        # 开始训练
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch() # 执行每个 epoch 前的操作
            self.run_epoch() # 执行每个 epoch 的训练
            self.after_epoch() # 执行每个 epoch 后的操作
        
        # 执行训练后的操作
        self.after_train() 


    def check_cfg(self, cfg):
        """
        检查配置中的某些变量是否正确设置（可选子类实现）。   

        例如，一个训练器可能需要特定的采样器进行训练，如 'RandomDomainSampler'，
        因此可以进行检查：assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'

        未实现
        """
        pass

    def before_train(self):
        """
        训练前的操作。 (可选子类实现)
        
        主要包括：
        * 设置输出目录
        * 如果输出目录存在检查点，则恢复检查点
        * 初始化 summary writer
        * 记录开始时间（用于计算经过的时间）
        """
        # 设置输出目录
        if self.cfg.RESUME: # 如果配置了 RESUME
            directory = self.cfg.RESUME # 恢复 RESUME 目录
        else: # 否则按照配置的输出目录
            directory = self.cfg.OUTPUT_DIR
        
        # 如果存在检查点，则恢复模型
        self.start_epoch = self.resume_model_if_exist(directory)  

        # 初始化 summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir) # 创建日志目录
        self.init_writer(writer_dir) # 初始化 writer

        # 记录开始时间（用于计算经过的时间）
        self.time_start = time.time()

    def after_train(self):
        """
        训练后的操作。 (可选子类实现)
        
        主要包括：
        * 如果训练后需要测试，则测试，并保存最佳模型；否则保存最后一个 epoch 的模型
        * 打印经过的时间
        * 关闭 writer
        """
        
        print("训练结束")
        
        # 如果训练后需要测试，则测试，并保存最佳模型；否则保存最后一个 epoch 的模型
        do_test = not self.cfg.TEST.NO_TEST 
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("测试验证性能最好的模型")
                self.load_model(self.output_dir)
            else:
                print("测试最后一个 epoch 的模型")
            self.test()

        # 打印经过的时间
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"经过时间：{elapsed}")

        # 关闭 writer
        self.close_writer()

    def before_epoch(self):
        """
        每个 epoch 前的操作。 (可选子类实现)  
        未实现
        """
        pass

    def after_epoch(self):
        """
        每个 epoch 后的操作。 (可选子类实现)
        
        主要包括：
        * 判断模型保存条件：是否是最后一个 epoch、是否需要验证、是否满足保存检查点的频率
        * 根据条件保存模型
        """
        # 判断模型保存条件：是否是最后一个 epoch、是否需要验证、是否满足保存检查点的频率
        is_last_epoch = (self.epoch + 1) == self.max_epoch # 是否是最后一个 epoch
        need_eval = self.cfg.TRAIN.DO_EVAL # 是否需要验证（每个 epoch 后都验证，并保存验证性能最好的模型）
        meet_checkpoint_freq = (  # 是否满足保存检查点的频率
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # 根据条件保存模型
        if need_eval:
            # 如果每个 epoch 后都验证，则进行验证，并保存验证性能最好的模型
            curr_result = self.test(split="val")  # TODO：是否有输出验证结果
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
        if meet_checkpoint_freq or is_last_epoch:
            # 如果满足保存检查点的频率或是最后一个 epoch，则保存模型
            self.save_model(self.epoch, self.output_dir)

    def run_epoch(self):
        """
        执行每个 epoch 的训练。 (子类可重写)
        此处采用标准有标签数据的训练模式

        主要包括：
        * 设置模型为训练模式
        * 初始化度量器：损失度量器、批次时间度量器、数据加载时间度量器
        * 开始迭代
            — 遍历有标签数据集 train_loader_x
            — 前向和反向传播，获取损失
            — 打印日志 (epoch、batch、时间、数据加载时间、损失、学习率、剩余时间)
        """
        
        # 设置模型为训练模式
        self.set_model_mode("train")  

        # 初始化度量器
        losses = MetricMeter()  # 初始化损失度量器
        batch_time = AverageMeter()  # 初始化批次时间度量器
        data_time = AverageMeter()  # 初始化数据加载时间度量器

        # 开始迭代
        self.num_batches = len(self.train_loader_x)  # 获取有标签数据集的批次数
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x): # 遍历有标签数据集 train_loader_x
            data_time.update(time.time() - end)  # 更新数据加载时间度量器，记录一个批次的数据加载时间
            
            # 前向和反向传播，获取损失
            loss_summary = self.forward_backward(batch)  
            
            # 打印日志
            batch_time.update(time.time() - end)  # 更新批次时间度量器
            losses.update(loss_summary)  # 更新损失度量器

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0  # 是否满足打印频率
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ  # 是否只有少量批次
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1  # 计算当前 epoch 剩余批次数
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches  # 计算后续 epoch 批次数
                eta_seconds = batch_time.avg * nb_remain  # 计算剩余时间
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                # 日志信息：epoch、batch、时间、数据加载时间、损失、学习率、剩余时间
                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))
            
            n_iter = self.epoch * self.num_batches + self.batch_idx  # 当前迭代次数
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)  # 记录损失到 TensorBoard
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)  # 记录学习率到 TensorBoard
            end = time.time()

    @torch.no_grad()
    def test(self, split=None):
        """
        测试。 (子类可重写)

        主要包括：
        * 设置模型模式为 eval, 重置评估器
        * 确定测试集 (val or test, 默认为测试集)
        * 开始测试
            - 遍历数据加载器
            - 解析测试批次，获取输入和标签 - self.parse_batch_test(batch)
            - 模型推理 - self.model_inference(input)
            - 评估器评估模型输出和标签 - self.evaluator.process(output, label)
        * 使用 evaluator 对结果进行评估，并将结果记录在 tensorboard
        * 返回结果 (此处为 accuracy)
        """
        
        self.set_model_mode("eval")
        self.evaluator.reset() # 重置评估器

        # 确定测试集（val or test，默认为测试集）
        if split is None: # 如果 split 为 None，则使用配置中的测试集
            split = self.cfg.TEST.SPLIT 
        if split == "val" and self.val_loader is not None: 
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        print(f"在 *{split}* 集上测试")

        # 开始测试
        for batch_idx, batch in enumerate(tqdm(data_loader)): # 遍历数据加载器
            input, label = self.parse_batch_test(batch) # 解析测试批次，获取输入和标签
            output = self.model_inference(input) # 模型推理
            self.evaluator.process(output, label) # 评估器评估模型输出和标签

        # 使用 evaluator 对结果进行评估，并将结果记录在 tensorboard
        results = self.evaluator.evaluate() 
        for k, v in results.items(): 
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0] # 返回第一个值：accuracy

    def parse_batch_train(self, batch):
        """
        解析训练批次。 (子类可重写)
        此处直接从 batch 字典中获取输入图像、类别标签和域标签。
        """
        input = batch["img"]  # 获取图像
        label = batch["label"]  # 获取标签

        input = input.to(self.device)  # 将图像移动到设备
        label = label.to(self.device)  # 将标签移动到设备

        return input, label  # 返回图像、标签


    def parse_batch_test(self, batch):
        """
        解析测试批次。 (子类可重写)   
        此处直接从 batch 字典中获取输入和标签。
        """
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def model_inference(self, input):
        """
        模型推理。 (子类可重写)  
        此处直接调用模型，返回模型输出。
        """
        return self.model(input) # 直接调用模型
    
    def init_model(self, cfg):
        """
        初始化模型（子类需要重写，仅提供示例）。
        主要包括：
        * 构建模型
        * 加载预训练权重
        * 冻结模型某些层
        * 将模型移动到设备
        * 将模型调整为精度混合训练
        * 将模型部署到多个 GPU 上
        * 为整个模型或部分模块构建优化器和学习率调度器，并注册
        
        参数：
            * cfg: 配置

        返回：
            * model: 模型
            * optim: 优化器
            * sched: 学习率调度器
        """
        # 构建模型
        self.model = build_model(cfg) # 构建模型
        print("模型参数数量：", count_num_param(self.model))
        
        # 给模型载入预训练权重 (如果配置了预训练权重)
        if cfg.MODEL.INIT_WEIGHTS_PATH: 
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS_PATH)  # 加载预训练权重

        # 冻结模型某些层 (如果配置了冻结层)
        pass  # 未实现
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 将模型调整为精度混合训练，以减少显存占用 (如果配置了精度混合训练)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # 将模型部署到多个 GPU 上 (如果有多个 GPU)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"检测到多个 GPU (n_gpus={device_count}), 使用所有 GPU!")
            self.model = nn.DataParallel(self.model)

        # 为 整个模型 或 部分模块 (例如 head) 构建优化器和学习率调度器，并注册
        self.optim = build_optimizer(self.model, cfg.OPTIM)  # 构建优化器
        self.sched = build_lr_scheduler(cfg, self.optim)  # 构建学习率调度器
        self.register_model(cfg.MODEL.NAME, self.model, self.optim, self.sched) # 注册模型

        return self.model, self.optim, self.sched

    def forward_backward(self, batch):
        """
        前向传播和反向传播。 (需要子类实现)
        未实现
        """
        raise NotImplementedError