import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from .build import EVALUATOR_REGISTRY
from .EvaluatorBase import EvaluatorBase

@EVALUATOR_REGISTRY.register()
class EvaluatorClassification(EvaluatorBase):
    """分类任务的评估器。"""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        """ 初始化分类评估器。
        参数:
            cfg (CfgNode): 配置。
            lab2cname (dict): 标签到类名的映射。
        
        属性:
            lab2cname (dict): 标签到类名的映射。
            correct (int): 正确预测的数量。
            total (int): 总数量。
            y_true (list): 真实标签。
            y_pred (list): 预测标签。
            per_class (bool): 是否评估每个类别的结果。
            per_class_res (dict): 每个类别的结果。
            calc_cmat (bool): 是否计算混淆矩阵。
        """
        super().__init__(cfg)
        self.lab2cname = lab2cname # 标签到类名的映射
        self.correct = 0 # 正确预测的数量
        self.total = 0 # 总数量
        
        self.y_true = [] # 真实标签
        self.y_pred = [] # 预测标签

        self.per_class = cfg.EVALUATOR.per_class # 是否评估每个类别的结果
        self.per_class_res = None # 每个类别的结果
        if self.per_class: # 是否评估每个类别的结果
            assert lab2cname is not None
            self.per_class_res = defaultdict(list) # 用于记录每个类别的结果的字典
        
        self.calc_cmat = cfg.EVALUATOR.calc_cmat # 是否计算混淆矩阵

    def reset(self):
        """(实现父类的方法) 重置评估器状态。"""
        self.correct = 0
        self.total = 0
        self.y_true = []
        self.y_pred = []
        if self.per_class_res is not None:
            self.per_class_res = defaultdict(list)

    def process(self, model_output, gt):
        """(实现父类的方法) 处理模型输出和真实标签。
        参数：
            model_output (torch.Tensor): 模型输出 [batch, num_classes]
            gt (torch.LongTensor): 真实标签 [batch]
        """
        pred = model_output.max(1)[1]  # 获取每个样本的预测类别
        matches = pred.eq(gt).float()  # 计算预测是否正确
        self.correct += int(matches.sum().item())  # 累加正确预测的数量
        self.total += gt.shape[0]  # 累加总样本数量

        self.y_true.extend(gt.data.cpu().numpy().tolist())  # 记录真实标签
        self.y_pred.extend(pred.data.cpu().numpy().tolist())  # 记录预测标签

        if self.per_class_res is not None:  # 如果需要记录每个类别的结果
            for i, label in enumerate(gt):  # 遍历每个样本
                label = label.item()  # 获取标签值
                matches_i = int(matches[i].item())  # 获取该样本的匹配结果
                self.per_class_res[label].append(matches_i)  # 记录该类别的匹配结果

    def evaluate(self):
        """(实现父类的方法) 计算评估结果并返回。"""
        results = OrderedDict()  # 用于存储评估结果的有序字典
        
        # 整体的评估结果
        acc = 100.0 * self.correct / self.total  # 计算准确率
        err = 100.0 - acc  # 计算错误率
        macro_f1 = 100.0 * f1_score( # 计算宏平均 F1 分数
            self.y_true,
            self.y_pred,
            average="macro",
            labels=np.unique(self.y_true)
        )  
        results["accuracy"] = acc  # 存储准确率
        results["error_rate"] = err  # 存储错误率
        results["macro_f1"] = macro_f1  # 存储宏平均 F1 分数
        print( # 打印评估结果
            "=> result\n"
            f"* total: {self.total:,}\n"
            f"* correct: {self.correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )  
        
        # 每个类别的结果
        if self.per_class_res is not None: 
            labels = list(self.per_class_res.keys())  # 获取所有类别标签
            labels.sort()  # 对标签进行排序
            print("=> per-class result")
            accs = []  # 用于存储每个类别的准确率
            for label in labels:
                classname = self.lab2cname[label]  # 获取类别名称
                res = self.per_class_res[label]  # 获取该类别的匹配结果
                correct = sum(res)  # 计算该类别的正确预测数量
                total = len(res)  # 计算该类别的总样本数量
                acc = 100.0 * correct / total  # 计算该类别的准确率
                accs.append(acc)  # 将准确率添加到列表中
                print( # 打印该类别的评估结果
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )  
            mean_acc = np.mean(accs)  # 计算平均准确率
            print(f"* average: {mean_acc:.1f}%")
            results["perclass_accuracy"] = mean_acc  # 存储每个类别的平均准确率

        # 混淆矩阵结果
        if self.calc_cmat:
            cmat = confusion_matrix(self.y_true, self.y_pred, normalize="true")  # 计算混淆矩阵
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")  # 混淆矩阵保存路径
            torch.save(cmat, save_path)  # 保存混淆矩阵
            print(f"Confusion matrix is saved to {save_path}")  # 打印混淆矩阵保存路径

        return results  # 返回评估结果
