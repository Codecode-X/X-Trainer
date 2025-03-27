import os
from .build import DATASET_REGISTRY
import pickle
from utils import mkdir_if_missing

@DATASET_REGISTRY.register()
class DatasetBase:
    """
    数据集类的基类。

    对外访问属性 (@property) : 
        - train (list): 带标签的训练数据。
        - val (list): 验证数据（可选）。
        - test (list): 测试数据。
    
    对内访问属性:
        - num_shots (int): 少样本数量。
        - seed (int): 随机种子。
        - p_trn (float): 训练集比例。
        - p_val (float): 验证集比例。
        - p_tst (float): 测试集比例。
        - dataset_dir (str): 数据集目录。
    
    基本方法:
        - get_data: 读取数据并分割为 train, val, test 数据集。
        - get_fewshot_data: 从 train 和 val 中进行 num_shots 少样本采样，生成少样本的 train 和 val 数据集。
    
    抽象方法/接口：
        - 需要 子基类 根据任务类别实现的：   
            - read_split: 读取数据分割文件。
            - save_split: 保存数据分割文件。
            - generate_fewshot_dataset: 生成小样本数据集。

        - 需要 具体数据集子类 根据数据保存格式 实现的抽象方法/接口:
            - read_and_split_data: 读取并分割数据。

    """

    def __init__(self, cfg):
        """ 
        初始化数据集的基本属性：

        参数:
            - cfg (CfgNode): 配置。

        配置：
            - dataset_dir (str): cfg.DATASET.DATASET_DIR: 数据集目录。
            - num_shots (int): cfg.DATASET.NUM_SHOTS: 少样本数量 | -1 表示使用全部数据，0 表示 zero-shot，1 表示 one-shot，以此类推。
            - seed (int): cfg.SEED: 随机种子。
            - p_trn (float): cfg.DATASET.SPLIT[0]: 训练集比例。
            - p_val (float): cfg.DATASET.SPLIT[1]: 验证集比例。
            - p_tst (float): cfg.DATASET.SPLIT[2]: 测试集比例。

        主要步骤：
            1. 读取配置。
            2. 读取数据并分割为 train, val, test 数据集。（待子类实现 get_data 和 get_fewshot_data 方法）
            3. 如果 num_shots >= 0，则从 train 和 val 中进行少样本采样，生成少样本的 train 和 val 数据集。

        """
        # ---读取配置---
        self.num_shots = cfg.DATASET.NUM_SHOTS  # 获取少样本数量
        self.seed = cfg.SEED  # 获取随机种子
        self.p_trn, self.p_val, self.p_tst = cfg.DATASET.SPLIT  # 获取训练、验证和测试集比例
        assert self.p_trn + self.p_val + self.p_tst == 1  # 断言训练、验证和测试集比例之和为 1
        self.dataset_dir = cfg.DATASET.DATASET_DIR  # 获取数据集目录，例如：/root/autodl-tmp/caltech-101

        # ---读取数据并分割为 train, val, test 数据集---
        self._train, self._val, self._test = self.get_data() 
        
        # ---如果 num_shots >= 0，则从 train 和 val 中进行少样本采样，生成少样本的 train 和 val 数据集---
        if self.num_shots >= 0: # -1: 使用全部数据，0: zero-shot，1: one-shot，以此类推
            self._train, self._val = self.get_fewshot_data(self._train, self._val)

    # -------------------属性-------------------
    @property
    def train(self):
        """返回带标签的训练数据"""
        return self._train

    @property
    def val(self):
        """返回验证数据"""
        return self._val

    @property
    def test(self):
        """返回测试数据"""
        return self._test

    # -------------------方法-------------------
    def get_data(self):
        """
        读取数据并分割为 train, val, test 数据集

        参数：
            - None
        
        返回：
            - 训练、验证和测试数据集。
        """
        
        split_path = os.path.join(self.dataset_dir, "split.json")  # 设置数据分割文件路径

        # 如果数据分割文件已经存在，则直接读取；否则根据 p_trn、p_val 分割数据并保存分割文件
        if os.path.exists(split_path):
            train, val, test = self.read_split(split_path)  # 读取数据分割
        else:
            train, val, test = self.read_and_split_data()  # 读取并分割数据
            self.save_split(train, val, test, split_path)  # 保存数据分割
        
        return train, val, test  # 返回训练、验证和测试数据集
    
    def get_fewshot_data(self, train, val):
        """
        从 train 和 val 中进行 num_shots 少样本采样，生成少样本的 train 和 val 数据集
        
        返回：
            - 少样本训练和验证数据集。
        """
        # 设置少样本数据集目录路径（如果不存在则创建）和 少样本数据集文件保存路径
        split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")  # 设置少样本分割目录路径
        mkdir_if_missing(split_fewshot_dir)  # 如果目录不存在则创建
        fewshot_dataset = os.path.join(split_fewshot_dir, f"shot_{self.num_shots}-seed_{self.seed}.pkl")  # 随机采样到的少样本数据集文件保存路径

        # 如果少样本数据集文件已经存在，则直接加载；否则生成少样本数据集并保存少样本数据集文件
        if os.path.exists(fewshot_dataset): # 如果少样本数据集文件已经存在，则直接加载
            print(f"Loading preprocessed few-shot data from {fewshot_dataset}")
            with open(fewshot_dataset, "rb") as file:
                data = pickle.load(file)  # 加载数据集
                train, val = data["train"], data["val"]  # 获取训练和验证数据
        
        else: # 如果少样本数据集文件不存在，则生成少样本数据集，并保存 (generate_fewshot_dataset 方法需要子基类实现)
            train = self.generate_fewshot_dataset(train, num_shots=self.num_shots)  # 生成少样本训练数据
            val = self.generate_fewshot_dataset(val, num_shots=min(self.num_shots, 4))  # 生成少样本验证数据
            data = {"train": train, "val": val}  # 创建数据字典
            print(f"Saving preprocessed few-shot data to {fewshot_dataset}")
            with open(fewshot_dataset, "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)  # 保存
        
        return train, val  # 返回少样本训练和验证数据集
    

    # -------------------子基类实现 - 抽象方法/接口-------------------

    def read_split(self, split_file):
        """
        读取数据分割文件 (需要子基类类实现的抽象方法)
        子基类根据任务的不同 (分类、回归) 进行定制。
        
        参数：
            - split_file (str): 数据分割文件路径。
        
        返回：
            - 训练、验证和测试数据集。
        """
        raise NotImplementedError
    

    def save_split(self, train, val, test, split_file):
        """
        保存数据分割文件 (需要子基类实现的抽象方法)
        子基类根据任务的不同 (分类、回归) 进行定制。
        
        参数：
            - train (list): 训练数据集。
            - val (list): 验证数据集。
            - test (list): 测试数据集。
            - split_file (str): 数据分割文件路径。
        
        返回：
            - None
        """
        raise NotImplementedError
    
    def generate_fewshot_dataset(self, dataset, num_shots=-1, repeat=False):
        """
        从数据集中随机采样少样本数据集。(需要子类实现的抽象方法)

        参数:
            - dataset (list): 数据集。
            - num_shots (int): 少样本数量。
            - repeat (bool): 是否允许重复采样。

        返回:
            - 如果 num_shots 小于 0，直接返回原始数据源。
            - 如果 num_shots 为 0，直接返回空列表。
            - 如果 num_shots 大于 0，raise NotImplementedError。
        """
        # 如果 num_shots 小于 0，直接返回原始数据源
        if num_shots < 0:  # 不进行少样本采样
            return dataset
        
        # 如果 num_shots 为 0，直接返回空列表
        if num_shots == 0:  
            print(f"正在创建一个 zero-shot 数据集....")
            return [] # zero-shot learning
        
        raise NotImplementedError # 需要子类实现具体的少样本采样逻辑
    

    # -------------------具体数据集子类实现 - 抽象方法/接口-------------------

    def read_and_split_data(self):
        """
        读取并分割数据。(需要子类实现的抽象方法)
        """
        
        raise NotImplementedError