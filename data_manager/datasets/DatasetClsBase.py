import random
from collections import defaultdict
from .build import DATASET_REGISTRY
from .DatasetBase import DatasetBase
from utils import read_json, write_json
import os
from utils import check_isfile

@DATASET_REGISTRY.register()
class DatasetClsBase(DatasetBase):
    """
    分类数据集类的基类。
    继承自 DatasetBase 类。

    对外访问属性 (@property) :
        - 父类 DatasetBase 的属性:
            - train (list): 带标签的训练数据。
            - val (list): 验证数据（可选）。
            - test (list): 测试数据。

        - lab2cname (dict): 标签到类别名称的映射。
        - classnames (list): 类别名称列表。
        - num_classes (int): 类别数量。

    对内访问属性:
        - 父类 DatasetBase 的属性:
            - num_shots (int): 少样本数量。
            - seed (int): 随机种子。
            - p_trn (float): 训练集比例。
            - p_val (float): 验证集比例。
            - p_tst (float): 测试集比例。
            - dataset_dir (str): 数据集目录。

        - image_dir (str): 图像目录。

    基本方法:
        - 实现 DatasetBase 类的抽象方法/接口：
            - read_split: 读取数据分割文件。
            - save_split: 保存数据分割文件。
            - generate_fewshot_dataset: 生成小样本数据集（通常用于训练集）。

    抽象方法/接口 (需要具体数据集子类实现):
        - read_and_split_data: 读取数据并分割为 train, val, test 数据集 (需要根据每个分类数据集的格式自定义)。
    
    """

    def __init__(self, cfg):
        """ 
        初始化数据集的基本属性

        参数:
            - cfg (CfgNode): 配置。

        配置
            - 当前读取：
                - image_dir (str): cfg.DATASET.IMAGE_DIR: 图像目录。

            - 父类读取：
                - dataset_dir (str): cfg.DATASET.DATASET_DIR: 数据集目录。
                - num_shots (int): cfg.DATASET.NUM_SHOTS: 少样本数量 | -1 表示使用全部数据，0 表示 zero-shot，1 表示 one-shot，以此类推。
                - seed (int): cfg.SEED: 随机种子。
                - p_trn (float): cfg.DATASET.SPLIT[0]: 训练集比例。
                - p_val (float): cfg.DATASET.SPLIT[1]: 验证集比例。
                - p_tst (float): cfg.DATASET.SPLIT[2]: 测试集比例。
        
        主要步骤：
            1. 读取新增配置。
            2. 调用父类 DatasetBase 构造方法：
                1. 读取配置。
                2. 读取数据并分割为 train, val, test 数据集。（待子类实现 get_data 和 get_fewshot_data 方法）
                3. 如果 num_shots >= 0，则从 train 和 val 中进行少样本采样，生成少样本的 train 和 val 数据集。
            3. 获取新增属性：类别数量、标签到类别名称的映射、类别名称列表。
        """

        # ---读取配置---
        self.image_dir = cfg.DATASET.IMAGE_DIR  # 获取图像目录，例如：/root/autodl-tmp/caltech-101/101_ObjectCategories
        
        # ---调用父类构造方法，获取 self.train, self.val, self.test 等属性---
        super().__init__(cfg)  # 调用父类构造方法
        
        # ---获取属性：类别数量、标签到类别名称的映射、类别名称列表---
        self._num_classes = _get_num_classes(self.train)
        self._lab2cname, self._classnames = _get_lab2cname(self.val)


    # -------------------属性-------------------
        
    @property
    def lab2cname(self):
        """返回标签到类别名称的映射"""
        return self._lab2cname

    @property
    def classnames(self):
        """返回类别名称列表"""
        return self._classnames

    @property
    def num_classes(self):
        """返回类别数量"""
        return self._num_classes
    

    # -----------------------父类 DatasetBase 要求子基类实现的抽象方法-----------------------

    def read_split(self, split_file):
        """
        读取数据分割文件 (实现父类的抽象方法)。
        
        参数：
            - img_path_prefix (str): 图像路径前缀，通常是图像所在的目录。
        
        返回：
            - 训练、验证和测试数据 (Datum 对象列表类型)。
        """
        img_path_prefix = self.image_dir  # 图像路径前缀
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(img_path_prefix, impath)  # 拼接图像路径
                item = Datum(impath=impath, label=int(label), classname=classname)  # 创建 Datum 对象
                out.append(item)
            return out

        print(f"Reading split from {split_file}")  # 打印读取分割文件的信息
        split = read_json(split_file)  # 读取 JSON 文件
        train = _convert(split["train"])  # 转换训练数据
        val = _convert(split["val"])  # 转换验证数据
        test = _convert(split["test"])  # 转换测试数据

        return train, val, test  # 返回训练、验证和测试数据 (Datum 对象列表类型)
    

    def save_split(self, train, val, test, split_file):
        """
        保存数据分割文件 (实现父类的抽象方法)。
        
        参数：
            - train (list): 训练数据集。
            - val (list): 验证数据集。
            - test (list): 测试数据集。
            - split_file (str): 数据分割文件路径。

        返回：
            - None
        """
        img_path_prefix = self.image_dir  # 图像路径前缀
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(img_path_prefix, "")  # 去除路径前缀
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out
        train = _extract(train)  # 提取训练数据
        val = _extract(val)  # 提取验证数据
        test = _extract(test)  # 提取测试数据

        split = {"train": train, "val": val, "test": test}  # 创建分割字典

        write_json(split, split_file)  # 写入 JSON 文件
        print(f"Saved split to {split_file}")  # 打印保存分割文件的信息
    

    def generate_fewshot_dataset(self, dataset, num_shots=-1, repeat=False):
        """生成小样本数据集，每个类别仅包含少量图像 (实现父类的抽象方法)。

        参数:
            - dataset (list): 包含 Datum 对象的列表。
            - num_shots (int): 每个类别采样的实例数量。| 默认 -1 即直接返回原始数据源
            - repeat (bool): 是否在需要时重复图像（默认：False）。
        
        返回:
            - fewshot_dataset (list): 包含少量图像的数据集。
        """
    
        # 非 few-shot 学习的情况，父类方法完成了实现
        if num_shots <= 0:
            super().generate_fewshot_dataset(dataset, num_shots, repeat)

        print(f"正在创建一个 {num_shots}-shot 数据集....")

        # 按 标签 整理数据集
        tracker = defaultdict(list)
        for item in dataset:
            tracker[item.label].append(item)

        # 每个 类别 随机采样 num_shots 个样本
        fewshot_dataset = []  # 少样本数据集
        for label, items in tracker.items():
            # 如果样本数量足够，随机采样 num_shots 个样本
            if len(items) >= num_shots:
                sampled_items = random.sample(items, num_shots) # 每个 类别 随机采样 num_shots 个样本 (num_shots)
            else:
                # 如果样本不足，根据 repeat 参数决定是否重复采样
                if repeat:
                    sampled_items = random.choices(items, k=num_shots)
                else:
                    sampled_items = items # 如果不重复采样，直接使用所有样本
            # 将采样的样本加入数据集
            fewshot_dataset.extend(sampled_items) # 包含 当前数据源 的 所有类别的 num_shots 个样本 (类别数*num_shots)

        return fewshot_dataset

    
    # -------------------具体数据集子类实现 - 抽象方法/接口-------------------

    def read_and_split_data(self):
        """
        读取数据并分割为 train, val, test 数据集 (需要根据每个分类数据集的格式自定义)

        需要返回：
            - 训练、验证和测试数据 (Datum 对象列表类型)。
        """
        raise NotImplementedError





# -------------------辅助类 和 函数-------------------

def _get_num_classes(dataset):
    """统计类别数量。在 init 实现

    参数:
        - data_source (list): 包含 Datum 对象的列表。
    
    返回:
        - num_classes (int): 类别数量。
    """
    # 使用集合存储唯一的标签
    label_set = set()
    for item in dataset:
        # 将每个数据实例的标签加入集合
        label_set.add(item.label)
    # 返回最大标签值加 1 作为类别数量
    return max(label_set) + 1

def _get_lab2cname(dataset):
    """
    获取标签到类别名称的映射（字典）。

    参数:
        - data_source (list): 包含 Datum 对象的列表。
    
    返回:
        - mapping (dict): 标签到类别名称的映射。| {label: classname}
        - lassnames (list): 类别名称列表。| [classname1, classname2, ...]
    """
    # 获取数据集中所有的 类别标签和类别名称 的映射关系 mapping
    container = {(item.label, item.classname) for item in dataset}
    mapping = {label: classname for label, classname in container}
    # 获取所有标签并排序
    labels = list(mapping.keys())
    labels.sort()
    # 根据排序后的标签生成类别名称列表
    classnames = [mapping[label] for label in labels]
    return mapping, classnames

class Datum:
    """数据实例类，定义了基本属性。

    参数:
        impath (str): 图像路径。
        label (int): 类别标签。
        classname (str): 类别名称。
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        """初始化数据实例。"""
        # 确保图像路径是字符串类型
        assert isinstance(impath, str)
        # 检查图像路径是否是有效文件
        assert check_isfile(impath)

        # 初始化图像路径
        self._impath = impath
        # 初始化类别标签
        self._label = label
        # 初始化类别名称
        self._classname = classname

    @property
    def impath(self):
        """返回图像路径"""
        return self._impath
    @property
    def label(self):
        """返回类别标签"""
        return self._label
    @property
    def classname(self):
        """返回类别名称"""
        return self._classname

