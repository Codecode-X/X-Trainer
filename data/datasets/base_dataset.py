import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown

from utils import check_isfile


class DatasetBase:
    """
    数据集类的基类。

    参数:
        train_x (list): 带标签的训练数据。
        val (list): 验证数据（可选）。
        test (list): 测试数据。

    定义了数据集的基本属性:
        - train_x (list): 带标签的训练数据。
        - train_u (list): 无标签的训练数据（可选）。
        - val (list): 验证数据（可选）。
        - test (list): 测试数据。
        - lab2cname (dict): 标签到类别名称的映射。
        - classnames (list): 类别名称列表。
        - num_classes (int): 类别数量。
    
    定义了数据集的基本方法:
        - get_num_classes: 统计类别数量。
        - get_lab2cname: 获取标签到类别名称的映射。
        - download_data: 下载 url 数据并解压保存。
        - generate_fewshot_dataset: 生成小样本数据集。
        - split_dataset_by_label: 按类别标签将数据集分组。

    """
    dataset_dir = ""  # 数据集存储的目录
    domains = []  # 所有域的名称列表

    def __init__(self, train=None, val=None, test=None):
        """ 初始化数据集的基本属性：
        - 带标签的训练数据
        - 验证数据（可选）
        - 测试数据
        - 类别数量
        - 标签到类别名称的映射
        - 类别名称列表
        """
        self._train = train
        self._val = val
        self._test = test
        self._num_classes = self.get_num_classes(train)
        self._lab2cname, self._classnames = self.get_lab2cname(train)

    @property
    def train_x(self):
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

    @staticmethod
    def get_num_classes(dataset):
        """统计类别数量。

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

    @staticmethod
    def get_lab2cname(dataset):
        """
        获取标签到类别名称的映射（字典）。

        参数:
            - data_source (list): 包含 Datum 对象的列表。
        
        返回:
            - mapping (dict): 标签到类别名称的映射。| {label: classname}
            - lassnames (list): 类别名称列表。| [classname1, classname2, ...]
        """
        # 获取数据集中所有的 类别标签和类别名称 的映射关系 mapping
        container = set() # 去重 set
        for item in dataset: 
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        # 获取所有标签并排序
        labels = list(mapping.keys())
        labels.sort()
        # 根据排序后的标签生成类别名称列表
        classnames = [mapping[label] for label in labels]
        return mapping, classnames
    

    def download_data(self, url, dst, from_gdrive=True):
        """
        下载数据并解压，支持 zip, tar, tar.gz 文件，解压后文件存储在目标路径的文件夹中。

        参数:
            - url (str): 数据下载链接。
            - dst (str): 下载文件的目标路径。
            - from_gdrive (bool): 是否从 Google Drive 下载。
        
        返回:
            - None
        """
        # 如果目标路径的父目录不存在，则创建
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            # 使用 gdown 下载文件
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        # 解压 zip 文件
        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        # 解压 tar 文件
        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        # 解压 tar.gz 文件
        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(self, dataset, num_shots=-1, repeat=False):
        """生成小样本数据集（通常用于训练集）。

        此函数用于在小样本学习设置中评估模型，
        每个类别仅包含少量图像。

        参数:
            - dataset (list): 包含 Datum 对象的列表。
            - num_shots (int): 每个类别采样的实例数量。| 默认 -1 即直接返回原始数据源
            - repeat (bool): 是否在需要时重复图像（默认：False）。
        
        返回:
            - fewshot_dataset (list): 包含少量图像的数据集。
        """
        # 如果 num_shots 小于 1，直接返回原始数据源
        if num_shots < 1:
            return dataset

        print(f"正在创建一个 {num_shots}-shot 数据集....")

        # 按标签分割数据集
        tracker = self.split_dataset_by_label(dataset) # 将数据集（Datum 对象列表）按类别标签分组存储在字典中。
        fewshot_dataset = [] 

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

    def split_dataset_by_label(self, dataset):
        """按 类别标签 将数据集（Datum 对象列表）分组存储在字典中。

        参数:
            - dataset (list): 包含 Datum 对象的列表。
        
        返回:
            - output (dict): 按类别标签分组的数据集。
        """
        output = defaultdict(list)
        for item in dataset:
            # 根据标签将数据实例分组
            output[item.label].append(item)
        return output



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