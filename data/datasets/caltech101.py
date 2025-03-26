import os
import pickle
import random
from data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from utils import read_json, write_json, mkdir_if_missing
from utils import listdir_nohidden, mkdir_if_missing
import math

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):
    """
    Caltech101 数据集类
    
    参数：
        cfg (CfgNode): 配置。

    属性：
        dataset_dir (str): 数据集目录。
        image_dir (str): 图像目录。
        split_path (str): 数据分割文件路径。
        split_fewshot_dir (str): 少样本分割目录路径。

    方法：
        read_split: 读取数据分割文件。
        subsample_classes: 子样本类别处理。
        save_split: 保存数据分割文件。
        read_and_split_data: 读取并分割数据。
    """

    dataset_dir = "caltech-101"  # 数据集目录

    def __init__(self, cfg):
        """
        初始化方法
        
        主要步骤：
        1. 读取数据并分割
        2. 生成少样本数据集
        3. 子样本类别处理
        5. 调用父类构造方法
        """
        
        # 读取数据并分割
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))  # 获取数据集根目录的绝对路径
        self.dataset_dir = os.path.join(root, self.dataset_dir)  # 设置数据集目录路径
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")  # 设置图像目录路径
        self.split_path = os.path.join(self.dataset_dir, "split_Caltech101.json")  # 设置数据分割文件路径
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")  # 设置少样本分割目录路径
        mkdir_if_missing(self.split_fewshot_dir)  # 如果目录不存在则创建

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)  # 读取数据分割
        else:
            train, val, test = self.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)  # 读取并分割数据
            self.save_split(train, val, test, self.split_path, self.image_dir)  # 保存数据分割

        num_shots = cfg.DATASET.NUM_SHOTS  # 获取少样本数量
        if num_shots >= 1:
            seed = cfg.SEED  # 随机种子
            fewshot_dataset = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")  # 随机采样到的少样本数据集文件保存路径
            
            if os.path.exists(fewshot_dataset): # 如果少样本数据集文件已经存在，则直接加载
                print(f"Loading preprocessed few-shot data from {fewshot_dataset}")
                with open(fewshot_dataset, "rb") as file:
                    data = pickle.load(file)  # 加载数据集
                    train, val = data["train"], data["val"]  # 获取训练和验证数据
            else: # 如果少样本数据集文件不存在，则生成少样本数据集，并保存
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)  # 生成少样本训练数据
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))  # 生成少样本验证数据
                data = {"train": train, "val": val}  # 创建数据字典
                print(f"Saving preprocessed few-shot data to {fewshot_dataset}")
                with open(fewshot_dataset, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)  # 保存

        # 调用父类构造方法
        super().__init__(train=train, val=val, test=test)  


    @staticmethod
    def read_split(filepath, path_prefix):
        """读取数据分割文件"""
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)  # 拼接图像路径
                item = Datum(impath=impath, label=int(label), classname=classname)  # 创建 Datum 对象
                out.append(item)
            return out

        print(f"Reading split from {filepath}")  # 打印读取分割文件的信息
        split = read_json(filepath)  # 读取 JSON 文件
        train = _convert(split["train"])  # 转换训练数据
        val = _convert(split["val"])  # 转换验证数据
        test = _convert(split["test"])  # 转换测试数据

        return train, val, test  # 返回训练、验证和测试数据

    
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        """保存数据分割文件"""
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")  # 去除路径前缀
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out
        train = _extract(train)  # 提取训练数据
        val = _extract(val)  # 提取验证数据
        test = _extract(test)  # 提取测试数据

        split = {"train": train, "val": val, "test": test}  # 创建分割字典

        write_json(split, filepath)  # 写入 JSON 文件
        print(f"Saved split to {filepath}")  # 打印保存分割文件的信息

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
        """读取并分割数据"""
        categories = listdir_nohidden(image_dir)  # 获取类别列表
        categories = [c for c in categories if c not in ignored]  # 过滤忽略的类别
        categories.sort()

        p_tst = 1 - p_trn - p_val  # 计算测试集比例
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")  # 打印分割比例

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # 创建 Datum 对象
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)  # 获取类别目录
            images = listdir_nohidden(category_dir)  # 获取图像列表
            images = [os.path.join(category_dir, im) for im in images]  # 拼接图像路径
            random.shuffle(images)  # 随机打乱图像列表
            n_total = len(images)
            n_train = round(n_total * p_trn)  # 计算训练集数量
            n_val = round(n_total * p_val)  # 计算验证集数量
            n_test = n_total - n_train - n_val  # 计算测试集数量
            assert n_train > 0 and n_val > 0 and n_test > 0  # 断言训练、验证和测试集数量均大于 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]  # 更新类别名称

            train.extend(_collate(images[:n_train], label, category))  # 收集训练数据
            val.extend(_collate(images[n_train : n_train + n_val], label, category))  # 收集验证数据
            test.extend(_collate(images[n_train + n_val :], label, category))  # 收集测试数据

        return train, val, test  # 返回训练、验证和测试数据
