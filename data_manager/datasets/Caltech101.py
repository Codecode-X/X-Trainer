import os
import random
from .build import DATASET_REGISTRY
from .DatasetClsBase import DatasetClsBase, Datum
from utils import listdir_nohidden


IGNORED = ["BACKGROUND_Google", "Faces_easy"] # 忽略的类别列表

NEW_CNAMES = { # 新类别名称映射
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetClsBase):
    """
    Caltech101 数据集类。
    继承自 DatasetClsBase 类。

    对外访问属性 (@property) :
        - 父类 DatasetClsBase 的属性：
            - 父类 DatasetBase 的属性：
                - train (list): 带标签的训练数据。
                - val (list): 验证数据（可选）。
                - test (list): 测试数据。

            - lab2cname (dict): 标签到类别名称的映射。
            - classnames (list): 类别名称列表。
            - num_classes (int): 类别数量。
            - impath (str): 图像路径。
            - label (int): 类别标签。
            - classname (str): 类别名称。

    对内访问属性：
        - 父类 DatasetClsBase 的属性：
            - 父类 DatasetBase 的属性：
                - num_shots (int): 少样本数量。
                - seed (int): 随机种子。
                - p_trn (float): 训练集比例。
                - p_val (float): 验证集比例。
                - p_tst (float): 测试集比例。
                - dataset_dir (str): 数据集目录。

            - image_dir (str): 图像目录。

    实现 DatasetClsBase 类的抽象方法/接口：
        - read_and_split_data: 读取数据并分割为 train, val, test 数据集 (需要根据每个分类数据集的格式自定义)。

    """

    def read_and_split_data(self):
        """
        读取数据并分割为 train, val, test 数据集（实现父类的抽象方法）。
        
        用到的类属性：
            - p_trn (float): 训练集比例。
            - p_val (float): 验证集比例。
            - p_tst (float): 测试集比例。
            - image_dir (str): 图像目录。
            - IGNORED (List[str]): 忽略的类别列表。
            - NEW_CNAMES (Dict[str, str]): 新类别名称映射。

        返回：
            - 训练、验证和测试数据 (Datum 对象列表类型)。

        主要步骤：
            1. 计算测试集比例。
            2. 获取类别列表。
            3. 遍历类别列表：
                - 获取类别目录下的图像列表。
                - 如果存在新类别名称映射，则更新类别名称。
                - 随机打乱图像列表，并计算训练、验证和测试集数量。
                - 打包数据，得到 (Datum 对象列表类型) 训练、验证和测试数据。
            4. 返回训练、验证和测试数据 (Datum 对象列表类型)。
        """
        print(f"Splitting into {self.p_trn:.0%} train, {self.p_val:.0%} val, and {self.p_tst:.0%} test")  # 打印分割比例
        new_cnames = NEW_CNAMES
        ignored_categories = IGNORED

        # ---获取类别列表---
        categories = listdir_nohidden(self.image_dir)  # 获取类别列表
        categories = [c for c in categories if c not in ignored_categories]  # 过滤忽略的类别
        categories.sort()

        train, val, test = [], [], []

        # ---遍历类别列表---
        for label, category in enumerate(categories):
            
            # ---获取类别目录下的图像列表---
            category_dir = os.path.join(self.image_dir, category)  # 获取类别目录
            images = listdir_nohidden(category_dir)  # 获取图像列表
            images = [os.path.join(category_dir, im) for im in images]  # 拼接图像路径

            # ---如果存在新类别名称映射，则更新类别名称---
            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]  # 更新类别名称
            
            # ---随机打乱图像列表，并计算训练、验证和测试集数量---
            random.shuffle(images)  # 随机打乱图像列表
            n_total = len(images)
            n_train = round(n_total * self.p_trn)  # 计算训练集数量
            n_val = round(n_total * self.p_val)  # 计算验证集数量
            n_test = n_total - n_train - n_val  # 计算测试集数量
            assert n_train > 0 and n_val > 0 and n_test > 0  # 断言训练、验证和测试集数量均大于 0

            # ---打包数据，得到 (Datum 对象列表类型) 训练、验证和测试数据---
            def _collate(imgs, label, classname):
                """ 打包 imgs 数据列表为 Datum 对象列表 """
                return [Datum(impath=imgpath, label=label, classname=classname) for imgpath in imgs]
            train.extend(_collate(images[:n_train], label, category))  # 收集训练数据
            val.extend(_collate(images[n_train : n_train + n_val], label, category))  # 收集验证数据
            test.extend(_collate(images[n_train + n_val :], label, category))  # 收集测试数据
            
        return train, val, test  # 返回训练、验证和测试数据 (Datum 对象列表类型)



