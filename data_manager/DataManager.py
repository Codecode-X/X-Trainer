import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset
from utils import read_image, transform_image
from .datasets import build_dataset
from .samplers import build_train_sampler, build_test_sampler
from .transforms import build_train_transform, build_test_transform
from .transforms import TRANSFORM_REGISTRY
from torchvision.transforms import Normalize, Compose

class DataManager:
    """
    数据管理器，用于加载数据集和构建数据加载器。
    
    参数：
        - cfg (CfgNode): 配置。
        - custom_tfm_train (list): 自定义训练数据增强。
        - custom_tfm_test (list): 自定义测试数据增强。
        - dataset_wrapper (DatasetWrapper): 数据集包装器。

    属性：
        - num_classes (int): 类别数量。
        - lab2cname (dict): 类别到名称的映射。
        - dataset (DatasetBase): 数据集。
        - train_loader (DataLoader): 训练数据加载器。
        - val_loader (DataLoader): 验证数据加载器。
        - test_loader (DataLoader): 测试数据加载器。 

    方法：
        - show_dataset_summary: 打印数据集摘要信息。
        
    """
    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_transform=None):
        """ 
        初始化数据管理器：构建 数据集 和 数据加载器。
        
        参数：
            - cfg (CfgNode): 配置。
            - custom_tfm_train (list): 自定义训练数据增强。
            - custom_tfm_test (list): 自定义测试数据增强。
            - dataset_transform (TransformedDataset): 数据集转换器 | 转换为 tensor, 数据增强等操作。

        主要步骤：
            1. 构建数据集对象。
            2. 构建数据增强。
            3. 构建数据加载器（数据集 + 数据增强）。
            4. 记录属性：类别数量、类别到名称的映射。
            5. 记录对象：数据集、训练数据加载器、验证数据加载器、测试数据加载器。
            6. 如果启用了详细信息打印，打印数据集摘要信息。
        """
        # ---构建数据集对象---
        dataset = build_dataset(cfg)

        # ---构建数据增强---
        if custom_tfm_train is None: # 构建训练数据增强
            tfm_train = build_train_transform(cfg)  # 使用配置默认的训练数据增强
        else:
            print("* 使用自定义训练数据增强")
            tfm_train = custom_tfm_train  # 使用自定义的训练数据增强
        
        if custom_tfm_test is None: # 构建测试数据增强
            tfm_test = build_test_transform(cfg)  # 使用配置默认的测试数据增强
        else:
            print("* 使用自定义测试数据增强")
            tfm_test = custom_tfm_test  # 使用自定义的测试数据增强

        # ---构建数据加载器（数据集 + 采样器 + 数据增强）---
        train_sampler = build_train_sampler(cfg, dataset.train) # 构建训练采样器
        train_loader = _build_data_loader( # 根据配置信息，构建训练数据加载器 train_loader
            cfg,
            sampler=train_sampler,  # 训练采样器
            data_source=dataset.train,  # 数据源
            batch_size=cfg.DATALOADER.BATCH_SIZE_TRAIN,  # 批大小
            tfm=tfm_train,  # 训练数据增强
            is_train=True,  # 训练模式
            dataset_transform=dataset_transform  # 数据集转换器，用于对数据集进行转换和增强
        )
        val_loader = None  
        if dataset.val:  # 构建验证数据加载器 val_loader (如果存在验证数据)
            val_sampler = build_test_sampler(cfg, dataset.val) # 构建验证集采样器
            val_loader = _build_data_loader(
                cfg,
                sampler=val_sampler,  # 验证采样器
                data_source=dataset.val,  # 数据源
                batch_size=cfg.DATALOADER.BATCH_SIZE_TEST,  # 批大小
                tfm=tfm_test,  # 验证数据增强
                is_train=False,  # 测试模式
                dataset_transform=dataset_transform # 数据集转换器，用于对数据集进行转换和增强
            )
        test_sampler = build_test_sampler(cfg, dataset.test) # 构建测试集采样器
        test_loader = _build_data_loader( # 构建测试数据加载器 test_loader
            cfg,
            sampler=test_sampler,  # 测试采样器
            data_source=dataset.test,  # 数据源
            batch_size=cfg.DATALOADER.BATCH_SIZE_TEST,  # 批大小
            tfm=tfm_test,  # 测试数据增强
            is_train=False,  # 测试模式
            dataset_transform=dataset_transform # 数据集转换器，用于对数据集进行转换和增强
        )

        # ---记录属性：类别数量、类别到名称的映射---
        self._num_classes = dataset.num_classes  # 类别数量
        self._lab2cname = dataset.lab2cname  # 类别到名称的映射
        
        # ---记录对象：数据集、训练数据加载器、验证数据加载器、测试数据加载器---
        self.dataset = dataset # 数据集
        self.train_loader = train_loader # 训练数据加载器
        self.val_loader = val_loader # 验证数据加载器
        self.test_loader = test_loader # 测试数据加载器

        if cfg.VERBOSE:  # 如果配置中启用了详细信息打印
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        """返回类别数量。"""
        return self._num_classes

    @property
    def lab2cname(self):
        """返回类别到名称的映射。"""
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        """打印数据集摘要信息。"""
        dataset_name = cfg.DATASET.NAME  # 数据集名称

        # 构建摘要表格
        table = []
        table.append(["数据集", dataset_name])
        table.append(["类别数量", f"{self.num_classes:,}"])
        table.append(["有标签训练数据", f"{len(self.dataset.train):,}"])
        if self.dataset.val:
            table.append(["验证数据", f"{len(self.dataset.val):,}"])
        table.append(["测试数据", f"{len(self.dataset.test):,}"])

        # 打印表格
        print(tabulate(table))


def _build_data_loader(cfg, sampler, data_source=None, batch_size=64, tfm=None, is_train=True, dataset_transform=None):
    """构建数据加载器。
    
    参数：
        - cfg (CfgNode): 配置。
        - sampler (Sampler): 采样器。
        - data_source (list): 数据源。
        - batch_size (int): 批大小。
        - tfm (list): 数据增强。
        - is_train (bool): 是否是训练模式。
        - dataset_transform (TransformeWrapper): 数据转换器 | 转换为 tensor, 数据增强等操作。
    返回：
        - DataLoader: 数据加载器。

    主要步骤：
        1. 通过 torch.utils.data.DataLoader 根据 数据转换器，数据源，批大小，采样器 构建数据加载器。
        2. 断言数据加载器长度大于 0。
    """

    # 数据转换器
    if dataset_transform is None:
        dataset_transform = TransformeWrapper(cfg, data_source, transform=tfm, is_train=is_train)

    # 构建数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset_transform, # 数据转换器（转换为 tensor, 数据增强等操作）
        batch_size=batch_size, 
        sampler=sampler, # 采样器
        num_workers=cfg.DATALOADER.NUM_WORKERS, # 工作进程数
        drop_last=(is_train and len(data_source) >= batch_size), # 只有在 训练模式下 且 数据源的长度大于等于批大小时 才丢弃最后一个批次
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA) # 只有在 CUDA 可用且使用 CUDA 时才将数据存储在固定内存中
    )

    assert len(data_loader) > 0
    return data_loader



class TransformeWrapper(TorchDataset):
    """
    数据集转换包装器，用于对数据集进行 转换 和 增强 操作。
    
    参数：
        - cfg (CfgNode): 配置。
        - data_source (list): 数据源。
        - transform (list): 数据增强。
        - is_train (bool): 是否是训练模式。

    主要功能：
        - 对数据源中的每个数据项应用数据转换 (resize + RGB + toTensor + normalize)。
        - (可选) 对数据源中的每个数据项应用数据增强。
        
    """

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        """ 
        初始化数据集转换包装器。
        
        主要步骤：
            1. 初始化属性，并获取相关配置信息。
            3. 构建一个不应用任何数据增强的预处理管道(resize+RGB+toToTensor+normalize)
        """
        # 初始化属性
        self.data_source = data_source  # 数据源
        self.transform = transform  # 数据增强，接受列表或元组作为输入
        self.is_train = is_train  # 是否是训练模式
        # 获取相关配置信息
        self.cfg = cfg  # 配置
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1 # 增强次数 | 如果是训练模式，获取数据增强次数；测试模式下默认为 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0  # 是否记录未增强的原始图像 | 默认为 False

        # 构建一个不应用任何数据增强的预处理管道(resize+RGB)
        self.no_aug = Compose([TRANSFORM_REGISTRY.get("StandardNoAugTransform")(cfg),  # 标准无增强预处理流程
                                T.ToTensor(),
                                Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD) if cfg.INPUT.NORMALIZE else None])

        # 如果需要对图像进行 K 次增强，但未提供 transform，则抛出异常
        if self.k_tfm > 1 and transform is None:
            raise ValueError("无法对图像进行 {} 次增强，因为 transform 为 None".format(self.k_tfm))


    def __len__(self):
        """返回数据源的长度。"""
        return len(self.data_source)

    def __getitem__(self, idx):
        """根据索引获取数据项。
        
        参数：
        - idx (int): 索引。
        
        返回：字典 output: 
            - label: 类别标签
            - impath: 图像路径
            - index: 索引
            - img or imgi: 增强后的图像 (第 i 个增强) | img: 第一个增强 | img1: 第二个增强 | ...
            - img0: 未增强处理的图像 | 即只经过预处理 (resize+RGB+toToTensor+normalize) 的图像
        """
        item = self.data_source[idx]  # 获取数据项

        # 初始化输出字典
        output = {
            "label": item.label,  # 类别标签
            "impath": item.impath,  # 图像路径
            "index": idx,  # 索引
        }

        img0 = read_image(item.impath)  # 原始图像
        # 如果提供了 transform, 则对图像进行增强；否则，返回无任何处理的原始图像
        if self.transform is not None:  
            self.transform = [self.transform] if not isinstance(self.transform, (list, tuple)) else self.transform # 如果 transform 不是列表或元组，则转为列表
            for i, tfm in enumerate(self.transform):  # 遍历每个 transform
                img = transform_image(tfm, img0, self.k_tfm) # 对原始图像应用 (K 次) tfm 增强
                keyname = f"img{i + 1}" if (i + 1) > 1 else "img"  # 键名为："img", "img1", "img2", ... 
                output[keyname] = img  # 增强后的图像
        else: 
            output["img"] = self.no_aug(img0) # 经过预处理的原始图像

        if self.return_img0:  # 如果需要返回未增强的原始图像
            output["img0"] = self.no_aug(img0) # 经过预处理的原始图像

        return output  # 返回输出字典
