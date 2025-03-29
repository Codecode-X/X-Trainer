"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from .tools import mkdir_if_missing

__all__ = [
    "save_checkpoint", # 保存检查点
    "load_checkpoint", # 加载检查点 
    "resume_from_checkpoint", # 从检查点恢复训练
    "open_all_layers", # 打开模型中的所有层进行训练
    "open_specified_layers", # 打开模型中指定的层进行训练
    "count_num_param", # 计算模型中的参数数量
    "load_pretrained_weights", # 加载预训练权重到模型
    "init_network_weights", # 初始化网络权重
    "transform_image", # 对图像应用 K 次 tfm 增强 并返回结果
    "standard_image_transform" # 图像预处理转换管道
    "patch_jit_model" # 修正 JIT 模型的设备和 dtype 信息
]

def save_checkpoint(state, save_dir, is_best=False,
                    remove_module_from_keys=True, model_name="" ):
    r"""保存检查点。

    参数:
        state (dict): 字典，包含模型状态。
        save_dir (str): 保存检查点的目录。
        is_best (bool, optional): 如果为True，这个检查点会被复制并命名为
            ``model-best.pth.tar``。默认值为False。
        remove_module_from_keys (bool, optional): 是否从层名称中移除"module."。
            默认值为True。
        model_name (str, optional): 保存的模型名称。
    """
    mkdir_if_missing(save_dir) # 创建保存目录

    if remove_module_from_keys:
        # 从 state_dict 的键中移除'module.'
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # 保存模型
    epoch = state["epoch"]
    if not model_name:
        model_name = "model.pth.tar-" + str(epoch)
    fpath = osp.join(save_dir, model_name)
    torch.save(state, fpath)
    print(f"检查点已保存到 {fpath}")

    # 保存当前模型名称
    checkpoint_file = osp.join(save_dir, "checkpoint")
    checkpoint = open(checkpoint_file, "w+")
    checkpoint.write("{}\n".format(osp.basename(fpath)))
    checkpoint.close()

    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
        shutil.copy(fpath, best_fpath)
        print('最佳检查点已保存到 "{}"'.format(best_fpath))

def load_checkpoint(fpath):
    r"""加载检查点。

    可以很好地处理``UnicodeDecodeError``，这意味着
    python2保存的文件可以从python3读取。

    参数:
        fpath (str): 检查点路径。

    返回:
        dict

    示例::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("文件路径为 None")

    if not osp.exists(fpath):
        raise FileNotFoundError('文件未找到 "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('无法从 "{}" 加载检查点'.format(fpath))
        raise

    return checkpoint

def resume_from_checkpoint(fdir, model, optimizer=None, scheduler=None):
    r"""从检查点恢复训练。

    这将加载 (1) 模型权重 和 (2) 优化器的``state_dict``（如果``optimizer``不为None）。

    参数:
        fdir (str): 保存模型的目录。
        model (nn.Module): 模型。
        optimizer (Optimizer, optional): 优化器。
        scheduler (Scheduler, optional): 调度器。

    返回:
        int: start_epoch。

    示例::
        >>> fdir = 'log/my_model'
        >>> start_epoch = resume_from_checkpoint(fdir, model, optimizer, scheduler)
    """
    with open(osp.join(fdir, "checkpoint"), "r") as checkpoint:
        model_name = checkpoint.readlines()[0].strip("\n")
        fpath = osp.join(fdir, model_name)

    print('从 "{}" 加载检查点'.format(fpath))
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint["state_dict"])
    print("已加载模型权重")

    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("已加载优化器")

    if scheduler is not None and "scheduler" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("已加载调度器")

    start_epoch = checkpoint["epoch"]
    print("上一个 epoch: {}".format(start_epoch))

    return start_epoch

def open_all_layers(model):
    r"""打开模型中的所有层进行训练。

    示例::
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True

def open_specified_layers(model, open_layers):
    r"""打开模型中指定的层进行训练，同时保持其他层冻结。

    参数:
        model (nn.Module): 神经网络模型。
        open_layers (str or list): 打开训练的层。

    示例::
        >>> # 只有model.classifier会被更新。
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # 只有model.fc和model.classifier会被更新。
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel): # 如果模型是 nn.DataParallel
        model = model.module # 获取模型

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    # 检查是否存在指定的层
    for layer in open_layers:
        assert hasattr(model, layer), f"{layer} 不是一个属性"

    # 遍历模型的的所有子模块
    for name, module in model.named_children(): 
        if name in open_layers: # 打开模型中指定的层进行训练
            module.train() 
            for p in module.parameters():
                p.requires_grad = True
        else: # 其他层冻结
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

def count_num_param(model=None, params=None):
    r"""计算模型中的参数数量。

    参数:
        model (nn.Module): 神经网络模型。
        params: 神经网络模型的参数。

    示例::
        >>> model_size = count_num_param(model)
    """
    if model is not None:
        return sum(p.numel() for p in model.parameters())

    if params is not None:
        s = 0
        for p in params:
            if isinstance(p, dict):
                s += p["params"].numel()
            else:
                s += p.numel()
        return s

    raise ValueError("model 和 params 必须至少提供一个。")

def load_pretrained_weights(model, weight_path):
    r"""加载预训练权重到模型。

    特性::
        - 不兼容的层（名称或大小不匹配）将被忽略。
        - 可以自动处理包含"module."的键。

    参数:
        model (nn.Module): 神经网络模型。
        weight_path (str): 预训练权重的路径。

    示例::
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict() # 获取模型当前的 state_dict
    new_state_dict = OrderedDict() # 存储匹配的层的预训练权重的 state_dict
    matched_layers, discarded_layers = [], [] # 匹配的层，丢弃的层

    # 遍历预训练权重的 state_dict，记录匹配和不匹配的层
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # 丢弃 module.

        if k in model_dict and model_dict[k].size() == v.size(): # 如果匹配（键名和大小匹配）
            new_state_dict[k] = v # 存储匹配的层的预训练权重
            matched_layers.append(k) # 记录匹配的层
        else: # 如果不匹配
            discarded_layers.append(k) # 记录丢弃的层
     
    # 更新模型的 state_dict
    model_dict.update(new_state_dict) # 将匹配的层的预训练权重更新到模型的 state_dict
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0: # 如果完全没有匹配的层
        warnings.warn(
            f"无法加载 {weight_path} (请手动检查键名)"
        )
    else: # 打印没有匹配的层
        print(f"成功从 {weight_path} 加载预训练权重")
        if len(discarded_layers) > 0:
            print(
                f"由于键名或大小不匹配而丢弃的层：{discarded_layers}"
            )

def init_network_weights(model, init_type="normal", gain=0.02):
    """初始化网络权重。
    参数:
        model (nn.Module): 神经网络模型。
        init_type (str): 初始化类型。可选值包括：
            - normal: 标准正态分布
            - xavier: Xavier 初始化
            - kaiming: Kaiming 初始化
            - orthogonal: 正交初始化
        gain (float): 缩放因子。
    """
    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("InstanceNorm") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)


def transform_image(tfm_func, img0, K=1):
    """
    对图像应用 K 次 tfm 增强 并返回得到的 K 个不同的增强结果（注意不是叠加 K 次）。

    参数：
    - tfm_func (callable): transform 函数。
    - img0 (PIL.Image): 原始图像。
    - K (int): 增强次数，生成 K 个增强结果。

    返回：
    - 增强后的单个图像 (如果只有一个增强结果) img_list[0] || 增强后的图像列表 img_list
    """
    img_list = []  # 初始化图像列表

    for k in range(K):  # 进行 K 次重复增强
        tfm_img = tfm_func(img0) # 对原始图像应用 transform
        img_list.append(tfm_img)

    # 如果进行了多次增强，则返回增强后的图像列表；否则，返回增强后的单个图像
    return img_list[0] if len(img_list) == 1 else img_list  


def standard_image_transform(input_size, interp_mode):
    """
    标准图像预处理转换管道。
    
    参数:
        - input_size (int): 输入图像的大小。
        - interp_mode (str): 插值模式 | 可选值包括: NEAREST, BILINEAR, BICUBIC
    返回:
        - Compose: 组合的图像转换。

    主要步骤：
        - 保持图像的宽高比，调整图像大小到 input_size，并使用指定的插值模式
        - 中心裁剪(CenterCrop)图像到 input_size*input_size
        - 转换为 RGB 图像
    """
    assert isinstance(input_size, int), "input_size 必须是整数"
    
    interp_mode = getattr(InterpolationMode, interp_mode.upper(), InterpolationMode.BILINEAR) # 获取插值模式
    
    def _image_to_rgb(image):
        return image.convert("RGB")
    
    return Compose([
        Resize(input_size, interpolation=interp_mode),
        CenterCrop(input_size),
        _image_to_rgb
    ])


def patch_jit_model(model, device="cuda"):
    """
    修正 JIT 模型的设备和 dtype 信息。
    该函数用于修正 JIT 模型的设备和数据类型信息，以确保模型在指定设备上运行。
    
    主要步骤：
        1. 生成目标设备的节点。
        2. 遍历模型的计算图，修正设备信息。
        3. 如果目标设备是 CPU，则修正数据类型为 float32。
        4. 返回修正后的模型。
    
    参数:
        - model (torch.jit.ScriptModule): 经过 torch.jit.trace 或 torch.jit.script 编译的模型
        - device (str): 目标设备 ("cuda" 或 "cpu")
    
    返回:
        - torch.jit.ScriptModule: 设备和 dtype 修正后的 JIT 模型
    """
    
    # ----生成目标设备的节点----
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """获取 JIT 计算图中的属性值"""
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        """修正 JIT 计算图中的设备信息"""
        graphs = []
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)  # 处理 forward1 变体

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    # ----对模型及其子模块应用设备修正----
    model.apply(patch_device)
    
    # × 只适用于clip，修改为下面 "递归遍历 JIT 模型的所有子模块"
    # patch_device(model.encode_image)
    # patch_device(model.encode_text)
    
    # √ 递归遍历 JIT 模型的所有子模块
    for name, submodule in model.named_modules():
        patch_device(submodule)

    # ----如果是 CPU，需要修正 dtype 为 float32----
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            """修正 JIT 计算图中的 dtype（仅限 CPU）"""
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []
                
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype 可能是 aten::to() 的第二个或第三个参数
                        if _node_get(inputs[i].node(), "value") == 5:  # 5 代表 float32
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        # 强制整个模型的 dtype 变为 float32
        model.float()

    return model
