"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
from difflib import SequenceMatcher
import PIL
import torch
from PIL import Image

__all__ = [
    "mkdir_if_missing",  # 如果缺少目录则创建
    "check_isfile",  # 检查是否是文件
    "read_json",  # 读取 JSON 文件
    "write_json",  # 写入 JSON 文件
    "set_random_seed",  # 设置随机种子
    "download_url",  # 从 URL 下载文件
    "read_image",  # 读取图像
    "collect_env_info",  # 收集环境信息
    "listdir_nohidden",  # 列出非隐藏项
    "get_most_similar_str_to_a_from_b",  # 获取最相似的字符串
    "check_availability",  # 检查可用性
    "tolist_if_not",  # 转换为列表
]


def mkdir_if_missing(dirname):
    """如果缺少目录则创建。"""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """检查给定路径是否是文件。
    参数:
        fpath (str): 文件路径。
    返回:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('在 "{}" 处未找到文件'.format(fpath))
    return isfile


def read_json(fpath):
    """从路径读取 JSON 文件。"""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """写入 JSON 文件。"""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def set_random_seed(seed):
    """设置随机种子。"""
    random.seed(seed) # 为 python 设置种子
    np.random.seed(seed) # 为 numpy 设置种子
    torch.manual_seed(seed) # 为 CPU 设置种子
    torch.cuda.manual_seed_all(seed) # 为所有 GPU 设置种子


def download_url(url, dst):
    """从 URL 下载文件到目标路径。

    参数:
        url (str): 下载文件的 URL。
        dst (str): 目标路径。
    """
    from six.moves import urllib

    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        """ 回调函数：用于下载进度报告。"""
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d 秒已过" %
            (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook) # 下载文件
    sys.stdout.write("\n")


def read_image(path):
    """使用 ``PIL.Image`` 从路径读取图像。并转换为 RGB 模式。
    参数:
        path (str): 图像路径。
    返回:
        PIL 图像
    """
    return Image.open(path).convert("RGB")


def collect_env_info():
    """返回环境信息字符串。
    包括 PyTorch 和 Pillow 的版本信息"""
    # 代码来源：github.com/facebookresearch/maskrcnn-benchmark
    from torch.utils.collect_env import get_pretty_env_info

    env_str = get_pretty_env_info()
    env_str += "\n        Pillow ({})".format(PIL.__version__)
    return env_str


def listdir_nohidden(path, sort=False):
    """列出目录中的 非隐藏文件。
    参数:
         path (str): 目录路径。
         sort (bool): 是否排序。
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")] # 列出非隐藏项
    if sort:
        items.sort()
    return items


def get_most_similar_str_to_a_from_b(a, b):
    """返回 b 中与 a 最相似的字符串。
    参数:
        a (str): 探测字符串。
        b (list): 候选字符串列表。
    """
    highest_sim = 0 # 最高相似度
    chosen = None # 选择的字符串
    for candidate in b:
        sim = SequenceMatcher(None, a, candidate).ratio() # 计算相似度
        if sim >= highest_sim:
            highest_sim = sim
            chosen = candidate 
    return chosen # 返回最相似的字符串


def check_availability(requested, available):
    """检查元素是否在列表中可用。
    参数:
        requested (str): 探测字符串。
        available (list): 可用字符串列表。
    """
    if requested not in available: # 如果请求的字符串不在可用列表中
        psb_ans = get_most_similar_str_to_a_from_b(requested, available)
        raise ValueError(
            "请求的字符串应属于 {}, 但得到 [{}] "
            "(你是指 [{}] 吗？)".format(available, requested, psb_ans)
        )


def tolist_if_not(x):
    """转换为列表。"""
    if not isinstance(x, list):
        x = [x]
    return x
