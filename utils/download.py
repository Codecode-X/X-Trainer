import hashlib
import os
import os.path as osp
import tarfile
import zipfile
import gdown
import urllib
import warnings
from tqdm import tqdm

__all__ = [
    "download_weight", # 通过 url 下载模型权重文件
    "download_data" # 下载数据并解压
]

def download_weight(url: str, root: str):
    """
    下载指定 URL 的权重文件到指定目录，并返回下载后的文件路径。
       
    参数：
        - url：str，文件的 URL 地址。
        - root：str，下载文件的目录路径。
        
    返回：
        - str，下载后的文件路径。

    主要步骤:
        1. 创建文件下载保存的路径 download_target
        2. 获取文件的 SHA256 值
        3. 检查目标路径是否已经存在下载好的文件
        4. 如果文件已经存在，检查其 SHA256 值是否匹配，如果匹配则直接返回文件路径，否则重新下载文件
        5. 下载文件并显示进度条
        6. 下载完成后再次检查 SHA256 值
    """

    # ---创建文件下载保存的路径 download_target ---
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)

    # ---下载文件---
    # 获取文件的 SHA256 值
    expected_sha256 = url.split("/")[-2]
    # 检查目标路径是否已经存在下载好的文件
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} 存在，但不是一个常规文件")
    # 如果文件已经存在，检查其 SHA256 值是否匹配，如果匹配则直接返回文件路径，否则重新下载文件
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} 已存在，但 SHA256 校验和不匹配；重新下载文件")
    # 下载文件并显示进度条
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    # ----文件校验----
    # 下载完成后再次检查 SHA256 值
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("文件已下载，但 SHA256 校验和不匹配")

    return download_target


def download_data(url, dst, from_gdrive=True):
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