import hashlib
import os
import urllib
import warnings
from tqdm import tqdm

def download(url: str, root: str):
    """
    下载指定 URL 的文件到指定目录，并返回下载后的文件路径。
       
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