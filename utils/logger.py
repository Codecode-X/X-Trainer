import os
import sys
import time
import os.path as osp

from .tools import mkdir_if_missing

__all__ = [
    "Logger",  # 将控制台输出写入外部文本文件类
    "setup_logger" # 设置标准输出日志
]


class Logger:
    """
    将控制台输出写入外部文本文件的类。
    参数:
        fpath (str): 保存日志文件的目录。

    示例::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout  # 保存当前的标准输出
        self.file = None  # 初始化文件为 None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))  # 如果目录不存在则创建
            self.file = open(fpath, "w")  # 打开文件进行写操作

    def __del__(self):
        self.close()  # 在对象被销毁时关闭文件

    def __enter__(self):
        pass  # 上下文管理器的进入方法

    def __exit__(self, *args):
        self.close()  # 上下文管理器的退出方法，关闭文件

    def write(self, msg):
        """将消息写入控制台，如果文件存在，也同时将消息写入文件。"""
        self.console.write(msg)  # 将消息写入控制台
        if self.file is not None:
            self.file.write(msg)  # 如果文件存在，将消息写入文件

    def flush(self):
        """
        强制刷新缓冲区，确保数据立即写入控制台和文件输出。
        确保每个线程的日志及时写入文件，防止多个线程的日志互相覆盖或丢失
        
        相关知识：当写入文件时，Python 可能会先将数据存入缓冲区，并在适当的时候（比如关闭文件或缓冲区满了）才写入磁盘。
        """
        self.console.flush()  # 刷新控制台输出
        if self.file is not None:
            self.file.flush()  # 刷新文件输出
            os.fsync(self.file.fileno())  # 确保文件内容写入磁盘

    def close(self):
        """关闭控制台和文件输出。"""
        self.console.close()  # 关闭控制台输出
        if self.file is not None:
            self.file.close()  # 关闭文件


def setup_logger(output=None):
    """设置标准输出日志。
    参数:
        output (str): 日志文件的路径 (以.txt 或.log 结尾)。
    """
    if output is None:
        return  # 如果没有提供输出路径，则返回

    if isinstance(output, str) and (output.endswith(".txt") or output.endswith(".log")):
        fpath = output  # 如果输出路径以.txt 或.log 结尾，则直接使用该路径
    else:
        fpath = osp.join(output, "log.txt")  # 否则，将输出路径与"log.txt"拼接

    if osp.exists(fpath):
        # 确保现有的日志文件不会被覆盖
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")  # 在文件名后添加时间戳

    sys.stdout = Logger(fpath)  # 将标准输出重定向到 Logger 实例
