"""
功能:
    1. 从 log.txt 文件中读取测试结果
    2. 计算不同文件夹（种子）之间的均值和标准差

用法:
    1. 解析单个实验的测试结果
        - 计算标准差：$ python tools/parse_test_res.py output/my_experiment
        - 计算 95% 置信区间：$ python tools/parse_test_res.py output/my_experiment --ci95
        - 路径结构如下：
            output/my_experiment
            ├── seed1
            │   └── log.txt
            ├── seed2
            │   └── log.txt
            └── seed3
                └── log.txt
    2. 解析多个实验的测试结果
        - $ python tools/parse_test_res.py output/my_experiment --multi-exp
        - 路径结构如下：
            output/my_experiment
            ├── exp-1
            │   ├── seed1
            │   │   └── log.txt
            │   ├── seed2
            │   │   └── log.txt
            │   └── seed3
            │       └── log.txt
            ├── exp-2
            │   ├── seed1
            │   │   └── log.txt
            │   ├── seed2
            │   │   └── log.txt
            │   └── seed3
            │       └── log.txt
            └── exp-3
                ├── seed1
                │   └── log.txt
                ├── seed2
                │   └── log.txt
                └── seed3
                    └── log.txt
"""
import re 
import numpy as np 
import os.path as osp 
import argparse 
from collections import OrderedDict, defaultdict 
from utils import check_isfile, listdir_nohidden, compute_ci95


def parse_result(*metrics, directory="", args=None, end_signal=None):
    """ 解析日志文件中的指标。
    参数：
        metrics (tuple): 包含指标的元组。
        directory (str): 日志文件所在的目录。
        args (argparse.Namespace): 命令行参数。
        end_signal (str): 结束信号。
    返回：
        OrderedDict: 包含解析结果的字典。
    """
    print(f"Parsing files in {directory}")  # 打印当前解析的目录
    subdirs = listdir_nohidden(directory, sort=True)  # 列出目录中的非隐藏子文件夹

    outputs = []  # 存储每个日志文件的解析结果

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")  # 构造日志文件路径
        assert check_isfile(fpath)  # 确保日志文件存在
        good_to_go = False  # 标志位，表示是否可以开始解析
        output = OrderedDict()  # 存储当前日志文件的解析结果

        with open(fpath, "r") as f:
            lines = f.readlines()  # 读取日志文件的所有行

            for line in lines:
                line = line.strip()  # 去除行首尾的空白字符

                if line == end_signal:  # 如果遇到结束信号
                    good_to_go = True  # 设置标志位为 True

                for metric in metrics:  # 遍历所有需要解析的指标
                    match = metric["regex"].search(line)  # 使用正则表达式匹配行内容
                    if match and good_to_go:  # 如果匹配成功且标志位为 True
                        if "file" not in output:  # 如果尚未记录文件路径
                            output["file"] = fpath  # 记录文件路径
                        num = float(match.group(1))  # 提取匹配的数值
                        name = metric["name"]  # 获取指标名称
                        output[name] = num  # 将指标名称和数值存入结果字典

        if output:  # 如果当前日志文件有解析结果
            outputs.append(output)  # 将结果添加到输出列表

    assert len(outputs) > 0, f"Nothing found in {directory}"  # 确保至少有一个解析结果

    metrics_results = defaultdict(list)  # 存储所有日志文件的指标结果

    for output in outputs:
        msg = ""  # 用于打印当前日志文件的解析结果
        for key, value in output.items():
            if isinstance(value, float):  # 如果值是浮点数
                msg += f"{key}: {value:.2f}%. "  # 格式化为百分比
            else:
                msg += f"{key}: {value}. "  # 直接打印值
            if key != "file":  # 如果不是文件路径
                metrics_results[key].append(value)  # 将值添加到指标结果中
        print(msg)  # 打印当前日志文件的解析结果

    output_results = OrderedDict()  # 存储最终的平均结果

    print("===")
    print(f"Summary of directory: {directory}")  # 打印当前目录的汇总结果
    for key, values in metrics_results.items():
        avg = np.mean(values)  # 计算均值
        std = compute_ci95(values) if args.ci95 else np.std(values)  # 计算标准差或 95% 置信区间
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")  # 打印均值和误差
        output_results[key] = avg  # 将均值存入最终结果
    print("===")

    return output_results  # 返回最终结果


def main(args, end_signal):
    # 定义要解析的指标，包括名称和匹配的正则表达式
    metric = {
        "name": args.keyword,  # 指标名称，从命令行参数中获取
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),  # 匹配指标的正则表达式
    }

    if args.multi_exp:  # 如果启用了多实验模式
        final_results = defaultdict(list)  # 存储所有实验的最终结果

        # 遍历目录中的每个子目录
        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)  # 构造子目录路径
            # 解析当前子目录的日志文件
            results = parse_result(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            # 将解析结果添加到最终结果中
            for key, value in results.items():
                final_results[key].append(value)

        print("Average performance")  # 打印平均性能
        # 遍历所有指标，计算均值并打印
        for key, values in final_results.items():
            avg = np.mean(values)  # 计算均值
            print(f"* {key}: {avg:.2f}%")  # 打印均值

    else:  # 如果不是多实验模式
        # 直接解析指定目录的日志文件
        parse_result(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to directory")  # 指定目录路径
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )  # 是否计算 95% 置信区间
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")  # 是否解析测试日志
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )  # 是否解析多个实验
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )  # 指定要提取的指标关键字
    args = parser.parse_args()  # 解析命令行参数

    # 设置结束信号，根据是否解析测试日志进行调整
    end_signal = "Finish training"
    if args.test_log:
        end_signal = "=> result"

    # 调用主函数
    main(args, end_signal)
