from .tools import (
    mkdir_if_missing,  # 如果缺少目录则创建
    check_isfile,  # 检查是否是文件
    read_json,  # 读取 JSON 文件
    write_json,  # 写入 JSON 文件
    set_random_seed,  # 设置随机种子
    download_url,  # 从 URL 下载文件
    read_image,  # 读取图像
    collect_env_info,  # 收集环境信息
    listdir_nohidden,  # 列出非隐藏项
    get_most_similar_str_to_a_from_b,  # 获取最相似的字符串
    check_availability,  # 检查可用性
    tolist_if_not,  # 转换为列表
    load_yaml_config,  # 加载配置文件
)

from .logger import (
    Logger,  # 将控制台输出写入外部文本文件类
    setup_logger # 设置标准输出日志
)

from .meters import (
    AverageMeter,  # 计算并存储平均值和当前值
    MetricMeter # 存储一组指标值
)

from .registry import Registry  # 注册表


from .torchtools import (
    save_checkpoint, # 保存检查点
    load_checkpoint, # 加载检查点 
    resume_from_checkpoint, # 从检查点恢复训练
    open_all_layers, # 打开模型中的所有层进行训练
    open_specified_layers, # 打开模型中指定的层进行训练
    count_num_param, # 计算模型中的参数数量
    load_pretrained_weights, # 加载预训练权重到模型
    init_network_weights, # 初始化网络权重
    transform_image, # 对图像应用 K 次 tfm 增强 并返回结果
    standard_image_transform, # 图像预处理转换管道
    patch_jit_model # 将模型转换为 JIT 模型
)

from .download import (
    download_weight, # 通过 url 下载模型权重文件
    download_data # 下载数据并解压
)

from .metrics import (
    compute_distance_matrix,  # 计算距离矩阵的函数
    compute_accuracy,  # 计算准确率的函数
    compute_ci95  # 计算 95% 置信区间的函数
)

from .simple_tokenizer import SimpleTokenizer # 简单的分词器类