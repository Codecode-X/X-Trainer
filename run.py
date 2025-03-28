from utils import load_yaml_config, setup_logger, set_random_seed, collect_env_info
from engine import build_trainer
import argparse
import torch

def reset_cfg(cfg, args):
    """ 将参数 (args) 的设置覆盖到配置 (cfg)。"""
    cfg.OUTPUT_DIR = args.output_dir
    cfg.RESUME = args.resume
    return cfg

def main(args):
    assert args.train or args.eval_only, "训练和评估模式至少要设置一个！"
    assert args.train != args.eval_only, "训练和评估模式不能同时设置！" 

    # -----读取配置文件-----
    cfg = load_yaml_config(args.config_path) # 读取配置
    print("\n=======配置信息=======\n" + str(cfg) + "\n=======配置信息=======\n") # 打印配置以验证

    # -----初始化-----
    # 如果设置了随机种子，则固定种子
    if cfg.SEED >= 0:
        print("设置固定种子：{}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    
    # 设置日志记录器
    setup_logger(cfg.OUTPUT_DIR) # 设置日志记录器

    # 如果支持 CUDA 且配置启用了 CUDA，则优化 CUDA 性能
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # 构建训练器
    trainer = build_trainer(cfg)

    # 测试和训练
    if args.eval_only: # 测试模式
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    else: # 训练模式
        trainer.train()


if __name__ == "__main__":
    default_config_path = 'config/defaults.yaml' # 配置文件路径
    default_output_dir = 'output' # 输出目录
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default=default_config_path, help='配置文件的路径')
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="输出目录")
    parser.add_argument("--resume", type=str, default="", help="检查点目录（从该目录恢复训练）")
    parser.add_argument('--seed', type=int, default=-1, help='随机种子')

    parser.add_argument('--train', action='store_true', help='设置为训练模式')

    parser.add_argument('--eval_only', action='store_true', help='设置为仅评估模式')
    parser.add_argument('--model_dir', type=str, default='', help='模型目录')
    parser.add_argument('--load_epoch', type=int, default=0, help='加载的模型的训练轮数')

    args = parser.parse_args()
    main(args)

    # python run.py --train --config_path config/defaults.yaml --output-dir output
