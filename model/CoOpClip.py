from .Clip import Clip, Tokenizer, tokenize
from torch import nn
import warnings
import torch
from .build import MODEL_REGISTRY
from .ModelBase import ModelBase

@MODEL_REGISTRY.register()
class CoOpClip(ModelBase):
    """ 
    CoOpClip 模型：基于提示学习调优的图像和文本的对比学习
    该模型使用 CLIP 模型作为基础，并在其上添加了一个提示学习器（Prompt Learner），用于生成每个类别的最优提示信息。
    
    主要步骤：
    1. 编码图像特征
    2. 利用可学习的 Prompt Learner 生成没饿过类别的文本提示词，并编码文本特征
    3. 计算图像和文本之间的余弦相似度
    4. 返回与图像最相似的文本提示词对应的类别作为结果
    """
    def __init__(self, cfg):
        """
        初始化 CoOpClip 模型
        
        参数：
        cfg: 配置文件，包含模型的超参数和训练设置
        """
        super().__init__(cfg)  # 调用父类 Clip 的构造函数
        # 读取预训练的 CLIP 模型
        pretrained_clip = Clip.build_model(cfg) # 调用父类 build 得到预训练的 CLIP 模型 (读取预训练权重)
        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":  # clip模型默认是 float16 精度
            pretrained_clip.float() # 将模型转换为 float32 精度
            
        self.image_encoder = pretrained_clip.visual  # 图像编码器
        self.text_encoder = pretrained_clip.transformer  # 文本编码器
        self.token_embedding = pretrained_clip.token_embedding  # 词嵌入层
        self.positional_embedding = pretrained_clip.positional_embedding  # 位置嵌入层
        self.ln_final = pretrained_clip.ln_final  # 最终的 LayerNorm 层
        self.text_projection = pretrained_clip.text_projection  # 文本投影层
        self.logit_scale = pretrained_clip.logit_scale  # 温度参数 τ 的倒数
        self.device = pretrained_clip.device  # 设备
        self.dtype = pretrained_clip.dtype  # 数据类型
        self.cfg = cfg  # 配置文件

        # 提示学习 - 通过训练器调用 init_promptLearner 初始化
        self.pptLearner = None  # 提示学习器
        self.eot_indices = None  # 每个类别的 prompt 的结束符位置

    def init_promptLearner(self, label_texts:list):
        """
        初始化提示学习器
        
        参数：
            - label_texts (list): 类别文本列表 | [num_classes] | ['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
        
        
        主要步骤：
            1. 初始化一个 PromptLearner 对象 self.pptLearner，用于学习提示信息
            2. 获取每个类别的 prompt 的结束符位置 eot_indices，保存至 self.eot_indices
        """
        print("正在初始化 PromptLearner...")
        self.pptLearner = PromptLearner(self.cfg, label_texts, self)  # 初始化一个 PromptLearner 对象，用于学习提示信息
        self.eot_indices = self.pptLearner.eot_indices  # 每个类别的 prompt 的结束符位置

    def forward(self, image, return_feature=False):
        """ 
        重写 forward 函数 
        
        参数：
            - image(torch.Tensor): 输入图像，形状为 (batch_size, 3, height, width)
            - return_feature(bool): 是否返回图像特征，默认为 False

        主要步骤：
            1. 编码图像特征
            2. 生成每个类别的 prompt，拼接可训练的 prompts 和 可训练的位置嵌入 得到文本输入
            3. 编码文本特征
            4. 计算图像和文本之间的余弦相似度
            5. 返回与图像最相似的文本提示词对应的类别作为结果
        """
        # ---编码图像特征---
        image_features = self.image_encoder(image.type(self.dtype))  
        # ---生成 prompt---
        prompts = self.pptLearner() # 获取学习到的每个类别的 prompt
        # ---编码文本特征---
        # 可训练的 prompts + 可训练的位置嵌入
        t = prompts + self.positional_embedding.type(self.dtype)
        # 交换维度，使其符合文本编码器输入格式 
        t = t.permute(1, 0, 2)  # (batch_size, n_ctx, width) -> (n_ctx, batch_size, width) | n_ctx(序列长度=num_patches+1) width(隐藏层宽度)
        t = self.text_encoder(t) # 编码文本特征
        t = t.permute(1, 0, 2)  # LND -> NLD
        t = self.ln_final(t).type(self.dtype) # (n_ctx, batch_size, width) -> (batch_size, n_ctx, width)
        # 获取文本编码特征
        batch_indices = torch.arange(t.shape[0])
        EOT = t[batch_indices, self.pptLearner.eot_indices]
        # 线性投影得到文本特征
        text_features = EOT @ self.text_projection
        # ---计算 图像 和 文本 间的 余弦相似度---
        logit_scale = self.logit_scale.exp() # 温度参数 τ 的倒数
        logits_per_image = logit_scale * image_features @ text_features.t() # image->text 相似度 | [batch, num_classes]
        # ---返回结果---
        if return_feature: 
            return logits_per_image, image_features
        else:
            return logits_per_image
        

class PromptLearner(nn.Module):
    """ 
    提示学习器：用学习所有类别通用的上下文词ctx来生成每个类别的提示信息
    
    参数：
        - cfg: 配置文件，包含模型的超参数和训练设置
        - classnames: 类别名称列表，用于生成提示信息
        - clip_model: 实例化的 CLIP 模型对象

    配置：
        - cfg.MODEL.init_ctx: 初始化的上下文词，例如 "a photo of a"
    """
    def __init__(self, cfg, classnames, clip_model):
        """
        初始化提示学习器

        参数：
            - cfg: 配置文件，包含模型的超参数和训练设置
            - classnames: 类别名称列表，用于生成提示信息 | 如 ['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
            - clip_model: 实例化的 CLIP 模型对象
        
        主要步骤：
        1. 读取参数和配置
        2. 初始化上下文前缀词和嵌入向量 | 上下文前缀词为所有类别通用
        3. 构造每个类别的完整的提示文本 -> [SOS] + 上下文向量 ctx(可学习) + 类别名 + 句号 + [EOS]
        4. 注册需持久化保存的张量 token_prefix 和 token_suffix
        5. 将上下文向量 learnable_ctx 设为可训练参数
        6. 记录每一类的提示文本的结尾 EOT 索引位置 eot_indices
        """
        super().__init__()
        # 读取参数
        n_cls = len(classnames)  # 类别数量
        dtype = clip_model.dtype  # CLIP 模型的数据类型
        clip_imsize = clip_model.image_encoder.input_resolution # CLIP 模型的输入图像尺寸
        # 读取配置
        ctx_init = cfg.MODEL.init_ctx  # 是否使用预设的上下文词 | 如 "a photo of a" | 所有类别通用的上下文词
        cfg_imsize = cfg.INPUT.SIZE # 配置文件中设定的输入图像尺寸
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) 必须等于 clip_imsize ({clip_imsize})" # 确保输入尺寸匹配
        
        """初始化 上下文前缀词 以及其 嵌入向量"""
        ctx_init = ctx_init.replace("_", " ")  # 将下划线替换为空格
        n_ctx = len(ctx_init.split(" "))  # 重新计算上下文词数量
        tokenized_ctx = tokenize(ctx_init)  # 将文字形式的 ctx_init 分词编码为 token
        with torch.no_grad():
            tokenized_ctx = tokenized_ctx.to(clip_model.device)
            ctx_embedding = clip_model.token_embedding(tokenized_ctx).type(dtype)  # 获取嵌入表示 (batch_size, seq_len, embedding_dim)
        ctx_vectors = ctx_embedding[0, 1:1+n_ctx, :]  # 提取上下文嵌入向量 | 1:1+n_ctx: 索引 0 位置对应 [SOS]
        prompt_prefix = ctx_init  # 设定上下文前缀就是初始上下文词 ctx_init
        print(f'初始上下文向量："{prompt_prefix}"')
        print(f"上下文 token 数为 (tokens): {n_ctx}")
        
        """构造 每个类别 的 prompt 文本（[SOS] + 上下文向量 ctx(可学习) + 类别名 + 句号 + [EOS]）"""
        classnames = [name.replace("_", " ") for name in classnames]  # 处理类别名称中的"_"
        prompts = [prompt_prefix + " " + name + "." for name in classnames]  # 生成 prompt 文本（前缀词 + name + 句号）
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])  # 将文字形式的 prompt 分词编码为 token
        eot_indices = tokenized_prompts.argmax(dim=-1)  # 获取 tokenized_prompts 的 [EOS] token 的 索引
        with torch.no_grad():
            tokenized_prompts = tokenized_prompts.to(clip_model.device)  # 将 tokenized_prompts 移动到 CLIP 模型所在设备
            ctx_embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # 得到 tokenized_prompts 的嵌入表示

        # 注册需持久化保存的张量。这些张量会成为模型状态字典（state_dict）的一部分，确保在模型保存和加载时被正确处理。
        self.register_buffer("token_prefix", ctx_embedding[:, :1, :])  # [SOS]
        self.register_buffer("token_suffix", ctx_embedding[:, 1 + n_ctx :, :])  # 包括 类别名 和 [EOS]

        self.n_cls = n_cls  # 类别数
        
        # 将 所有类别通用的 上下文向量 ctx 设为 可训练参数
        self.ctx = nn.Parameter(ctx_vectors)

        # 记录每一类的 prompt 的结尾 EOT 索引位置
        self.eot_indices = eot_indices

    def forward(self):
        learnable_ctx = self.ctx  # 取出上下文向量
        if learnable_ctx.dim() == 2: # 只存在于所有类别都用同样的前缀
            learnable_ctx = learnable_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # 维度适配

        # register_buffer 保存的 prefix 和 suffix
        prefix = self.token_prefix  # [SOS]
        suffix = self.token_suffix  # 包括 类别名 和 [EOS]
        
        # 类别名称放在结尾（论文实验显示"end"效果最好）
        prompts = torch.cat(  # [SOS] + 上下文 + 类别名 + [EOS]
            [
                prefix,  # (n_cls, 1, dim) [SOS]
                learnable_ctx,     # (n_cls, n_ctx, dim) 学习到的上下文
                suffix,  # (n_cls, *, dim) 包括 类别名+[EOS]
            ],
            dim=1,
        )
        return prompts # 完整的 prompts -> [SOS] + 上下文 + 类别名 + [EOS]