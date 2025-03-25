from .build import MODEL_REGISTRY
from .ModelBase import ModelBase
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import utils
import os

# 可用的预训练模型
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

@MODEL_REGISTRY.register()
class CLIP(ModelBase):

    def __init__(self,
                 embed_dim: int,
                 # 视觉 (Vision) 部分
                 image_resolution: int, # 输入图像的分辨率 (如 224)
                 vision_layers: Union[Tuple[int, int, int, int], int], # 视觉 Transformer 的层数
                 vision_width: int,  # 视觉 Transformer 的隐藏层宽度
                 vision_patch_size: int,  # Patch 的大小 (ViT 模型)
                 # 文本 (Text) 部分
                 context_length: int,  # 最大文本长度 (序列长度)
                 vocab_size: int, # 词表大小 (Embedding 层的输入维度)
                 transformer_width: int, # Transformer 的隐藏层宽度
                 transformer_heads: int, # Transformer 的注意力头数
                 transformer_layers: int, # Transformer 的层数
                 ):
        super().__init__()
        self.output_logits = None  # 记录模型输出结果
        self.output_featuer = None # 记录模型输出特征

        self.context_length = context_length # 文本序列长度

        # 创建视觉 Transformer
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(  # 基于 ResNet
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64    # 计算 Transformer 头数，通常设为 width/64
            self.visual = VisionTransformer(  # 基于 VIT
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        
        # 构建文本 Transformer
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size  # 词表大小
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # 词嵌入层
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # 位置编码
        self.ln_final = LayerNorm(transformer_width)  # LayerNorm 归一化层

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # project head 将文本特征投影到 embed_dim 维度

        # 可训练的 logit_scale = log(1 / τ) | CLIP 通过 可训练的 logit_scale 让模型自动调整 τ，适应不同数据和任务。
        τ = 0.07 # CLIP 训练时的 softmax 温度参数 τ，较小的 τ(较大的 logit_scale)：分布更加陡峭，相似度高的样本更加突出
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / τ))  # 训练时学习的是 logit_scale，而不是 τ，直接学习 τ 可能会导致梯度更新过大或过小。

        self.initialize_parameters() # 初始化权重
    
    # 权重初始化
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    
    # 构建 因果注意力掩码 (causal attention mask) - 上三角矩阵 - 仅允许 Transformer 看到 当前 Token 及其之前的 Token，防止未来信息泄露。
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    # 视觉编码器
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    # 文本编码器
    def encode_text(self, text):
        # 将输入文本转换为 token 嵌入，形状为 [batch_size, n_ctx(上下文长度), transformer_width]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # 加上可训练的位置编码，保留序列位置信息
        x = x + self.positional_embedding.type(self.dtype)
        
        # 通过 Transformer 进行文本编码
        x = x.permute(1, 0, 2)  # 调整维度为 [n_ctx, batch_size, transformer_width] 以适配 Transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # 还原维度为 [batch_size, n_ctx, transformer_width]

        # 通过 layerNorm 层归一化数据
        x = self.ln_final(x).type(self.dtype)

        # 使用 EOT (End-of-Text) token 对应的特征作为整个文本序列的表示 (类似 Bert 用 [cls] token)
        EOT = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  

        # 通过 `text_projection` 进行线性变换，得到最终的文本特征
        text_features  = EOT @ self.text_projection  

        return text_features

    def forward(self, input, return_feature=False):
        """
        NPCLIP 前向传播。
        
        参数：
            - input (dict): 输入数据 | 包含 'image' 和 'text' 两个键，对应图像和文本数据
            - return_feature (bool): 是否返回特征
        
        返回：
            - output (dict): 输出结果 | 包含 'logits_per_image' 和 'logits_per_text' 两个键，对应图像和文本的相似度
            - feature (dict): 特征 | 包含 'image' 和 'text' 两个键，对应图像和文本的特征
        
        主要步骤：
            1. 提取文本、图像特征
            2. 归一化特征向量
            3. 计算 图像 和 文本 间的 余弦相似度
        """
        # 提取文本、图像特征
        image, text = input['image'], input['text']
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 归一化特征向量：沿着特征维度计算特征向量的模，并用特征向量除以模
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算 图像 和 文本 间的 余弦相似度
        logit_scale = self.logit_scale.exp() # 温度参数 τ 的倒数
        logits_per_image = logit_scale * image_features @ text_features.t() # image->text 相似度
        logits_per_text = logits_per_image.t() # text->image 相似度

        # 返回结果
        self.output_logits = {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }
        self.output_featuer = {
            'image': image_features,
            'text': text_features
        }
        if return_feature: 
            return self.output_logits, self.output_featuer
        else:
            return self.output_logits
    
    @staticmethod
    def build_model(cfg: dict):
        """
        从预训练模型的状态字典 (state_dict) 构建 NPCLIP 模型。
        (未实现 JIT 加载)
        
        参数：
            - cfg (dict): 配置

        返回：
            - model (NPCLIP): 构建的 NPCLIP 模型

        主要步骤：
            1. 载入预训练模型的参数 state_dict
            2. 读取 state_dict 中的模型参数
            3. 根据模型参数实例化 NPCLIP 模型
            4. 将预训练模型的参数加载到 NPCLIP 模型中
        """
        # ---载入预训练模型的参数 state_dict---
        # 从配置中读取 模型名称 和 预训练权重下载保存路径
        name = cfg.MODEL.NAME  # 例如 'RN50'、'ViT-B/32' 等
        if hasattr(cfg.MODEL, 'DOWNLOAD_ROOT'): # 预训练权重保存路径
            download_root = cfg.MODEL.DOWNLOAD_ROOT 
        else:
            download_root = os.path.expanduser("~/.cache/clip")
        if name in _MODELS:
            model_path = utils.download(_MODELS[name], download_root)
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found; available models = {_MODELS.keys()}")
        state_dict = None
        with open(model_path, 'rb') as opened_file:
            state_dict = torch.load(opened_file, map_location="cpu")

        # ---读取 state_dict 中的模型参数---
        vit = "visual.proj" in state_dict
        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

        # ---实例化 NPCLIP 模型---
        model = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

        # ---将预训练模型的参数加载到 NPCLIP 模型中---
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]
        convert_weights(model)
        model.load_state_dict(state_dict)

        # 返回 eval 模式下的 NPCLIP 模型
        return model.eval()




# ------以下为 NPCLIP 模型的子模块或辅助函数的实现------

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """QuickGELU: 通过 x * sigmoid(1.702 * x) 近似 GELU，计算更快，适用于大规模 Transformer 任务"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """残差注意力模块"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    # def attention(self, x: torch.Tensor):
    def attention(self, x: torch.Tensor, attention_maps=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        
        if attention_maps is None:  # 不可视化注意力热图
            output = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
            return output
        else:  # 可视化注意力热图
            output, attn_weights = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
        
        # ---用于可视化注意力热图---
        if attention_maps is not None:
            attention_maps.append(attn_weights)
        # ---------------------------------

        return output
    
    # def forward(self, x: torch.Tensor):
    def forward(self, x: torch.Tensor, attention_maps=None):
        x = x + self.attention(self.ln_1(x), attention_maps)
        x = x + self.mlp(self.ln_2(x))
        return x

"""残差注意力网络"""
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attention_maps=None):
        if attention_maps is None:
            return self.resblocks(x)
        else:
            for block in self.resblocks:
                x = block(x, attention_maps)
            return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        """
        视觉 Transformer (ViT) 结构，适用于图像分类等任务。

        参数:
        - input_resolution (int): 输入图像的分辨率 (例如 224 表示 224x224)。
        - patch_size (int): 每个 patch 的大小 (例如 16 表示 16x16)。
        - width (int): Transformer 的隐藏层维度 (即 embedding 维度)。
        - layers (int): Transformer 的层数 (encoder 块的数量)。
        - heads (int): 多头注意力机制中的头数。
        - output_dim (int): 最终输出的特征维度。
        """
        super().__init__()
        # 保存输入分辨率和输出维度
        self.input_resolution = input_resolution  # 输入图像的分辨率，例如 224×224
        self.output_dim = output_dim  
        
        # **卷积层：将输入图像转换为 patch embedding**
        # 这里使用一个 2D 卷积层，类似于 ViT 直接将图像分割为 patch，并进行 embedding
        self.conv1 = nn.Conv2d(
            in_channels=3,      # 输入通道数，RGB 图像为 3
            out_channels=width, # 输出通道数，即 embedding 维度，即 Transformer 的 隐藏层维度
            kernel_size=patch_size, # patch 大小，例如 16×16
            stride=patch_size,   # 步长等于 patch 大小，相当于划分成不重叠的 patch
            bias=False          # 不使用偏置
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, attention_maps=None):
        """
        前向传播
        参数:
        - x (Tensor): 输入的图像张量，形状为 (batch_size, 3, input_resolution, input_resolution)
        - attention_maps(List): 用来记录注意力热图，可视化
        返回:
        - x (Tensor): 处理后的特征向量，形状为 (batch_size, output_dim)
        """
        # **1. Patch Embedding：将输入图像转换为 patch 级别的 token**
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        
        # **2. 调整形状**
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # **3. 拼接 x 和 class token**
        # 利用自动广播扩展为 batchsize 个 class token
        batch_class_tokens = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], 
                                                                            dtype=x.dtype, device=x.device)
        x = torch.cat([batch_class_tokens, x], dim=1)  # 形状变为 (batch_size, patch_num + 1, width)
        
        # **4. 加入位置编码**
        x = x + self.positional_embedding.to(x.dtype)
        
        x = self.ln_pre(x)  # LN

        # **5. 进入 Transformer 编码器**
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, width) -> (seq_len, batch_size, width)
        x = self.transformer(x, attention_maps)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, width) -> (batch_size, seq_len, width)

        # **6. 取出 CLS token 表示整个视频序列的特征**
        x = self.ln_post(x[:, 0, :])  # LN

        # **7. 进行投影 (如果启用)**
        if self.proj is not None:
            x = x @ self.proj  # 可学习的投影层

        return x  # 返回最终特征

