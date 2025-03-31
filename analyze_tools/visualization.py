"""
使用示例：

python -m analyze_tools.visualization

功能：
    1. 可视化注意力权重
    2. 可视化每一层每个区域的注意力
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from model import build_model
from utils import mkdir_if_missing, load_yaml_config, standard_image_transform
from torchvision.transforms import Compose, ToTensor

def get_attention_map(model, img):
    """
    获取 Clip 图像编码器每一层注意力层的的注意力权重
    
    参数:
        - model (torch.nn.Module): 安装了钩子的注意力模型。
        - img (torch.Tensor): 图像张量 | [batch_size, 3, input_size, input_size]
        
    返回:
        - attention_maps (np.ndarray): 注意力权重，形状为 (num_layers, num_heads, height, width)。
    """

    # 1. 获取 model 的每一层的注意力权重
    attention_maps = []
    with torch.no_grad():  # 关闭梯度计算
       model.visual(img, attention_maps) # attention_maps 是放置在模型中的一个钩子，用于存储每一层的注意力权重
    
    # 2. 转换注意力权重为 NumPy 数组
    attention_maps = np.array([attn.detach().cpu().numpy() for attn in attention_maps])
    return attention_maps


def display_attn_weight(model, img, visualize=False, output_path=None):
    """
    可视化注意力权重

    参数:
        - model (torch.nn.Module): ViT 等包含注意力层的模型。
        - img（torch.Tensor）: 图像张量 | [batch_size, 3, input_size, input_size]
    """
    # 3. 计算注意力映射
    attention_maps = get_attention_map(model, img)

    if not visualize:
        return attention_maps
    else:
        # 5. 可视化注意力映射
        plt.figure(figsize=(12, 6))
        num_layers = min(20, len(attention_maps))  # 只画最多 6 层

        for i in range(num_layers):
            plt.subplot(num_layers//4, 4, i + 1)
            plt.imshow(attention_maps[i].squeeze(0), cmap="Reds")
            plt.title(f"Layer {i+1}")
            plt.colorbar()  # 添加颜色条
            plt.axis("off")

        plt.tight_layout()  # 自动调整子图布局
        plt.savefig(output_path)

        return attention_maps  # 返回注意力权重



# 可视化图像每一层每个区域的注意力
def display_img_attn(model, img, visualize=False, output_path=None):
    """
    可视化每一层每个区域的注意力

    参数:
        - model (torch.nn.Module): ViT 等包含注意力层的模型。
        - img（torch.Tensor）: 图像张量 | [batch_size, 3, input_size, input_size]
    """
    # 1. 获取模型的每一层的注意力权重
    attention_maps = get_attention_map(model, img)
    attention_maps = np.expand_dims(attention_maps[:, :, 0, 1:], axis=2)  # 只观察 cls token 关注哪些像素  (12, 1, 1, 49) 
    
    # 2. 计算每个 patch 受到的总注意力
    # 对每一层的每列求和，得到每个 patch 受到的关注度
    num_layers = len(attention_maps)
    layer_attentions = []
    for i in range(num_layers):
        layer_attention = np.sum(attention_maps[i][0], axis=0)  # 取第 i 层的注意力权重 (1, 50, 49)->(1, 49) 
        layer_attention = np.clip(layer_attention.astype(np.float32),0,1)
        # 归一化
        layer_attention = (layer_attention - layer_attention.min()) / (layer_attention.max() - layer_attention.min())
        layer_attentions.append(layer_attention)

    # 3. 计算所有层的注意力热图叠加 并 归一化
    total_attention = np.sum(layer_attentions, axis=0)
    total_attention = (total_attention - total_attention.min()) / (total_attention.max() - total_attention.min())

    # 4. 重新映射回 224x224 图像
    patchsize = model.visual.patch_size
    size = int(np.sqrt(224*224 // (patchsize*patchsize)))  # 224*224/16*16 = 14*14
    heatmaps = []
    for layer_attention in layer_attentions:
        heatmap = cv2.resize(layer_attention.reshape(size, size), (224, 224), interpolation=cv2.INTER_CUBIC) # ViTb16: 224*224/16*16 = 14*14
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 重新归一化到 [0, 1] | 插值后会导致数值范围变化
        heatmaps.append(heatmap)
    total_heatmap = cv2.resize(total_attention.reshape(size, size), (224, 224), interpolation=cv2.INTER_CUBIC)

    # 5. 可视化原图 + 热图
    if not visualize:
        return heatmaps, total_heatmap
    else:
        plt.figure(figsize=(15, 15))

        # 显示原图
        image_for_display = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_for_display = np.clip(image_for_display.astype(np.float32),0,1) # 转为 float32 类型并归一化到 [0, 1]
        # image_for_display = (image_for_display - image_for_display.min()) / (image_for_display.max() - image_for_display.min())
        plt.subplot(4, 4, 1)  # 在 4x4 网格的第一个位置显示原图
        plt.imshow(image_for_display)
        plt.axis("off")
        plt.title("Original Image")

        # 显示每一层的热图
        for i, heatmap in enumerate(heatmaps):
            plt.subplot(4, 4, i + 2)  # 在 4x4 网格的后续位置显示每一层的热图
            plt.imshow(image_for_display, alpha=0.5)
            plt.imshow(heatmap, cmap="jet", alpha=0.5)
            plt.colorbar()
            plt.axis("off")
            plt.title(f"Attention Heatmap Layer {i+1}")

        # 显示总热图（所有层叠加）
        plt.subplot(4, 4, num_layers + 2)  # 在 4x4 网格的最后一个位置显示总热图
        plt.imshow(image_for_display, alpha=0.5)
        plt.imshow(total_heatmap, cmap="jet", alpha=0.5)
        plt.colorbar()
        plt.axis("off")
        plt.title("Total Attention Heatmap (Sum of All Layers)")

        plt.tight_layout()
        mkdir_if_missing(output_path)
        print(f"保存 attention heatmaps 至 {output_path}/img_attn.png")
        plt.savefig(f"{output_path}/img_attn.png")

if __name__ == "__main__":
    cfg_path = "/root/NP-CLIP/X-Trainer/config/Clip-VitB32-ep10-Caltech101-AdamW.yaml"
    image_path = "/root/NP-CLIP/X-Trainer/analyze_tools/imgs/white.jpg"  # 图像路径
    output_path = "/root/NP-CLIP/X-Trainer/analyze_tools/output"  # 输出路径
    cfg = load_yaml_config(cfg_path) # 读取配置

    # 加载模型
    model = build_model(cfg)  

    # 加载图像，转为张量[batch_size, 3, input_size, input_size]
    input_size = cfg.INPUT.SIZE  # 输入大小
    intermode = cfg.INPUT.INTERPOLATION  # 插值模式
    img = Image.open(image_path)  # 打开图像
    transform = Compose([standard_image_transform(input_size, intermode), ToTensor()])  # 定义转换
    img = transform(img)  # 转换图像
    img = img.unsqueeze(0)  # 添加 batch 维度
    img = img.to(dtype=model.dtype, device=model.device)  # 转移到模型的设备上

    # 可视化注意力权重
    display_img_attn(model, img, visualize=True, output_path=output_path)  # 可视化注意力权重