import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import torch
from pathlib import Path
import os
import math
from loguru import logger
plt.ioff()

# 看某个通道的特征图
def Feaftures_vision(features_map, channel_index):
    B, N, C = features_map.shape
    features_map = features_map.reshape(1, 14, 14, C)
    features_map = features_map.permute(0, 3, 1, 2)
    plt.imshow(features_map[0, channel_index, :, :].cpu())


def CNN_Feature_Visualization(x, module_type, n=32, save_path=Path('feat')):
    batch, channels, height, width = x.shape  # b,c,h,w
    # os.makedirs(save_path, exist_ok=True)
    if height > 1 and width > 1:
        f = save_path / f"{module_type.split('.')[-1]}_features.png"
        block = torch.chunk(x[0].cpu(), channels, dim=0)  # tuple len(block) = channels
        n = min(n, channels)
        fig, ax = plt.subplots(math.ceil(n / 8), 8, figsize=(16, 16), tight_layout=True)
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):  # 遍历前n个通道
            ax[i].imshow(block[i].squeeze())  # block[i].squeeze() shape is (h, w)
            ax[i].axis('off')
        logger.info(f'Saving {f}... ({n}/{channels})')
        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close(fig)
