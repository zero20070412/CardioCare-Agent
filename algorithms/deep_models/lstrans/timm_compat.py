"""
timm 兼容层 - 避免 timm 库导入错误
提供 lsnet_se.py 所需的 trunc_normal_, SqueezeExcite, DropPath
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(tensor, std=0.02):
    """
    截断正态分布初始化（等效于 timm 的 trunc_normal_）
    """
    nn.init.normal_(tensor, mean=0.0, std=std)
    if std > 0:
        with torch.no_grad():
            values = tensor.normal_(0.0, std)
            values = values.clamp_(-2 * std, 2 * std)
            tensor.copy_(values)
    return tensor


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation block (等效于 timm.layers.SqueezeExcite)
    用于 2D 特征图（model_zoo.py 中的 CNN 模型使用）
    """
    def __init__(self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        if rd_channels is None:
            rd_channels = int(in_chs * rd_ratio)
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) (等效于 timm.layers.DropPath)
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
