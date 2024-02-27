import torch
from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch, input_dim: int, seq_len: int):
    conv_blks = []
    orin_seq_len = seq_len
    in_channels = 1  # 卷积层部分
    for num_convs, out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        input_dim //= 2
        seq_len //= 2
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),  # 全连接层部分
        nn.Linear(out_channels * input_dim * seq_len, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        # nn.ReLU(),
        # nn.Linear(4096, 10)
    )
