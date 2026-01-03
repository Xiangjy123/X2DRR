"""
models.py
定义用于 X-ray → DRR 生成任务的 Pix2Pix 网络结构，包括 U-Net 生成器与 PatchGAN 判别器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Generator (U-Net)
# ----------------------
def down_block(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)] # 卷积核大小为4，步长为2，padding为1
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True)) # 0.2是LeakyReLU的负斜率
    return nn.Sequential(*layers)

def up_block(in_c, out_c, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # x: [batch_size=4, channel=1, height=256, width=256]
        self.d1 = down_block(in_channels, 64, norm=False) # 第一层不batchNorm，避免丢失重要信息
        # [4, 64, 128, 128]
        self.d2 = down_block(64, 128)
        # [4, 128, 64, 64]
        self.d3 = down_block(128, 256)
        # [4, 256, 32, 32]
        self.d4 = down_block(256, 512)
        # [4, 512, 16, 16]
        self.d5 = down_block(512, 512)
        # [4, 512, 8, 8]
        self.d6 = down_block(512, 512)
        # [4, 512, 4, 4]
        self.d7 = down_block(512, 512)
        # [4, 512, 2, 2]
        self.d8 = down_block(512, 512, norm=False) # 特征已经高度抽象，归一化可能限制模型的表达能力
        # [4, 512, 1, 1]

        self.u1 = up_block(512, 512, dropout=True)
        # [4, 512, 2, 2]
        # [4, 1024, 2, 2]
        self.u2 = up_block(1024, 512, dropout=True)
        # [4, 512, 4, 4]
        # [4, 1024, 4, 4]
        self.u3 = up_block(1024, 512, dropout=True)
        # [4, 512, 8, 8]
        # [4, 1024, 8, 8]
        self.u4 = up_block(1024, 512)
        # [4, 512, 16, 16]
        # [4, 1024, 16, 16]
        self.u5 = up_block(1024, 256)
        # [4, 256, 32, 32]
        # [4, 512, 32, 32]
        self.u6 = up_block(512, 128)
        # [4, 128, 64, 64]
        # [4, 256, 64, 64]
        self.u7 = up_block(256, 64)
        # [4, 64, 128, 128]

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [batch_size=4, channel=1, height=256, width=256]
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1)) # 每个上采样层都拼接了对应编码器层的特征（沿通道拼接）
        u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1))
        u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1))
        u7 = self.u7(torch.cat([u6, d2], 1))

        return self.final(torch.cat([u7, d1], 1))

# ----------------------
# Discriminator (PatchGAN)
# ----------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)] # 卷积核大小为4，步长为2，padding为1
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(

            *block(in_channels, 64, norm=False), # *是解包操作符，把block()返回的列表解包为多个元素
            
            *block(64, 128),

            *block(128, 256),

            *block(256, 512),

            nn.Conv2d(512, 1, 4, 1, 1)

        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))