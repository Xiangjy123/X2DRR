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

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for U-Net skip connections.
    Inputs:
        x: encoder feature map
        g: decoder feature map (upsampled)
    Output:
        x * attention, same size as x
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # W_g: 1x1 conv for decoder feature
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # W_x: 1x1 conv for encoder feature
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Psi: attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetGenerator(nn.Module):
    """
    U-Net Generator for Pix2Pix X-ray -> DRR.
    Supports optional Attention Gate in skip connections.

    Args:
        in_channels (int): input channel, default 1
        out_channels (int): output channel, default 1
        use_attention (bool): whether to use attention gates in skip connections
    """
    def __init__(self, in_channels=1, out_channels=1, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        # ----------------- Encoder -----------------
        self.d1 = down_block(in_channels, 64, norm=False)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512)
        self.d5 = down_block(512, 512)
        self.d6 = down_block(512, 512)
        self.d7 = down_block(512, 512)
        self.d8 = down_block(512, 512, norm=False)

        # ----------------- Decoder -----------------
        self.u1 = up_block(512, 512, dropout=True)
        self.u2 = up_block(1024, 512, dropout=True)
        self.u3 = up_block(1024, 512, dropout=True)
        self.u4 = up_block(1024, 512)
        self.u5 = up_block(1024, 256)
        self.u6 = up_block(512, 128)
        self.u7 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

        # ----------------- Attention Gates -----------------
        if self.use_attention:
            self.ag1 = AttentionGate(F_g=512, F_l=512, F_int=256)
            self.ag2 = AttentionGate(F_g=512, F_l=512, F_int=256)
            self.ag3 = AttentionGate(F_g=512, F_l=512, F_int=256)
            self.ag4 = AttentionGate(F_g=512, F_l=512, F_int=256)
            self.ag5 = AttentionGate(F_g=256, F_l=256, F_int=128)
            self.ag6 = AttentionGate(F_g=128, F_l=128, F_int=64)
            self.ag7 = AttentionGate(F_g=64, F_l=64, F_int=32)

    def forward(self, x):
        # ----------------- Encoder -----------------
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        # ----------------- Decoder -----------------
        u1 = self.u1(d8)

        if self.use_attention:
            d7_ = self.ag1(d7, u1)
            u2 = self.u2(torch.cat([u1, d7_], 1))

            d6_ = self.ag2(d6, u2)
            u3 = self.u3(torch.cat([u2, d6_], 1))

            d5_ = self.ag3(d5, u3)
            u4 = self.u4(torch.cat([u3, d5_], 1))

            d4_ = self.ag4(d4, u4)
            u5 = self.u5(torch.cat([u4, d4_], 1))

            d3_ = self.ag5(d3, u5)
            u6 = self.u6(torch.cat([u5, d3_], 1))

            d2_ = self.ag6(d2, u6)
            u7 = self.u7(torch.cat([u6, d2_], 1))

            d1_ = self.ag7(d1, u7)
            out = self.final(torch.cat([u7, d1_], 1))
        else:
            # 原本的跳跃连接
            u2 = self.u2(torch.cat([u1, d7], 1))
            u3 = self.u3(torch.cat([u2, d6], 1))
            u4 = self.u4(torch.cat([u3, d5], 1))
            u5 = self.u5(torch.cat([u4, d4], 1))
            u6 = self.u6(torch.cat([u5, d3], 1))
            u7 = self.u7(torch.cat([u6, d2], 1))
            out = self.final(torch.cat([u7, d1], 1))

        return out

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
            # [4, 2, 256, 256]
            *block(in_channels, 64, norm=False), # *是解包操作符，把block()返回的列表解包为多个元素
            # [4, 64, 128, 128]
            *block(64, 128),
            # [4, 128, 64, 64]
            *block(128, 256),
            # [4, 256, 32, 32]
            *block(256, 512),
            # [4, 512, 16, 16]
            nn.Conv2d(512, 1, 4, 1, 1)
            # [4, 512, 15, 15]
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))