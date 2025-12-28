import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Generator (U-Net)
# ----------------------
def down_block(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
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
        self.d1 = down_block(in_channels, 64, norm=False)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512)
        self.d5 = down_block(512, 512)
        self.d6 = down_block(512, 512)
        self.d7 = down_block(512, 512)
        self.d8 = down_block(512, 512, norm=False)

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

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1))
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
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))
    
class MultiScaleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_res = UNetGenerator()  # 原始 UNet
        self.refine = UNetGenerator()   # refine UNet

    def forward(self, x):
        # 不下采样，直接原始输入
        fake_low = self.low_res(x)
        # 上采样到原始尺寸（如果 low_res 生成尺寸被修改过）
        fake_low_up = F.interpolate(fake_low, size=x.shape[2:], mode='bilinear', align_corners=False)
        refined = self.refine(fake_low_up)
        return refined
