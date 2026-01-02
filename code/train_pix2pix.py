"""
使用 Pix2Pix 框架训练 X-ray → DRR 生成模型，并支持多种感知与结构一致性损失。
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import pytorch_msssim

from dataset import XrayDRRDataset
from models import UNetGenerator, PatchDiscriminator


# =====================================================
# Argument
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser("Pix2Pix X-ray to DRR")

    # -------- training --------
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)

    # -------- loss weights --------
    parser.add_argument("--lambda_l1", type=float, default=100)
    parser.add_argument("--lambda_perc", type=float, default=1)
    parser.add_argument("--lambda_edge", type=float, default=1)
    parser.add_argument("--lambda_ssim", type=float, default=1)

    return parser.parse_args()


# =====================================================
# Loss
# =====================================================
class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu", layers=("relu1_2", "relu2_2", "relu3_3")):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)
        self.layer_map = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3"}
        self.layers = layers
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        loss = 0.0
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            y = layer(y)
            if self.layer_map.get(name) in self.layers:
                loss += self.criterion(x, y)
        return loss

def edge_loss(pred, target):
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=torch.float32,
        device=pred.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=torch.float32,
        device=pred.device,
    ).view(1, 1, 3, 3)

    gx_p = F.conv2d(pred, sobel_x, padding=1)
    gy_p = F.conv2d(pred, sobel_y, padding=1)
    gx_t = F.conv2d(target, sobel_x, padding=1)
    gy_t = F.conv2d(target, sobel_y, padding=1)

    return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

def ssim_loss(pred, target):
    return 1.0 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)

def generator_loss(fake, real, disc_pred, gan_loss_fn, perc_loss_fn, cfg):
    gan = gan_loss_fn(disc_pred, torch.ones_like(disc_pred))
    l1 = F.l1_loss(fake, real)
    perc = perc_loss_fn(fake, real)
    edge = edge_loss(fake, real)
    ssim = ssim_loss(fake, real)

    total = (
        gan
        + cfg.lambda_l1 * l1
        + cfg.lambda_perc * perc
        + cfg.lambda_edge * edge
        + cfg.lambda_ssim * ssim
    )
    return total


# =====================================================
# Utils
# =====================================================
def to_uint8(t):
    t = (t * 0.5 + 0.5) * 255.0
    return t.squeeze().detach().cpu().numpy().astype("uint8")

def save_loss_curve(G_losses, D_losses, out_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()


# =====================================================
# Main
# =====================================================
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BASE_DIR = Path(__file__).resolve().parent.parent
    data_root = BASE_DIR / "data" / "deepfluoro"
    output_dir = BASE_DIR / "outputs"
    ckpt_dir = BASE_DIR / "checkpoints"

    output_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    subjects = [f"subject0{i}" for i in range(1, 7)]

    # -------- Dataset --------
    dataset = XrayDRRDataset(data_root, subjects, cfg.img_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # -------- Models --------
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999)) # betas是指数衰减率
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    gan_loss_fn = nn.MSELoss()
    perc_loss_fn = PerceptualLoss(device=device)

    G_losses, D_losses = [], []

    # -------- Training --------
    for epoch in range(cfg.epochs):
        # 设置成训练模式
        G.train()
        D.train()

        g_sum, d_sum = 0.0, 0.0

        for xray, drr in loader:
            xray, drr = xray.to(device), drr.to(device)

            # ---- D ----
            fake_drr = G(xray)
            real_pred = D(xray, drr)
            fake_pred = D(xray, fake_drr.detach())

            loss_D = 0.5 * (
                gan_loss_fn(real_pred, torch.ones_like(real_pred))
                + gan_loss_fn(fake_pred, torch.zeros_like(fake_pred))
            )

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---- G ----
            fake_pred = D(xray, fake_drr)
            loss_G = generator_loss(
                fake_drr, drr, fake_pred, gan_loss_fn, perc_loss_fn, cfg
            )

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_sum += loss_G.item()
            d_sum += loss_D.item()

        g_avg = g_sum / len(loader)
        d_avg = d_sum / len(loader)

        G_losses.append(g_avg)
        D_losses.append(d_avg)

        print(f"Epoch [{epoch+1}/{cfg.epochs}]  G: {g_avg:.4f}  D: {d_avg:.4f}")

    # -------- Save --------
    torch.save(G.state_dict(), ckpt_dir / "G_final.pth")
    torch.save(D.state_dict(), ckpt_dir / "D_final.pth")

    save_loss_curve(G_losses, D_losses, output_dir)

    # -------- Visualization --------
    x = xray[0]
    y_real = drr[0]
    y_fake = fake_drr[0]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(to_uint8(x), cmap="gray")
    ax[0].set_title("X-ray")
    ax[1].imshow(to_uint8(y_real), cmap="gray")
    ax[1].set_title("Real DRR")
    ax[2].imshow(to_uint8(y_fake), cmap="gray")
    ax[2].set_title("Generated DRR")
    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "final_comparison.png")
    plt.close()


# =====================================================
# Entry
# =====================================================
if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
