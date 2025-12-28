import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import pytorch_msssim
from pathlib import Path
from dataset import XrayDRRDataset
from models import MultiScaleGenerator, PatchDiscriminator, UNetGenerator

# ----------------------
# 配置
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent.parent
data_root = BASE_DIR / "data" / "xray_drr"
output_dir = BASE_DIR / "outputs"
checkpoint_dir = BASE_DIR / "checkpoints"
os.makedirs(output_dir, exist_ok=True)

subjects = [f"subject0{i}" for i in range(1,7)]
batch_size = 4
img_size = 256
num_epochs = 400

# ----------------------
# Dataset & Dataloader
# ----------------------
dataset = XrayDRRDataset(data_root, subjects, img_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------------
# Models
# ----------------------
G = MultiScaleGenerator().to(device)
D = PatchDiscriminator().to(device)

# ----------------------
# Optimizer & Loss
# ----------------------
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))
criterion_GAN = nn.MSELoss()

# Loss 权重
lambda_L1, lambda_perc, lambda_edge, lambda_ssim = 100, 1, 1, 1

# ----------------------
# 感知损失
# ----------------------
class PerceptualLoss(nn.Module):
    def __init__(self, device='cpu', layers=['relu1_2','relu2_2','relu3_3']):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        for p in vgg.parameters(): p.requires_grad=False
        self.vgg_layers = vgg.to(device)   # <-- 迁移到 GPU
        self.layer_name_mapping = {'3':"relu1_2",'8':"relu2_2",'15':"relu3_3"}
        self.layers = layers
        self.criterion = nn.L1Loss()

    def forward(self,x,y):
        x = x.repeat(1,3,1,1) if x.shape[1]==1 else x
        y = y.repeat(1,3,1,1) if y.shape[1]==1 else y
        loss = 0
        for name,module in self.vgg_layers._modules.items():
            x = module(x)
            y = module(y)
            if self.layer_name_mapping.get(name) in self.layers:
                loss += self.criterion(x,y)
        return loss

perceptual_criterion = PerceptualLoss(device=device)


# ----------------------
# 边缘损失
# ----------------------
def edge_loss(pred,target):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32,device=pred.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32,device=pred.device).view(1,1,3,3)
    gx_pred = F.conv2d(pred,sobel_x,padding=1)
    gy_pred = F.conv2d(pred,sobel_y,padding=1)
    gx_target = F.conv2d(target,sobel_x,padding=1)
    gy_target = F.conv2d(target,sobel_y,padding=1)
    return F.l1_loss(gx_pred,gx_target)+F.l1_loss(gy_pred,gy_target)

# ----------------------
# SSIM Loss
# ----------------------
def ssim_loss(pred,target):
    return 1 - pytorch_msssim.ssim(pred,target,data_range=1.0,size_average=True)

# ----------------------
# Generator Loss
# ----------------------
def generator_loss(fake_drr, real_drr, disc_pred):
    gan_loss = criterion_GAN(disc_pred, torch.ones_like(disc_pred))
    l1 = nn.L1Loss()(fake_drr, real_drr)
    perc = perceptual_criterion(fake_drr, real_drr)
    edge = edge_loss(fake_drr, real_drr)
    ssim = ssim_loss(fake_drr, real_drr)
    total = gan_loss + lambda_L1*l1 + lambda_perc*perc + lambda_edge*edge + lambda_ssim*ssim
    return total

# ----------------------
# Training
# ----------------------
G_losses, D_losses = [], []

for epoch in range(num_epochs):
    epoch_G_loss, epoch_D_loss = 0, 0
    for xray, drr in loader:
        xray, drr = xray.to(device), drr.to(device)

        # ---- Train Discriminator ----
        fake_drr = G(xray)
        real_pred = D(xray, drr)
        fake_pred = D(xray, fake_drr.detach())
        loss_D = 0.5 * (criterion_GAN(real_pred, torch.ones_like(real_pred)) +
                        criterion_GAN(fake_pred, torch.zeros_like(fake_pred)))
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---- Train Generator ----
        fake_pred = D(xray, fake_drr)
        loss_G = generator_loss(fake_drr, drr, fake_pred)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        epoch_G_loss += loss_G.item()
        epoch_D_loss += loss_D.item()

    epoch_G_loss /= len(loader)
    epoch_D_loss /= len(loader)
    G_losses.append(epoch_G_loss)
    D_losses.append(epoch_D_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]  G_loss: {epoch_G_loss:.4f}  D_loss: {epoch_D_loss:.4f}")

# ----------------------
# 训练结束后统一保存
# ----------------------
torch.save(G.state_dict(), os.path.join(checkpoint_dir, "G_final.pth"))
torch.save(D.state_dict(), os.path.join(checkpoint_dir, "D_final.pth"))

# ---- 生成对比图 ----
x = xray[0].detach().cpu()
y_real = drr[0].detach().cpu()
y_fake = fake_drr[0].detach().cpu()

def to_uint8(t):
    t = (t*0.5 + 0.5)*255
    t = t.squeeze().numpy().astype("uint8")
    return t

x_img = to_uint8(x)
y_real_img = to_uint8(y_real)
y_fake_img = to_uint8(y_fake)

fig, axes = plt.subplots(1,3,figsize=(12,4))
axes[0].imshow(x_img, cmap="gray"); axes[0].set_title("X-ray"); axes[0].axis("off")
axes[1].imshow(y_real_img, cmap="gray"); axes[1].set_title("Real DRR"); axes[1].axis("off")
axes[2].imshow(y_fake_img, cmap="gray"); axes[2].set_title("Generated DRR"); axes[2].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "final_comparison.png"))
plt.close()

# ---- 保存 Loss 曲线 ----
plt.figure(figsize=(8,6))
plt.plot(range(1,num_epochs+1), G_losses, label="Generator Loss")
plt.plot(range(1,num_epochs+1), D_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pix2Pix Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()
