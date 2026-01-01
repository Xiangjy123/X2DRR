"""
visualize.py
X-ray → DRR 模型可视化与可解释性分析
支持:
1. Feature Map 可视化
2. Grad-CAM 可视化
3. Occlusion 可视化
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models import UNetGenerator
from dataset import XrayDRRDataset

# ----------------------------
# 配置
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型 checkpoint
checkpoint_path = "../checkpoints/G_final.pth"

# 数据集
root_dir = "../data/deepfluoro"
subjects = ["subject01", "subject02"]  # 根据需要修改
img_size = 256

# 可视化保存路径
save_dir = "../visualizations"
os.makedirs(save_dir, exist_ok=True)

# 可视化方法: 'featuremap' / 'gradcam' / 'occlusion'
method = 'featuremap'
# 指定中间层名称（U-Net 编码器 d4/d5/d6/d7/d8 都可以尝试）
target_layer_name = 'd5'
max_samples = 5

# ----------------------------
# 数据加载
# ----------------------------
dataset = XrayDRRDataset(root_dir, subjects, img_size)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ----------------------------
# 模型加载
# ----------------------------
model = UNetGenerator(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ----------------------------
# Hook 用于 Feature Map / Grad-CAM
# ----------------------------
activation = {}
gradients = {}

def forward_hook(module, input, output):
    activation['value'] = output.detach()

def backward_hook(module, grad_input, grad_output):
    gradients['value'] = grad_output[0]

# 获取指定层
target_layer = dict(model.named_modules())[target_layer_name]
target_layer.register_forward_hook(forward_hook)
if method == 'gradcam':
    target_layer.register_backward_hook(backward_hook)

# ----------------------------
# 可视化函数
# ----------------------------
def show_image(img_tensor, title=None, save_path=None):
    img = img_tensor.squeeze().cpu().numpy()
    plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def feature_map_vis(x, idx):
    _ = model(x)
    fmap = activation['value'][0]  # [C, H, W]
    n_channels = min(16, fmap.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(8,8))
    for i, ax in enumerate(axes.flatten()):
        if i < n_channels:
            ax.imshow(fmap[i].cpu(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{idx}_featuremap.png"))
    plt.close()

def gradcam_vis(x, y, idx):
    x = x.requires_grad_(True)
    output = model(x)
    loss = F.l1_loss(output, y)
    model.zero_grad()
    loss.backward()
    grad = gradients['value']  # [1, C, H, W]
    fmap = activation['value']  # [1, C, H, W]
    weights = grad.mean(dim=(2,3), keepdim=True)  # global average pooling
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    x_img = x.squeeze().cpu().numpy()
    plt.imshow(x_img, cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{idx}_gradcam.png"))
    plt.close()

def occlusion_vis(x, y, idx, patch_size=16):
    x_np = x.squeeze().cpu().numpy()
    H, W = x_np.shape
    heatmap = np.zeros((H, W))
    baseline_output = model(x).detach()
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            x_occ = x.clone()
            x_occ[:, :, i:i+patch_size, j:j+patch_size] = 0
            out = model(x_occ).detach()
            diff = F.l1_loss(out, baseline_output).item()
            heatmap[i:i+patch_size, j:j+patch_size] = diff
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    plt.imshow(x_np, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{idx}_occlusion.png"))
    plt.close()

# ----------------------------
# 批量可视化
# ----------------------------
for idx, (x, y) in enumerate(loader):
    if idx >= max_samples:
        break
    x = x.to(device)
    y = y.to(device)
    if method == 'featuremap':
        feature_map_vis(x, idx)
    elif method == 'gradcam':
        gradcam_vis(x, y, idx)
    elif method == 'occlusion':
        occlusion_vis(x, y, idx)
    print(f"Saved visualization for sample {idx}")

print("可视化完成，结果保存在:", save_dir)
