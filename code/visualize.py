"""
visualize.py
X-ray → DRR 模型可视化与可解释性分析
支持:
1. Feature Map 可视化
2. Grad-CAM 可视化
3. Occlusion 可视化
4. 输出误差可视化
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models import UNetGenerator
from dataset import XrayDRRDataset
import argparse
import matplotlib.gridspec as gridspec

# ----------------------------
# 命令行参数
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
parser.add_argument("--root_dir", type=str, required=True, help="数据集根目录")
parser.add_argument("--subjects", type=str, nargs='+', default=None, help="要处理的 subject 列表")
parser.add_argument("--img_size", type=int, default=256, help="输入图像尺寸")
parser.add_argument(
    "--method", type=str, nargs='+',
    default=['featuremap', 'gradcam', 'occlusion', 'output_error'],
    help="可视化方法，支持多个: featuremap gradcam occlusion output_error"
)
parser.add_argument("--layer", type=str, default="d5", help="目标中间层名称")
parser.add_argument("--max_samples", type=int, default=5, help="最多处理样本数")
parser.add_argument("--save_dir", type=str, default="../visualizations", help="可视化保存目录")
args = parser.parse_args()

# ----------------------------
# 配置
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.save_dir, exist_ok=True)

methods = args.method
target_layer_name = args.layer
max_samples = args.max_samples
img_size = args.img_size
subjects = args.subjects

# ----------------------------
# 数据加载
# ----------------------------
dataset = XrayDRRDataset(args.root_dir, subjects, img_size)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ----------------------------
# 模型加载
# ----------------------------
model = UNetGenerator(in_channels=1, out_channels=1).to(device)
state_dict = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ----------------------------
# Hook 用于 Feature Map / Grad-CAM
# ----------------------------
activation = {}
gradients = {}

def forward_hook(module, input, output):
    activation['value'] = output.detach()

def backward_hook(module, grad_input, grad_output):
    gradients['value'] = grad_output[0].detach()

# -------- 安全获取目标层 --------
named_modules = dict(model.named_modules())
if target_layer_name not in named_modules:
    raise ValueError(
        f"target_layer_name='{target_layer_name}' 不存在于模型中，"
        f"可选层包括：{list(named_modules.keys())}"
    )
target_layer = named_modules[target_layer_name]

# -------- 注册 Hook --------
if 'featuremap' in methods or 'gradcam' in methods:
    target_layer.register_forward_hook(forward_hook)
if 'gradcam' in methods:
    target_layer.register_full_backward_hook(backward_hook)

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

def feature_map_vis(x, y, idx, top_k=8):
    """
    Feature Map 可视化
    - 左侧一列：X-ray (第一排) + GT DRR (第二排)
    - 右侧独立网格显示前 top_k 个 feature map
    """
    _ = model(x)
    fmap = activation['value'][0]  # [C,H,W]
    n_channels = min(top_k, fmap.shape[0])
    x_img = x.squeeze().detach().cpu().numpy()
    y_img = y.squeeze().detach().cpu().numpy()

    n_cols = 4
    n_rows = int(np.ceil(n_channels / n_cols))

    fig = plt.figure(figsize=(4*(n_cols+1), 4*(max(n_rows,2))))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(max(n_rows,2), n_cols + 1, figure=fig, wspace=0.1, hspace=0.1)

    # 左侧固定显示 X 和 GT
    ax_x = fig.add_subplot(gs[0,0])
    ax_x.imshow(x_img, cmap='gray')
    ax_x.set_title("Input X-ray")
    ax_x.axis('off')

    ax_y = fig.add_subplot(gs[1,0])
    ax_y.imshow(y_img, cmap='gray')
    ax_y.set_title("GT DRR")
    ax_y.axis('off')

    # 右侧 feature map 网格
    for i in range(n_channels):
        row = i // n_cols
        col = (i % n_cols) + 1  # 第一列留给 X / GT
        ax = fig.add_subplot(gs[row, col])
        fm = fmap[i].detach().cpu().numpy()
        vmin, vmax = np.percentile(fm, [1, 99])
        ax.imshow(fm, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dirs["featuremap"], f"{idx}_featuremap_{target_layer_name}.png")
    plt.savefig(save_path)
    plt.close()



import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os

def gradcam_vis(x, y, idx, img_size=256):
    """
    显示三张图：X-ray原图、目标DRR图、Grad-CAM热力图叠加
    并保存到指定目录。
    
    :param x: 输入X-ray张量, [1, 1, H, W]
    :param y: 对应的DRR张量, [1, 1, H, W]
    :param idx: 图像索引或名称，用于保存文件
    :param img_size: 输出图像大小
    """
    x = x.requires_grad_(True)
    output = model(x)
    
    # L1 loss
    loss = F.l1_loss(output, y)
    model.zero_grad()
    loss.backward()
    
    # 获取梯度和特征图
    grad = gradients['value']      # [1, C, h, w]
    fmap = activation['value']     # [1, C, h, w]
    
    # Grad-CAM
    weights = grad.mean(dim=(2,3), keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(img_size, img_size), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # 原图和目标DRR
    x_img = x.squeeze().cpu().detach().numpy()
    y_img = y.squeeze().cpu().detach().numpy()
    
    # 创建三张图的subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原X-ray
    axes[0].imshow(x_img, cmap='gray')
    axes[0].set_title("X-ray")
    axes[0].axis('off')
    
    # 对应DRR
    axes[1].imshow(y_img, cmap='gray')
    axes[1].set_title("DRR")
    axes[1].axis('off')
    
    # Grad-CAM叠加
    axes[2].imshow(x_img, cmap='gray')
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title("Grad-CAM")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dirs["gradcam"], f"{idx}_gradcam_{target_layer_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def occlusion_vis(x, y, idx, patch_size=8):
    x_np = x.squeeze().cpu().detach().numpy()
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
    save_path = os.path.join(save_dirs["occlusion"], f"{idx}_occlusion_{target_layer_name}.png")
    plt.savefig(save_path)

def output_error_vis(x, y, idx):
    with torch.no_grad():
        out = model(x)
    error = torch.abs(out - y).squeeze().cpu().numpy()
    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)
    x_img = x.squeeze().cpu().numpy()
    y_img = y.squeeze().cpu().numpy()
    out_img = out.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(x_img, cmap='gray'); axes[0].set_title("Input X-ray"); axes[0].axis('off')
    axes[1].imshow(y_img, cmap='gray'); axes[1].set_title("GT DRR"); axes[1].axis('off')
    axes[2].imshow(out_img, cmap='gray'); axes[2].set_title("Generated DRR"); axes[2].axis('off')
    axes[3].imshow(error_norm, cmap='hot'); axes[3].set_title("Absolute Error Map"); axes[3].axis('off')
    plt.tight_layout()
    save_path = os.path.join(save_dirs["output_error"], f"{idx}_output_error.png")
    plt.savefig(save_path)

# ----------------------------
# 批量可视化
# ----------------------------
# 基础保存路径
save_dir = "../visualizations"

# 每种可视化单独文件夹
save_dirs = {
    "featuremap": os.path.join(save_dir, "featuremap"),
    "gradcam": os.path.join(save_dir, "gradcam"),
    "occlusion": os.path.join(save_dir, "occlusion"),
    "output_error": os.path.join(save_dir, "output_error")
}

# 创建文件夹
for d in save_dirs.values():
    os.makedirs(d, exist_ok=True)


for idx, (x, y) in enumerate(loader):
    if idx >= max_samples:
        break
    x = x.to(device)
    y = y.to(device)

    if 'output_error' in methods:
        output_error_vis(x, y, idx)
    if 'featuremap' in methods:
        feature_map_vis(x, y, idx)

    if 'gradcam' in methods:
        gradcam_vis(x, y, idx)
    if 'occlusion' in methods:
        occlusion_vis(x, y, idx)

    print(f"Saved visualizations for sample {idx}")

print("可视化完成，结果保存在:", args.save_dir)
