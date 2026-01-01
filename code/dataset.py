"""
用于加载成对的 X-ray 与 DRR(bone) 图像，
供 PyTorch 训练 X-ray → DRR 模型使用。
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class XrayDRRDataset(Dataset):
    def __init__(self, root_dir, subjects, img_size=256):
        """
        Args:
            root_dir (str): 数据根目录，例如 data/deepfluoro
            subjects (list): ['subject01', 'subject02', ...]
            img_size (int): 输出图像大小（正方形）
        """
        self.pairs = []  # 存储 (xray_path, drr_path)

        for subj in subjects:
            xray_dir = os.path.join(root_dir, subj, "xrays_png")
            drr_dir = os.path.join(root_dir, subj, "DRRs_bone_png")

            if not os.path.isdir(xray_dir):
                raise FileNotFoundError(f"X-ray 目录不存在: {xray_dir}")
            if not os.path.isdir(drr_dir):
                raise FileNotFoundError(f"DRR 目录不存在: {drr_dir}")

            for fname in sorted(os.listdir(xray_dir)):
                if not fname.lower().endswith(".png"):
                    continue

                # xray000.png -> drr000_bone.png
                xray_path = os.path.join(xray_dir, fname)
                index = fname.replace("xray", "").replace(".png", "")
                drr_name = f"drr{index}_bone.png"
                drr_path = os.path.join(drr_dir, drr_name)

                if os.path.exists(drr_path):
                    self.pairs.append((xray_path, drr_path))
                else:
                    print(f"[Warning] 找不到对应 DRR 文件: {drr_path}")

        if len(self.pairs) == 0:
            raise RuntimeError("未找到任何有效的 X-ray / DRR 配对，请检查路径或命名规则。")

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # 映射到 [-1, 1]
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        xray_path, drr_path = self.pairs[idx]

        xray = Image.open(xray_path).convert("L")  # 单通道
        drr = Image.open(drr_path).convert("L")

        xray = self.transform(xray)
        drr = self.transform(drr)

        return xray, drr
