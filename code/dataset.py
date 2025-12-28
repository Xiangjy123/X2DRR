import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class XrayDRRDataset(Dataset):
    def __init__(self, root_dir, subjects, img_size=256):
        """
        root_dir: 数据根目录，例如 F:/Files/Graduate_Research/X2DRR/data/xray_drr
        subjects: ['subject01', 'subject02', ...]
        """
        self.pairs = [] # 存储 (xray_path, drr_path) 对

        for subj in subjects:
            xray_dir = os.path.join(root_dir, subj, "xray")
            drr_dir = os.path.join(root_dir, subj, "DRR")

            for fname in os.listdir(xray_dir):
                if fname.lower().endswith(".png"):
                    xray_path = os.path.join(xray_dir, fname)
                    drr_name = fname.lower().replace("xray", "drr")  # 关键：xray000.png → drr000.png
                    drr_path = os.path.join(drr_dir, drr_name)
                    if os.path.exists(drr_path):
                        self.pairs.append((xray_path, drr_path))
                    else:
                        print(f"找不到对应 DRR 文件: {drr_path}")


        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # [-1, 1]
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        xray_path, drr_path = self.pairs[idx]

        xray = Image.open(xray_path).convert("L") # 灰度图
        drr = Image.open(drr_path).convert("L")

        xray = self.transform(xray)
        drr = self.transform(drr)

        return xray, drr
