""" 根据CT DICOM生成DRR图像 """
import os
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from diffdrr.data import read
import pydicom
import numpy as np
from pathlib import Path

# -------------------------------
# 1. 将 DICOM 读取并保存为 NIfTI
# -------------------------------
def load_ct_as_nifti(dicom_dir, temp_nii_path):
    """
    读取 DICOM 文件夹并保存为 NIfTI
    """
    reader = sitk.ImageSeriesReader()
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not series_file_names:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    # 过滤非 CT 文件
    valid_files = []
    for f in series_file_names:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if ds.Modality == "CT":
                valid_files.append(f)
        except:
            continue

    if not valid_files:
        raise ValueError(f"No valid CT DICOM files in {dicom_dir}")

    reader.SetFileNames(valid_files)
    image = reader.Execute()
    sitk.WriteImage(image, temp_nii_path)
    return temp_nii_path

# -------------------------------
# 2. 遍历数据集，选择文件数大于阈值的子文件夹
# -------------------------------
def select_dicom_folder(folder_path, min_files=50):
    """
    遍历 folder_path 下的子文件夹，返回文件数 >= min_files 的文件夹列表
    """
    selected = []
    for sub in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, sub)
        if os.path.isdir(sub_path):
            num_files = sum(os.path.isfile(os.path.join(sub_path, f)) for f in os.listdir(sub_path))
            if num_files >= min_files:
                selected.append(sub_path)
    # 如果没有符合条件的子文件夹，则直接使用根目录
    if not selected:
        selected = [folder_path]
    return selected

# -------------------------------
# 3. 图像归一化 + 对比度拉伸 + 可选伽马
# -------------------------------
def normalize_contrast(img, lower_percent=2, upper_percent=98, gamma=None):
    """
    img: np.ndarray
    lower_percent, upper_percent: 百分比裁剪范围
    gamma: 可选，伽马校正
    返回归一化到 0~1 的图像
    """
    low, high = np.percentile(img, (lower_percent, upper_percent))
    img = np.clip((img - low) / (high - low), 0, 1)
    if gamma is not None:
        img = img ** gamma
    return img

# -------------------------------
# 4. 主程序
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 项目根目录：X2DRR
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据集路径
data_root = BASE_DIR / "data" / "CT"

# 输出文件夹
nifti_dir = BASE_DIR / "data" / "nifti"
drr_dir = BASE_DIR / "data" / "DRR"

nifti_dir.mkdir(parents=True, exist_ok=True)
drr_dir.mkdir(parents=True, exist_ok=True)

# 阈值：只处理文件数大于该值的文件夹
min_files_threshold = 50

# 遍历每个病例文件夹
for case_folder in os.listdir(data_root):
    folder_path = os.path.join(data_root, case_folder)
    if not os.path.isdir(folder_path):
        continue

    # 获取符合阈值的子文件夹
    dicom_dirs = select_dicom_folder(folder_path, min_files=min_files_threshold)

    for dicom_dir in dicom_dirs:
        try:
            print(f"Processing: {dicom_dir}")

            # 构造文件名
            if dicom_dir == folder_path:
                # 没有子文件夹，直接用病例文件夹名
                file_name = case_folder
            else:
                # 有子文件夹，用病例名 + 子文件夹名
                subfolder_name = os.path.basename(dicom_dir)
                file_name = f"{case_folder}_{subfolder_name}"

            # 构造 NIfTI 文件路径
            nii_path = os.path.join(nifti_dir, f"{file_name}.nii.gz")

            # 1. 读取 DICOM 并保存为 NIfTI
            nii_path = load_ct_as_nifti(dicom_dir, nii_path)

            # 2. 读取 NIfTI 生成 DiffDRR Subject
            subject = read(
                volume=nii_path,
                orientation="AP"  # 可根据你的数据修改
            )

            # 3. 初始化 DRR 渲染器
            drr = DRR(
                subject,
                sdd=2040.0,
                height=600,
                delx=1.0,
            ).to(device)

            # 4. 设置相机旋转和平移
            rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
            translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)

            # 5. 渲染 DRR
            img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")

            # 6. 转为 numpy 并归一化 + 对比度拉伸
            img_np = img.squeeze().cpu().numpy()
            img_norm = normalize_contrast(img_np, lower_percent=2, upper_percent=98, gamma=None)

            # 7. 保存 DRR 图片
            plt.figure()
            plt.imshow(img_norm, cmap='gray')
            plt.axis('off')
            drr_path = os.path.join(drr_dir, f"{file_name}.png")
            plt.savefig(drr_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved DRR: {drr_path}")

        except Exception as e:
            print(f"Skipping {dicom_dir} due to error: {e}")
            continue
