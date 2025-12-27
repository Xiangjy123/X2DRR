import os
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from diffdrr.data import read
import pydicom
import numpy as np

# -------------------------------
# 1. 将 DICOM 读取并保存为 NIfTI
# -------------------------------
def load_ct_as_nifti(dicom_dir, temp_nii_path):
    reader = sitk.ImageSeriesReader()
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not series_file_names:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    
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
    selected = []
    for sub in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, sub)
        if os.path.isdir(sub_path):
            num_files = sum(os.path.isfile(os.path.join(sub_path, f)) for f in os.listdir(sub_path))
            if num_files >= min_files:
                selected.append(sub_path)
    if not selected:
        selected = [folder_path]
    return selected

# -------------------------------
# 3. 图像归一化 + 对比度拉伸 + 可选伽马 + 背景抑制
# -------------------------------
def enhance_bone_drr(img, lower_percent=1, upper_percent=99, gamma=0.7, background_threshold=0.02):
    """
    img: np.ndarray, DRR 输出
    lower_percent, upper_percent: 百分比裁剪
    gamma: gamma 增强
    background_threshold: 低值背景置0阈值
    """
    # 归一化 + 对比度拉伸
    low, high = np.percentile(img, (lower_percent, upper_percent))
    img_norm = np.clip((img - low) / (high - low), 0, 1)

    # Gamma增强
    if gamma is not None:
        img_norm = img_norm ** gamma

    # 背景抑制
    img_norm[img_norm < background_threshold] = 0
    return img_norm

# -------------------------------
# 4. 提取骨头
# -------------------------------
def extract_bone(nii_path, hu_threshold=250):
    """
    只保留骨头 HU > hu_threshold
    """
    image = sitk.ReadImage(nii_path)
    img_np = sitk.GetArrayFromImage(image)
    bone_mask = img_np > hu_threshold
    img_bone = img_np * bone_mask
    img_bone_sitk = sitk.GetImageFromArray(img_bone)
    img_bone_sitk.CopyInformation(image)
    temp_bone_path = nii_path.replace(".nii.gz", "_bone.nii.gz")
    sitk.WriteImage(img_bone_sitk, temp_bone_path)
    return temp_bone_path

# -------------------------------
# 5. 主程序
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = r"F:\Files\Graduate_Research\X2DRR\data\CT"
nifti_dir = r"F:\Files\Graduate_Research\X2DRR\data\nifti"
os.makedirs(nifti_dir, exist_ok=True)
drr_dir = r"F:\Files\Graduate_Research\X2DRR\data\DRR"
os.makedirs(drr_dir, exist_ok=True)

min_files_threshold = 50

for case_folder in os.listdir(data_root):
    folder_path = os.path.join(data_root, case_folder)
    if not os.path.isdir(folder_path):
        continue

    dicom_dirs = select_dicom_folder(folder_path, min_files=min_files_threshold)

    for dicom_dir in dicom_dirs:
        try:
            print(f"Processing: {dicom_dir}")

            if dicom_dir == folder_path:
                file_name = case_folder
            else:
                subfolder_name = os.path.basename(dicom_dir)
                file_name = f"{case_folder}_{subfolder_name}"

            nii_path = os.path.join(nifti_dir, f"{file_name}.nii.gz")
            nii_path = load_ct_as_nifti(dicom_dir, nii_path)

            # 提取骨头
            bone_nii_path = extract_bone(nii_path, hu_threshold=250)

            # 生成 DiffDRR Subject
            subject = read(
                volume=bone_nii_path,
                orientation="AP"
            )

            drr = DRR(
                subject,
                sdd=2040.0,
                height=600,
                delx=1.0,
            ).to(device)

            rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
            translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)

            img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")

            img_np = img.squeeze().cpu().numpy()

            # 后处理增强骨头
            img_enhanced = enhance_bone_drr(
                img_np,
                lower_percent=1,
                upper_percent=99,
                gamma=0.7,
                background_threshold=0.02
            )

            plt.figure()
            plt.imshow(img_enhanced, cmap='gray')
            plt.axis('off')
            drr_path = os.path.join(drr_dir, f"{file_name}.png")
            plt.savefig(drr_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved enhanced bone DRR: {drr_path}")

        except Exception as e:
            print(f"Skipping {dicom_dir} due to error: {e}")
            continue
