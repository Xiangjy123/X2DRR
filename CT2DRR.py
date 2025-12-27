import os
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import numpy as np

from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from diffdrr.data import read

def load_ct_as_nifti(dicom_dir, temp_nii_path="temp_ct.nii.gz"):
    """
    先用 SimpleITK 读取 DICOM，再保存为 NIfTI，
    之后交由 diffdrr.data.read() 处理。
    """
    reader = sitk.ImageSeriesReader()
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not series_file_names:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    reader.SetFileNames(series_file_names) # 把文件名列表传给 reader
    image = reader.Execute() # SimpleITK把切片堆叠成3D图像

    sitk.WriteImage(image, temp_nii_path) # 保存为 NIfTI 文件

    return temp_nii_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集路径，里面有若干文件夹，每个文件夹可能包含一例或多例 CT
data_root = r"F:\Files\Graduate_Research\X2DRR\data\CT"

# 新建 NIfTI 输出文件夹
nifti_dir = r"F:\Files\Graduate_Research\X2DRR\data\nifti"
os.makedirs(nifti_dir, exist_ok=True)

# 新建 DRR 输出文件夹
drr_dir = r"F:\Files\Graduate_Research\X2DRR\data\DRR"
os.makedirs(drr_dir, exist_ok=True)

# 遍历每一例 CT 文件夹
for case_folder in os.listdir(data_root):
    folder_path = os.path.join(data_root, case_folder)

    # 检查是否为多例子文件夹
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if subfolders:
        # 如果有子文件夹，选择文件数量最多的那个子文件夹
        max_files = 0
        selected_subfolder = None
        for sub in subfolders:
            sub_path = os.path.join(folder_path, sub)
            num_files = sum(os.path.isfile(os.path.join(sub_path, f)) for f in os.listdir(sub_path))
            if num_files > max_files:
                max_files = num_files
                selected_subfolder = sub_path
        dicom_dir = selected_subfolder
    else:
        # 否则直接使用当前文件夹
        dicom_dir = folder_path

    # 构造 NIfTI 文件保存路径，使用原 CT 文件夹名
    nii_path = os.path.join(nifti_dir, f"{case_folder}.nii.gz")

    # 1. 读取 CT 并保存为临时 NIfTI
    nii_path = load_ct_as_nifti(dicom_dir, nii_path)

    # 2. 用官方 read() 构造 Subject（包含 reorient 等）
    subject = read(
        volume=nii_path,
        orientation="AP"  # 如果你有特定方向可修改
    )

    # 3. 初始化 DRR 渲染器
    drr = DRR(
        subject,
        sdd=1020.0,
        height=400,
        delx=1.0,
    ).to(device)

    # 4. 设置相机旋转和平移
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)

    # 5. 渲染并保存 DRR 图片
    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
    plt.figure()
    plot_drr(img, ticks=False)
    # 保存到 DRR 文件夹
    drr_path = os.path.join(drr_dir, f"{case_folder}.png")
    plt.savefig(drr_path, bbox_inches='tight', pad_inches=0)
    plt.close()
