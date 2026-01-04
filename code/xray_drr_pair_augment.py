"""
xray_drr_pair_augment_multisubject.py

该脚本针对多 subject 的配对 X-ray 图像和 DRR 图像进行数据增强，以便用于像素级监督学习（如 Pix2Pix）。
增强策略如下：
1. 几何增强（仿射变换：旋转、平移、缩放）同步作用于 X-ray 和 DRR，
   保证像素级对齐。
2. 亮度/噪声增强（对比度调整、高斯模糊、高斯噪声）仅作用于 X-ray，
   模拟成像变化而不破坏监督信号。
3. 增强后的图像文件名中包含增强参数和版本号，便于追踪，例如：
   xray000_rot3_trans-5_2_scale1.03_v0.png
   drr000_rot3_trans-5_2_scale1.03_v0.png
4. 每个 subject 可以指定增强倍数，以平衡不同 subject 样本量。
"""
import os
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torchvision.transforms.functional as TF
import numpy as np

# ------------------------
# 配置
# ------------------------
subjects = [f"subject{i:02d}" for i in range(1, 7)]  # subject01 ~ subject06
data_root = Path("data/deepfluoro")
aug_suffix_dir = "_aug"

# 增强倍数（每张原图生成多少增强图）
subject_aug_factor = {
    "subject01": 0,
    "subject02": 0,
    "subject03": 3,
    "subject04": 1,
    "subject05": 1,
    "subject06": 3,
}

# 增强参数
max_translation = 10
max_rotation = 5
scale_range = (0.95, 1.05)
noise_std = 0.01
contrast_range = (0.9, 1.1)
blur_radius = 1.0
invert_xray = False  # 是否对X-ray反转（骨头白背景黑）

# ------------------------
# 辅助函数
# ------------------------
def random_affine_params():
    translate = (random.randint(-max_translation, max_translation),
                 random.randint(-max_translation, max_translation))
    angle = random.uniform(-max_rotation, max_rotation)
    scale = random.uniform(*scale_range)
    return angle, translate, scale

def apply_affine(img, angle, translate, scale):
    return TF.affine(
        img,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=0,
        interpolation=TF.InterpolationMode.BILINEAR
    )

def add_intensity_noise(img):
    # 对比度
    contrast_factor = random.uniform(*contrast_range)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    # 高斯模糊
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # 高斯噪声
    if random.random() < 0.5:
        arr = np.array(img).astype(float) / 255.0
        noise = np.random.normal(0, noise_std, arr.shape)
        arr = arr + noise
        arr = (arr * 255).clip(0, 255).astype('uint8')
        img = Image.fromarray(arr)
    return img

# ------------------------
# 主循环：遍历所有 subject
# ------------------------
for subject in subjects:
    xray_dir = data_root / subject / "xrays_png"
    drr_dir = data_root / subject / "DRRs_bone_png"

    output_xray_dir = data_root / subject / f"xrays{aug_suffix_dir}"
    output_drr_dir = data_root / subject / f"DRRs_bone{aug_suffix_dir}"

    output_xray_dir.mkdir(parents=True, exist_ok=True)
    output_drr_dir.mkdir(parents=True, exist_ok=True)

    xrays_list = sorted(xray_dir.glob("*.png"))
    drrs_list = sorted(drr_dir.glob("*.png"))

    num_aug = subject_aug_factor.get(subject, 1)  # 默认每张生成1张增强图
    print(f"Processing {subject}: {len(xrays_list)} pairs, generating {num_aug} augmentations each...")

    for xray_path, drr_path in zip(xrays_list, drrs_list):
        xray = Image.open(xray_path).convert("L")
        drr = Image.open(drr_path).convert("L")

        # 可选反转
        if invert_xray:
            xray = ImageOps.invert(xray)

        for i in range(num_aug):
            # 生成随机仿射参数
            angle, translate, scale = random_affine_params()

            # 同步几何增强
            xray_aff = apply_affine(xray, angle, translate, scale)
            drr_aff = apply_affine(drr, angle, translate, scale)

            # X-ray 额外强度增强
            xray_final = add_intensity_noise(xray_aff)

            # 构建增强文件名
            aug_suffix = f"rot{int(angle)}_trans{translate[0]}_{translate[1]}_scale{scale:.2f}_v{i}"
            xray_name = xray_path.stem + f"_{aug_suffix}.png"
            drr_name = drr_path.stem + f"_{aug_suffix}.png"

            # 保存
            xray_final.save(output_xray_dir / xray_name)
            drr_aff.save(output_drr_dir / drr_name)

print("All subjects processed successfully!")
