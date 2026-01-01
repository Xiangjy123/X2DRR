"""
基于 DiffDRR 从 CT 体数据生成与 X-ray 位姿一致的 DRR，
支持原 DRR + 骨增强 DRR，并输出 PNG 与 DICOM。
可通过命令行参数控制处理病人、骨增强开关及骨阈值。
"""

import torch
import pydicom
from torchvision.transforms.functional import center_crop
from diffdrr.drr import DRR
from diffdrr.data import read
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
from pydicom.dataset import Dataset, FileDataset
import datetime
import SimpleITK as sitk
import argparse

# ===============================
# 骨提取函数
# ===============================
def extract_bone_from_ct(nii_path: str, hu_threshold: int = 250) -> str:
    """
    从 CT NIfTI 图像中提取骨结构（HU > hu_threshold），保存为新 NIfTI 文件
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

# ===============================
# DICOM 保存函数
# ===============================
def save_drr_as_dicom(
    img_tensor, output_path: str, sdd: float, delx: float,
    detector_origin: list, rows: int, cols: int
):
    img_np = img_tensor.squeeze().cpu().numpy()
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    img_16bit = (img_norm * 65535).astype(np.uint16)

    meta = Dataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.XRayAngiographicImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.1"
    meta.ImplementationVersionName = "PYDICOM 2.4.4"

    ds = FileDataset(output_path, {}, file_meta=meta, preamble=b"\0" * 128)
    dt = datetime.datetime.now()
    ds.StudyDate = dt.strftime("%Y%m%d")
    ds.StudyTime = dt.strftime("%H%M%S")
    ds.Modality = "OT"
    ds.PatientName = "Synthetic"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = meta.MediaStorageSOPClassUID

    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = img_16bit.tobytes()
    ds.WindowCenter = 32896
    ds.WindowWidth = 65535

    ds.DistanceSourceToDetector = "%.10f" % sdd
    ds.DetectorActiveOrigin = ["%.10f" % detector_origin[0], "%.10f" % detector_origin[1]]
    ds.PixelSpacing = ["%.10f" % delx, "%.10f" % delx]

    ds.save_as(output_path)

# ===============================
# 处理单个 X-ray
# ===============================
def process_single_xray(patient_id, xray_id, data_dir, output_dir, device,
                        crop_size=1436, resize_size=256, hu_threshold=250, use_bone=True):
    patient_dir = data_dir / f"subject{patient_id:02d}"
    xrays_dir = patient_dir / "xrays"
    volume_path = patient_dir / "volume.nii.gz"
    
    # 位姿
    pt_path = xrays_dir / f"{xray_id:03d}.pt"
    data = torch.load(pt_path, weights_only=False)
    true_pose = data["pose"].to(device)
    
    # X-ray 裁剪 + resize
    dcm_path = xrays_dir / f"{xray_id:03d}.dcm"
    dicom_img = pydicom.dcmread(dcm_path)
    pixel_array = dicom_img.pixel_array
    pixel_tensor = torch.from_numpy(pixel_array).float()
    cropped_tensor = center_crop(pixel_tensor, (crop_size, crop_size))
    resize_tensor = torch.nn.functional.interpolate(
        cropped_tensor.unsqueeze(0).unsqueeze(0),
        size=(resize_size, resize_size),
        mode='bilinear',
        align_corners=False
    )
    resized_pixel_array = resize_tensor.squeeze().numpy()
    
    # 输出目录
    patient_output_dir = output_dir / f"subject{patient_id:02d}"
    xray_output_dir = patient_output_dir / "xrays_png"
    drr_output_dir = patient_output_dir / "DRRs_png"
    dicom_output_dir = patient_output_dir / "DRRs_dicom"
    xray_output_dir.mkdir(parents=True, exist_ok=True)
    drr_output_dir.mkdir(parents=True, exist_ok=True)
    dicom_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 X-ray PNG
    xray_path = xray_output_dir / f"xray{xray_id:03d}.png"
    plt.imsave(xray_path, resized_pixel_array, cmap="gray")
    
    # -------- 原 DRR --------
    subject = read(str(volume_path), orientation="PA")
    drr = DRR(
        subject,
        sdd=1020,
        height=resize_size,
        delx=0.1940000057220459 * crop_size / resize_size,
        renderer="trilinear",
        reverse_x_axis=True,
    ).to(device)
    drr_img = drr(true_pose)
    
    # 保存 DRR PNG
    drr_path = drr_output_dir / f"drr{xray_id:03d}.png"
    plt.imsave(drr_path, drr_img.cpu().squeeze().numpy(), cmap="gray")
    
    # 保存 DRR DICOM
    dicom_path = dicom_output_dir / f"drr{xray_id:03d}.dcm"
    origin_x = -drr.detector.x0 * drr.detector.delx
    origin_y = -drr.detector.y0 * drr.detector.dely
    save_drr_as_dicom(drr_img, dicom_path, 1020,
                      0.1940000057220459 * crop_size / resize_size,
                      [origin_x, origin_y],
                      resize_size, resize_size)
    
    # -------- 骨增强 DRR --------
    if use_bone:
        bone_volume_path = extract_bone_from_ct(str(volume_path), hu_threshold)
        subject_bone = read(str(bone_volume_path), orientation="PA")
        drr_bone = DRR(
            subject_bone,
            sdd=1020,
            height=resize_size,
            delx=0.1940000057220459 * crop_size / resize_size,
            renderer="trilinear",
            reverse_x_axis=True,
        ).to(device)
        drr_bone_img = drr_bone(true_pose)
        
        # 保存骨增强 PNG
        drr_bone_png_dir = patient_output_dir / "DRRs_bone_png"
        drr_bone_png_dir.mkdir(parents=True, exist_ok=True)
        drr_bone_path = drr_bone_png_dir / f"drr{xray_id:03d}_bone.png"
        plt.imsave(drr_bone_path, drr_bone_img.cpu().squeeze().numpy(), cmap="gray")
        
        # 保存骨增强 DICOM
        drr_bone_dcm_dir = patient_output_dir / "DRRs_bone_dicom"
        drr_bone_dcm_dir.mkdir(parents=True, exist_ok=True)
        dicom_path_bone = drr_bone_dcm_dir / f"drr{xray_id:03d}_bone.dcm"
        origin_x = -drr_bone.detector.x0 * drr_bone.detector.delx
        origin_y = -drr_bone.detector.y0 * drr_bone.detector.dely
        save_drr_as_dicom(drr_bone_img, dicom_path_bone, 1020,
                          0.1940000057220459 * crop_size / resize_size,
                          [origin_x, origin_y],
                          resize_size, resize_size)
    else:
        drr_bone_path = None
    
    return xray_path, drr_path, drr_bone_path

# ===============================
# 批量处理函数
# ===============================
def batch_process_drr_generation(data_dir="data/deepfluoro", 
                                 output_dir="xray_drr",
                                 patient_ids=None,
                                 device=None,
                                 use_bone=False,
                                 hu_threshold=250):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # 自动选择设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 60)
    
    # 自动检测所有病人
    if patient_ids is None:
        patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("subject")])
        patient_ids = [int(d.name.replace("subject", "")) for d in patient_dirs]
    
    print(f"找到 {len(patient_ids)} 个病人: {patient_ids}")
    print("-" * 60)
    
    total_xrays = 0
    success_count = 0
    error_count = 0
    
    for patient_id in patient_ids:
        patient_dir = data_dir / f"subject{patient_id:02d}"
        xrays_dir = patient_dir / "xrays"
        
        if not xrays_dir.exists():
            print(f"警告: 病人 {patient_id:02d} 的 xrays 目录不存在，跳过")
            continue
        
        pt_files = sorted(xrays_dir.glob("*.pt"))
        xray_ids = [int(f.stem) for f in pt_files]
        
        print(f"\n处理病人 {patient_id:02d}: {len(xray_ids)} 个 X-ray 图像")
        
        for xray_id in tqdm(xray_ids, desc=f"Subject {patient_id:02d}", ncols=80):
            try:
                process_single_xray(
                    patient_id, xray_id, data_dir, output_dir, device,
                    use_bone=use_bone,
                    hu_threshold=hu_threshold
                )
                success_count += 1
                total_xrays += 1
            except Exception as e:
                print(f"\n错误: 处理病人 {patient_id:02d} X-ray {xray_id:03d} 时出错: {e}")
                error_count += 1
    
    print("\n" + "=" * 60)
    print("批量处理完成！")
    print(f"总计处理: {total_xrays} 个 X-ray 图像")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

# ===============================
# 命令行入口
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=int, nargs="+", default=None)
    parser.add_argument("--use_bone", type=int, default=0, help="是否生成骨增强 DRR，1=是, 0=否")
    parser.add_argument("--hu_threshold", type=int, default=250, help="骨增强阈值 HU")
    parser.add_argument("--data_dir", type=str, default="data/deepfluoro")
    parser.add_argument("--output_dir", type=str, default="data/deepfluoro")
    args = parser.parse_args()
    
    use_bone_flag = bool(args.use_bone)
    
    batch_process_drr_generation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        patient_ids=args.patients,
        use_bone=use_bone_flag,
        hu_threshold=args.hu_threshold
    )
