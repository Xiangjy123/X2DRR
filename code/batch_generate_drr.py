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

def save_drr_as_dicom(
    img_tensor,           # (1, 1, H, W) or (1, H, W) torch tensor
    output_path: str,     # Output DICOM path
    sdd: float,           # Source to detector distance
    delx: float,          # Pixel spacing (mm)
    detector_origin: list, # [x0_mm, y0_mm]
    rows: int,            # Image height (H)
    cols: int             # Image width (W)
):
    # Convert tensor to normalized uint16 with white background
    img_np = img_tensor.squeeze().cpu().numpy()
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    # img_inverted = 1.0 - img_norm  # dark bone, white background
    img_16bit = (img_norm * 65535).astype(np.uint16)

    # Create file meta
    meta = Dataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.XRayAngiographicImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.1"
    meta.ImplementationVersionName = "PYDICOM 2.4.4"

    # Create DICOM dataset
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

    # Image dimensions and type
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned integer
    ds.PixelData = img_16bit.tobytes()

    # Display window
    ds.WindowCenter = 32896
    ds.WindowWidth = 65535

    # Geometry
    ds.DistanceSourceToDetector = "%.10f" % sdd
    ds.DetectorActiveOrigin = ["%.10f" % detector_origin[0], "%.10f" % detector_origin[1]]
    ds.PixelSpacing = ["%.10f" % delx, "%.10f" % delx]

    # Save DICOM
    ds.save_as(output_path)
    print(f"✅ DICOM saved to {output_path}")

def process_single_xray(patient_id, xray_id, data_dir, output_dir, device, crop_size=1436, resize_size=256):
    """
    处理单个 X-ray 图像，生成对应的 DRR
    
    Args:
        patient_id: 病人ID
        xray_id: X-ray ID
        data_dir: 数据根目录
        output_dir: 输出根目录
        device: 计算设备
        crop_size: 裁剪尺寸
        resize_size: 调整尺寸
    """
    # 构建文件路径
    patient_dir = data_dir / f"subject{patient_id:02d}"
    xrays_dir = patient_dir / "xrays"
    volume_path = patient_dir / "volume.nii.gz"
    
    # 加载 .pt 文件（包含 pose 和 intrinsics）
    pt_path = xrays_dir / f"{xray_id:03d}.pt"
    data = torch.load(pt_path, weights_only=False)
    true_pose = data["pose"].to(device)
    
    # 加载 .dcm 文件（DICOM 影像）
    dcm_path = xrays_dir / f"{xray_id:03d}.dcm"
    dicom_img = pydicom.dcmread(dcm_path)
    
    # 提取像素数据并转换为张量
    pixel_array = dicom_img.pixel_array
    pixel_tensor = torch.from_numpy(pixel_array).float()
    
    # 应用中心裁剪
    cropped_tensor = center_crop(pixel_tensor, (crop_size, crop_size))
    
    # Resize 到目标尺寸
    resize_tensor = torch.nn.functional.interpolate(
        cropped_tensor.unsqueeze(0).unsqueeze(0),
        size=(resize_size, resize_size),
        mode='bilinear',
        align_corners=False
    )
    
    # 转换回 numpy 数组
    resized_pixel_array = resize_tensor.squeeze().numpy()
    
    # 加载 CT 体数据（.nii.gz）
    subject = read(str(volume_path), orientation="PA")
    
    # 创建 DRR 渲染器
    drr = DRR(
        subject,
        sdd=1020,
        height=resize_size,
        delx=0.1940000057220459 * crop_size / resize_size,
        renderer="trilinear",
        reverse_x_axis=True,
    ).to(device)
    
    # 生成 DRR 图像
    drr_img = drr(true_pose)


    
    # 创建输出目录结构
    patient_output_dir = output_dir / f"subject{patient_id:02d}"
    xray_output_dir = patient_output_dir / "xrays_png"
    drr_output_dir = patient_output_dir / "DRRs_png"
    dicom_output_dir = patient_output_dir / "DRRs_dicom"
    
    xray_output_dir.mkdir(parents=True, exist_ok=True)
    drr_output_dir.mkdir(parents=True, exist_ok=True)
    dicom_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 X-ray 图像
    xray_path = xray_output_dir / f"xray{xray_id:03d}.png"
    plt.imsave(xray_path, resized_pixel_array, cmap="gray")
    
    # 保存 DRR 图像
    drr_path = drr_output_dir / f"drr{xray_id:03d}.png"
    plt.imsave(drr_path, drr_img.cpu().squeeze().numpy(), cmap="gray")

    #保存DRR dicom
    dicom_path= dicom_output_dir / f"drr{xray_id:03d}.dcm"
    origin_x = -drr.detector.x0 * drr.detector.delx
    origin_y = -drr.detector.y0 * drr.detector.dely
    detector_origin = [origin_x, origin_y]

    save_drr_as_dicom(
        img_tensor=drr_img,  # (1, 1, H, W)
        output_path=dicom_path,
        sdd=1020,
        delx=0.1940000057220459 * crop_size / resize_size,  # as per your metadata
        detector_origin=detector_origin,
        rows=resize_size,  # or img.shape[-2]
        cols=resize_size  # or img.shape[-1]
    )


    return xray_path, drr_path


def batch_process_drr_generation(data_dir="xvr/data/deepfluoro", 
                                  output_dir="xray_drr",
                                  patient_ids=None,
                                  device=None):
    """
    批量处理 DRR 生成
    
    Args:
        data_dir: 数据根目录
        output_dir: 输出根目录
        patient_ids: 要处理的病人ID列表，如果为None则处理所有病人
        device: 计算设备，如果为None则自动选择
    """
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
    
    # 统计信息
    total_xrays = 0
    success_count = 0
    error_count = 0
    
    # 遍历每个病人
    for patient_id in patient_ids:
        patient_dir = data_dir / f"subject{patient_id:02d}"
        xrays_dir = patient_dir / "xrays"
        
        if not xrays_dir.exists():
            print(f"警告: 病人 {patient_id:02d} 的 xrays 目录不存在，跳过")
            continue
        
        # 获取所有 .pt 文件
        pt_files = sorted(xrays_dir.glob("*.pt"))
        xray_ids = [int(f.stem) for f in pt_files]
        
        print(f"\n处理病人 {patient_id:02d}: {len(xray_ids)} 个 X-ray 图像")
        
        # 遍历每个 X-ray
        for xray_id in tqdm(xray_ids, desc=f"Subject {patient_id:02d}", ncols=80):
            try:
                xray_path, drr_path = process_single_xray(
                    patient_id, xray_id, data_dir, output_dir, device
                )
                success_count += 1
                total_xrays += 1
            except Exception as e:
                print(f"\n错误: 处理病人 {patient_id:02d} X-ray {xray_id:03d} 时出错: {e}")
                error_count += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("批量处理完成！")
    print(f"总计处理: {total_xrays} 个 X-ray 图像")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "xvr/data/deepfluoro"
    OUTPUT_DIR = "xvr/data/deepfluoro"

    # 指定要处理的病人ID（None表示处理所有）
    PATIENT_IDS = None  # 或者 [1, 2] 只处理特定病人

    # 运行批量处理
    batch_process_drr_generation(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        patient_ids=PATIENT_IDS
    )

    # data_dir = "xvr/data/deepfluoro"
    # output_dir = "xvr/data/deepfluoro"
    #
    # data_dir = Path(data_dir)
    # output_dir = Path(output_dir)
    #
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # xray_path, drr_path = process_single_xray(
    #     1, 1, data_dir=data_dir, output_dir=output_dir, device=device
    # )
