"""
使用训练好的模型从X-ray生成DRR
"""
import torch
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import sys

# 添加当前目录到路径，以便导入models
sys.path.append(str(Path(__file__).parent))

from models import UNetGenerator


def generate_drr_with_model():
    print("=" * 60)
    print("使用训练好的模型生成DRR")
    print("=" * 60)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载训练好的模型
    model_path = Path("checkpoints/G_final.pth")
    if not model_path.exists():
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行 train_pix2pix.py 训练模型")
        return None

    print(f"加载模型: {model_path}")
    model = UNetGenerator().to(device)

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

    model.eval()  # 设置为评估模式

    # 2. 准备输入数据（X-ray图像）
    xray_dir = Path("data/deepfluoro/subject01/xrays_png")
    if not xray_dir.exists():
        print(f"错误: X-ray目录不存在 {xray_dir}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"尝试的路径: {xray_dir.absolute()}")
        return None

    xray_files = list(xray_dir.glob("xray*.png"))
    if not xray_files:
        print(f"错误: 在 {xray_dir} 中未找到X-ray PNG文件")
        print(f"存在的文件: {list(xray_dir.glob('*'))}")
        return None

    print(f"找到 {len(xray_files)} 个X-ray图像")

    # 3. 创建输出目录
    output_dir = Path("generated_by_model/subject01")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 图像预处理转换（与训练时相同）
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
    ])

    # 5. 批量生成DRR
    print("\n开始生成DRR图像...")
    generated_count = 0

    for i, xray_file in enumerate(xray_files):
        try:
            # 加载X-ray图像
            xray_img = Image.open(xray_file).convert("L")
            xray_tensor = transform(xray_img).unsqueeze(0).to(device)  # [1,1,256,256]

            # 使用模型生成DRR
            with torch.no_grad():
                generated_drr = model(xray_tensor)

            # 将张量转回图像
            generated_np = generated_drr.squeeze().cpu().numpy()

            # 从[-1,1]转换到[0,255]
            generated_np = (generated_np + 1) * 127.5  # [-1,1] -> [0,255]
            generated_np = np.clip(generated_np, 0, 255).astype(np.uint8)

            # 保存生成的DRR
            # 文件名对应：xray000.png -> gen_drr000.png
            filename = xray_file.name.replace("xray", "gen_drr")
            output_path = output_dir / filename

            # 使用matplotlib保存（保持与训练时相同的格式）
            plt.imsave(output_path, generated_np, cmap="gray")

            generated_count += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(xray_files):
                print(f"  已生成 {i + 1}/{len(xray_files)}: {filename}")

        except Exception as e:
            print(f"  处理 {xray_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n生成完成！结果保存在: {output_dir.absolute()}")
    print(f"总共生成 {generated_count} 张DRR图像")

    return output_dir


if __name__ == "__main__":
    output_dir = generate_drr_with_model()
    if output_dir:
        print("\n下一步：运行评估脚本比较生成的DRR与真实DRR")
        print(
            f"命令：python evaluate_drr_quality.py --test_type custom --generated_dir {output_dir} --target_dir ../deepfluoro/subject01/DRRs_png")