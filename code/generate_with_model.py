"""
generate_with_model.py
使用训练好的模型从X-ray生成DRR
兼容旧UNet和新UNet+AttentionGate模型
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

from models import UNetGenerator  # 确保 models.py 中 UNetGenerator 支持 use_attention 参数

def generate_drr_with_model(subject="subject01", use_attention=True, model_path="checkpoints/G_final.pth"):
    """
    使用训练好的模型生成DRR图像
    支持新旧模型权重兼容：
      - 新模型权重（UNet+AG）完整加载
      - 旧模型权重部分加载，AG模块随机初始化
    Args:
        subject (str): 要生成的subject目录名
        use_attention (bool): 是否使用AttentionGate
        model_path (str or Path): 模型权重路径
    Returns:
        output_dir (Path): 生成的DRR图像保存目录
    """

    print("=" * 60)
    print(f"使用训练好的模型生成DRR - subject: {subject} - AttentionGate: {use_attention}")
    print("=" * 60)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行 train_pix2pix.py 训练模型")
        return None

    print(f"加载模型: {model_path}")
    model = UNetGenerator(use_attention=use_attention).to(device)

    state_dict = torch.load(model_path, map_location=device)

    # 2. 尝试加载权重
    try:
        model.load_state_dict(state_dict)
        print("成功加载完整权重（新模型权重）")
    except RuntimeError as e:
        print("检测到旧模型权重或部分不匹配，尝试部分加载...")
        model.load_state_dict(state_dict, strict=False)
        print("部分加载权重，新增模块将随机初始化")

        # 初始化新增的注意力模块
        def init_weights(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)

    model.eval()  # 设置为评估模式

    # 3. 准备输入数据（X-ray图像）
    xray_dir = Path(f"data/deepfluoro/{subject}/xrays_png")
    if not xray_dir.exists():
        print(f"错误: X-ray目录不存在 {xray_dir}")
        print(f"当前工作目录: {os.getcwd()}")
        return None

    xray_files = list(xray_dir.glob("xray*.png"))
    if not xray_files:
        print(f"错误: 在 {xray_dir} 中未找到X-ray PNG文件")
        return None

    print(f"找到 {len(xray_files)} 个X-ray图像")

    # 4. 创建输出目录
    output_dir = Path(f"generated_by_model/{subject}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. 图像预处理转换（与训练时相同）
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
    ])

    # 6. 批量生成DRR
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
            generated_np = (generated_np + 1) * 127.5
            generated_np = np.clip(generated_np, 0, 255).astype(np.uint8)

            # 保存生成的DRR
            filename = xray_file.name.replace("xray", "gen_drr")
            output_path = output_dir / filename

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
    # 可修改参数进行消融实验
    subject = "subject01"
    use_attention = True  # True: 使用AttentionGate，False: 不使用
    model_path = "checkpoints/G_final.pth"

    output_dir = generate_drr_with_model(subject, use_attention, model_path)
    if output_dir:
        print("\n下一步：运行评估脚本比较生成的DRR与真实DRR")
        print(
            f"命令：python evaluate_drr_quality.py --test_type custom --generated_dir {output_dir} --target_dir ../deepfluoro/{subject}/DRRs_png"
        )
