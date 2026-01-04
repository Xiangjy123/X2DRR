"""
CT-X光特征迁移质量评估脚本
批量评估生成DRR与CT获得的DRR之间的相似度
计算四个核心指标：PSNR, SSIM, LPIPS, NMI
"""

import numpy as np
import torch
import lpips
from scipy.stats import entropy
import cv2
import os
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from datetime import datetime


class DRREvaluator:
    """DRR图像质量评估器"""

    def __init__(self, use_gpu=False, net_type='alex'):
        """
        初始化评估器

        参数：
        use_gpu : bool, 是否使用GPU加速LPIPS计算
        net_type : str, LPIPS网络类型 ('alex', 'vgg', 'squeeze')
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.net_type = net_type

        # 初始化LPIPS模型
        if self.device.type == 'cuda':
            self.loss_fn = lpips.LPIPS(net=net_type, verbose=False).cuda()
        else:
            self.loss_fn = lpips.LPIPS(net=net_type, verbose=False)

        print(f"使用设备: {self.device}")
        print(f"LPIPS网络类型: {net_type}")

    def load_image(self, image_path):
        """
        加载图像并归一化到[0,1]

        参数：
        image_path : str, 图像文件路径

        返回：
        numpy.ndarray : 归一化后的图像数组
        """
        try:
            # 使用OpenCV读取图像（支持多种格式）
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                # 如果OpenCV失败，尝试用PIL
                from PIL import Image
                img = Image.open(image_path).convert('L')
                img = np.array(img)

            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")

            # 归一化到[0,1]
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0

            return img
        except Exception as e:
            print(f"加载图像 {image_path} 时出错: {e}")
            return None

    def compute_psnr(self, img1, img2, max_val=1.0):
        """计算峰值信噪比 (PSNR)"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * np.log10(max_val ** 2 / mse)
        return psnr

    def compute_ssim(self, img1, img2, data_range=1.0):
        """计算结构相似性指数 (SSIM)"""
        # 确保图像形状相同
        if img1.shape != img2.shape:
            # 调整大小为较小的尺寸
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]

        try:
            ssim_val = ssim(img1, img2, data_range=data_range, channel_axis=None)
        except:
            # 备用计算方式
            ssim_val = self._simple_ssim(img1, img2)

        return ssim_val

    def _simple_ssim(self, img1, img2, window_size=11, sigma=1.5):
        """简化的SSIM实现"""
        from scipy.signal import gaussian

        gauss = gaussian(window_size, sigma)
        window = np.outer(gauss, gauss)
        window = window / window.sum()

        pad = window_size // 2
        img1_pad = np.pad(img1, pad, mode='reflect')
        img2_pad = np.pad(img2, pad, mode='reflect')

        ssim_map = np.zeros(img1.shape)

        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                win1 = img1_pad[i:i + window_size, j:j + window_size]
                win2 = img2_pad[i:i + window_size, j:j + window_size]

                mu1 = np.sum(win1 * window)
                mu2 = np.sum(win2 * window)

                sigma1_sq = np.sum(window * (win1 - mu1) ** 2)
                sigma2_sq = np.sum(window * (win2 - mu2) ** 2)
                sigma12 = np.sum(window * (win1 - mu1) * (win2 - mu2))

                C1 = (0.01 * 1.0) ** 2
                C2 = (0.03 * 1.0) ** 2

                ssim_map[i, j] = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                                 ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def compute_lpips(self, img1, img2):
        """计算学习感知图像块相似度 (LPIPS)"""
        # 确保为3通道
        if len(img1.shape) == 2:  # 灰度图
            img1 = np.stack([img1] * 3, axis=0)  # [3, H, W]
            img2 = np.stack([img2] * 3, axis=0)  # [3, H, W]
        elif len(img1.shape) == 3 and img1.shape[2] == 1:  # [H, W, 1]
            img1 = np.repeat(img1, 3, axis=2)
            img2 = np.repeat(img2, 3, axis=2)
            img1 = img1.transpose(2, 0, 1)  # [3, H, W]
            img2 = img2.transpose(2, 0, 1)
        elif len(img1.shape) == 3 and img1.shape[0] == 1:  # [1, H, W]
            img1 = np.repeat(img1, 3, axis=0)  # [3, H, W]
            img2 = np.repeat(img2, 3, axis=0)

        # 转换为tensor并归一化到[-1, 1]
        img1_tensor = torch.FloatTensor(img1).unsqueeze(0)  # [1, 3, H, W]
        img2_tensor = torch.FloatTensor(img2).unsqueeze(0)

        # 从[0,1]转换到[-1,1]
        img1_tensor = img1_tensor * 2 - 1
        img2_tensor = img2_tensor * 2 - 1

        if self.device.type == 'cuda':
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()

        # 计算LPIPS
        with torch.no_grad():
            lpips_val = self.loss_fn(img1_tensor, img2_tensor).item()

        return lpips_val

    def compute_nmi(self, img1, img2, bins=256, normalize_method='average'):
        """计算归一化互信息 (NMI)"""
        flat1 = img1.ravel()
        flat2 = img2.ravel()

        # 归一化到[0,1]
        if flat1.max() > 1.0:
            flat1 = (flat1 - flat1.min()) / (flat1.max() - flat1.min() + 1e-8)
        if flat2.max() > 1.0:
            flat2 = (flat2 - flat2.min()) / (flat2.max() - flat2.min() + 1e-8)

        # 离散化
        flat1 = np.floor(flat1 * (bins - 1)).astype(np.int32)
        flat2 = np.floor(flat2 * (bins - 1)).astype(np.int32)

        # 计算联合直方图
        joint_hist, _, _ = np.histogram2d(flat1, flat2, bins=bins)

        # 计算边缘直方图
        hist1 = np.histogram(flat1, bins=bins)[0]
        hist2 = np.histogram(flat2, bins=bins)[0]

        # 计算概率分布
        p_xy = joint_hist / np.sum(joint_hist)
        p_x = hist1 / np.sum(hist1)
        p_y = hist2 / np.sum(hist2)

        # 计算熵
        h_x = entropy(p_x, base=2)
        h_y = entropy(p_y, base=2)
        h_xy = entropy(p_xy.flatten(), base=2)

        # 计算互信息和NMI
        mi = h_x + h_y - h_xy

        if normalize_method == 'min':
            nmi = mi / min(h_x, h_y) if min(h_x, h_y) > 0 else 0
        else:  # 'average'
            nmi = 2 * mi / (h_x + h_y) if (h_x + h_y) > 0 else 0

        return nmi

    def evaluate_single_pair(self, generated_path, target_path):
        """
        评估一对图像

        参数：
        generated_path : str, 生成的DRR图像路径
        target_path : str, 由CT获得的真实DRR图像路径

        返回：
        dict : 包含四个指标值的字典
        """
        print(f"评估图像对:")
        print(f"  生成图像: {generated_path}")
        print(f"  真实DRR: {target_path}")

        # 加载图像
        generated_img = self.load_image(generated_path)
        target_img = self.load_image(target_path)

        if generated_img is None or target_img is None:
            print("警告：无法加载图像，跳过此对")
            return None

        # 检查图像形状，如不同则调整
        if generated_img.shape != target_img.shape:
            print(f"警告：图像形状不匹配！{generated_img.shape} vs {target_img.shape}")
            # 调整到最小尺寸
            min_h = min(generated_img.shape[0], target_img.shape[0])
            min_w = min(generated_img.shape[1], target_img.shape[1])
            generated_img = generated_img[:min_h, :min_w]
            target_img = target_img[:min_h, :min_w]

        # 计算四个指标
        print("  计算指标中...")

        psnr_value = self.compute_psnr(generated_img, target_img)
        ssim_value = self.compute_ssim(generated_img, target_img)
        lpips_value = self.compute_lpips(generated_img, target_img)
        nmi_value = self.compute_nmi(generated_img, target_img)

        # 返回结果
        results = {
            'generated_path': str(generated_path),
            'target_path': str(target_path),
            'psnr': float(psnr_value),
            'ssim': float(ssim_value),
            'lpips': float(lpips_value),
            'nmi': float(nmi_value)
        }

        return results

    def batch_evaluate(self, generated_dir, target_dir, match_pattern='*.png'):
        """
        批量评估目录下的所有图像

        参数：
        generated_dir : str, 生成的DRR图像目录
        target_dir : str, 真实DRR图像目录
        match_pattern : str, 文件匹配模式

        返回：
        list : 包含所有图像对评估结果的列表
        """
        generated_dir = Path(generated_dir)
        target_dir = Path(target_dir)

        if not generated_dir.exists():
            raise ValueError(f"生成的图像目录不存在: {generated_dir}")
        if not target_dir.exists():
            raise ValueError(f"真实DRR目录不存在: {target_dir}")

        # 获取所有生成的图像文件
        generated_files = list(generated_dir.glob(match_pattern))

        if not generated_files:
            print(f"警告：在 {generated_dir} 中未找到 {match_pattern} 文件")
            return []

        print(f"找到 {len(generated_files)} 个生成的DRR图像")

        all_results = []

        for gen_file in tqdm(generated_files, desc="评估进度"):
            # 构造对应的真实DRR文件名
            # 假设文件名规则：生成的是 xray000.png，真实的是 drr000.png
            # 或者生成的是 drr000_bone.png，真实的是 drr000_bone.png

            # 尝试多种匹配方式
            target_file = None

            # 方式1：直接同名匹配
            target_file = target_dir / gen_file.name

            # 方式2：如果是xray开头的，替换为drr
            if not target_file.exists() and gen_file.name.startswith('xray'):
                target_name = gen_file.name.replace('xray', 'drr')
                target_file = target_dir / target_name

            # 方式3：尝试在目标目录中查找相同序号的文件
            if not target_file.exists():
                # 提取文件序号
                import re
                match = re.search(r'(\d+)', gen_file.stem)
                if match:
                    file_num = match.group(1).zfill(3)  # 补齐3位
                    # 查找目标目录中以drr开头且包含该序号的文件
                    possible_files = list(target_dir.glob(f'*{file_num}*'))
                    if possible_files:
                        target_file = possible_files[0]

            if not target_file.exists():
                print(f"警告：找不到与 {gen_file.name} 对应的真实DRR图像")
                continue

            # 评估这一对
            results = self.evaluate_single_pair(gen_file, target_file)
            if results:
                all_results.append(results)

        print(f"成功评估 {len(all_results)} 对图像")
        return all_results

    def save_results(self, results, output_dir='evaluation_results'):
        """
        保存评估结果到文件和可视化

        参数：
        results : list, 评估结果列表
        output_dir : str, 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存为JSON
        json_path = output_dir / f"evaluation_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存为CSV
        if results:
            df = pd.DataFrame(results)
            csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')

            # 计算统计信息
            stats = {
                'total_pairs': len(results),
                'psnr_mean': df['psnr'].mean(),
                'psnr_std': df['psnr'].std(),
                'ssim_mean': df['ssim'].mean(),
                'ssim_std': df['ssim'].std(),
                'lpips_mean': df['lpips'].mean(),
                'lpips_std': df['lpips'].std(),
                'nmi_mean': df['nmi'].mean(),
                'nmi_std': df['nmi'].std(),
            }

            # 保存统计信息
            stats_path = output_dir / f"evaluation_stats_{timestamp}.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            # 生成可视化图表
            self.plot_results(df, output_dir, timestamp)

            print(f"\n评估结果已保存到: {output_dir}")
            print(f"JSON文件: {json_path}")
            print(f"CSV文件: {csv_path}")
            print(f"统计信息: {stats_path}")

            # 打印统计摘要
            self.print_statistics(stats)

        return output_dir

    def plot_results(self, df, output_dir, timestamp):
        """生成评估结果的可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('DRR生成质量评估结果', fontsize=16)

        # 1. PSNR分布
        axes[0, 0].hist(df['psnr'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df['psnr'].mean(), color='red', linestyle='--', label=f'均值: {df["psnr"].mean():.2f}')
        axes[0, 0].set_xlabel('PSNR (dB)')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('PSNR分布')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. SSIM分布
        axes[0, 1].hist(df['ssim'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(df['ssim'].mean(), color='red', linestyle='--', label=f'均值: {df["ssim"].mean():.4f}')
        axes[0, 1].set_xlabel('SSIM')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('SSIM分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. LPIPS分布
        axes[1, 0].hist(df['lpips'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 0].axvline(df['lpips'].mean(), color='red', linestyle='--', label=f'均值: {df["lpips"].mean():.4f}')
        axes[1, 0].set_xlabel('LPIPS')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('LPIPS分布（值越低越好）')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. NMI分布
        axes[1, 1].hist(df['nmi'], bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].axvline(df['nmi'].mean(), color='red', linestyle='--', label=f'均值: {df["nmi"].mean():.4f}')
        axes[1, 1].set_xlabel('NMI')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('NMI分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def print_statistics(self, stats):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("评估统计摘要")
        print("="*60)
        print(f"评估图像对总数: {stats['total_pairs']}")
        print("\n各指标平均值 ± 标准差:")
        print(f"  PSNR:  {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB")
        print(f"  SSIM:  {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")
        print(f"  LPIPS: {stats['lpips_mean']:.4f} ± {stats['lpips_std']:.4f}")
        print(f"  NMI:   {stats['nmi_mean']:.4f} ± {stats['nmi_std']:.4f}")
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量评估生成DRR与真实DRR的质量')

    parser.add_argument('--generated_dir', type=str,
                       default='generated_by_model/subject01',
                       help='生成的DRR图像目录')

    parser.add_argument('--target_dir', type=str,
                       default='data/deepfluoro/subject01/DRRs_bone_png',
                       help='真实DRR图像目录（由CT获得）')

    parser.add_argument('--output_dir', type=str,
                       default='evaluation_results2',
                       help='评估结果输出目录')

    parser.add_argument('--use_gpu', action='store_true',
                       help='使用GPU加速LPIPS计算')

    parser.add_argument('--net_type', type=str, default='alex',
                       choices=['alex', 'vgg', 'squeeze'],
                       help='LPIPS网络类型')

    parser.add_argument('--match_pattern', type=str, default='*.png',
                       help='文件匹配模式')

    parser.add_argument('--subjects', type=str, nargs='+',
                       help='要评估的subject列表，如subject01 subject02')

    args = parser.parse_args()

    print("="*60)
    print("DRR生成质量批量评估")
    print("="*60)

    # 初始化评估器
    evaluator = DRREvaluator(use_gpu=args.use_gpu, net_type=args.net_type)

    all_results = []

    if args.subjects:
        # 批量评估多个subject
        for subject in args.subjects:
            print(f"\n评估 {subject}...")

            generated_path = Path(args.generated_dir).parent.parent / subject / "xrays_png"
            target_path = Path(args.target_dir).parent.parent / subject / "DRRs_png"

            if not generated_path.exists():
                print(f"警告：{generated_path} 不存在，跳过")
                continue

            if not target_path.exists():
                print(f"警告：{target_path} 不存在，跳过")
                continue

            subject_results = evaluator.batch_evaluate(
                generated_path, target_path, args.match_pattern
            )
            all_results.extend(subject_results)
    else:
        # 评估单个目录
        all_results = evaluator.batch_evaluate(
            args.generated_dir, args.target_dir, args.match_pattern
        )

    if all_results:
        # 保存结果
        output_dir = evaluator.save_results(all_results, args.output_dir)

        # 打印结果解释
        print("\n" + "="*60)
        print("结果解释:")
        print("="*60)
        print("1. PSNR (峰值信噪比):")
        print("   >35 dB: 极好, >30 dB: 很好, >25 dB: 良好, >20 dB: 一般, <20 dB: 较差")

        print("\n2. SSIM (结构相似性):")
        print("   >0.95: 极好, >0.90: 很好, >0.85: 良好, >0.80: 一般, <0.80: 较差")

        print("\n3. LPIPS (感知相似度):")
        print("   <0.15: 极好, <0.25: 很好, <0.35: 良好, <0.45: 一般, >0.45: 较差")
        print("   (注意: LPIPS值越低越好)")

        print("\n4. NMI (归一化互信息):")
        print("   >0.80: 极好, >0.70: 很好, >0.60: 良好, >0.50: 一般, <0.50: 较差")

        print("\n总体评估:")
        print("如果所有指标都在'良好'以上，则生成质量优秀；")
        print("如果3个指标在'良好'以上，则生成质量良好；")
        print("如果2个指标在'良好'以上，则需要改进；")
        print("如果1个或更少指标在'良好'以上，则生成质量较差。")
    else:
        print("\n警告：未评估到任何有效的图像对")


if __name__ == "__main__":
    main()