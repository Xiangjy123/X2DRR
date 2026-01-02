# X2DRR
根据X射线图生成降噪后的DRR图像。

## 使用指南
直接把数据集`data`放在根目录下即可

## 可运行脚本说明
### DiffDRR DRR Generation
`batch_generate_drr.py`
用于批量生成数字重建影像(DRR)的脚本。

**功能：**  
从CT数据中批量生成DRR影像  
支持多种参数配置  
可选择基于骨骼或软组织生成DRR

**常用参数：**  
`--patients`: 指定处理的病人数量  
`--use_bone`: 是否使用骨骼信息（0-软组织，1-骨骼）  
`--hu_threshold`: HU值阈值，用于组织分割  
### train
`train_pix2pix.py`  
用于训练pix2pix生成对抗网络的脚本。

**功能：**  
训练DRR到CT的pix2pix生成模型  
支持多任务损失函数  
包含多种评估和可视化功能  
主要损失函数：  
　　L1损失 (像素级重建)  
　　边缘损失 (结构保持)  
　　SSIM损失 (结构相似性)  
　　感知损失 (特征匹配)

**常用参数：**  
`--epochs`: 训练轮数  
`--batch_size`: 批大小  
`--lr`: 学习率  
`--lambda_l1`: L1损失权重  
`--lambda_edge`: 边缘损失权重  
`--lambda_ssim`: SSIM损失权重  
`--lambda_perc`: 感知损失权重  

## 项目架构
```
.
├── .vscode/                 # VS Code 调试配置
├── checkpoints/             # 训练模型权重
├── code/                    # 核心代码
├── data/                    # 数据集（详细见下文）
├── outputs/                 # 实验结果与可视化
├── visualizations/          # 中间特征与误差可视化
└── README.md
```

## 数据集架构
```
data/
└── deepfluoro/
    ├── subject01/
    │   ├── DRRs_bone_dicom/ # DICOM格式的骨增强DRR
    │   ├── DRRs_bone_png/   # png格式的骨增强DRR
    │   ├── DRRs_dicom/      # DICOM格式的DRR
    │   ├── DRRs_png/        # png格式的DRR
    │   ├── xrays/           # DICOM格式的x-ray及pose
    │   ├── xrays_png/       # png格式的x-ray
    │   └── ...
    ├── subject02/
    │   └── ...
    ├── ...
    │   └── ...
    └── subject06/
        └── ...
```
## 可视化
1. Feature Map 可视化  
可视化 U-Net 编码器指定中间层的特征图  
可以看到模型提取的不同通道特征，观察关注区域  

2. Grad-CAM 可视化  
基于梯度生成热力图，叠加在输入 X-ray 上  
显示模型对预测输出贡献最大的区域  

3. Occlusion 可视化  
将输入分块遮挡，观察输出变化  
生成热力图，标出对模型输出敏感的区域  

4. 输出误差可视化（Output Error）  
比较生成的 DRR 与真实 DRR 的像素级差异  
显示输入 X-ray、GT DRR、生成 DRR 和绝对误差图  