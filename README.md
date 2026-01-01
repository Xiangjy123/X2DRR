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