import cv2
import torch
import numpy as np
import os
from pathlib import Path

# 加载模型
model_path = r"E:\a1234\daima\kozhao\MaskDetect-YOLOv4-PyTorch-master (1)\MaskDetect-YOLOv4-PyTorch-master\model_data\yolov4_maskdetect_weights1.pth"

print("正在加载模型...")
print(f"权重文件: {model_path}")
print(f"文件存在: {os.path.exists(model_path)}")

# 如果权重文件不存在，提示用户
if not os.path.exists(model_path):
    print("\n⚠️  YOLOv4 权重文件不存在！")
    print("请先运行 train.py 完成训练。")
else:
    print("\n✅ 找到模型权重文件")
    print("现在你可以:")
    print("1. 运行 GUI: python gui_detect.py")
    print("2. 用 YOLOv5 检测: python detect.py --weights runs/train/mask_detection/weights/best.pt --source 图片路径")
