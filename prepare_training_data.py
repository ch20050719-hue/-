import os
import random

SOURCE_FILE = r'E:\a1234\daima\kozhao\MaskDatasets\datasets\train\mask_train.txt'
TARGET_DIR = r'E:\a1234\daima\kozhao\MaskDetect-YOLOv4-PyTorch-master (1)\MaskDetect-YOLOv4-PyTorch-master\model_data'
VAL_RATIO = 0.1

with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.shuffle(lines)
num_val = max(1, int(len(lines) * VAL_RATIO))
val_lines = lines[:num_val]
train_lines = lines[num_val:]

train_path = os.path.join(TARGET_DIR, 'mask_train.txt')
val_path = os.path.join(TARGET_DIR, 'mask_val.txt')

with open(train_path, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open(val_path, 'w', encoding='utf-8') as f:
    f.writelines(val_lines)

print(f"训练集: {len(train_lines)} 条 -> {train_path}")
print(f"验证集: {len(val_lines)} 条 -> {val_path}")
