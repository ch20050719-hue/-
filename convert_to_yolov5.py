
import os
import shutil
from pathlib import Path
from PIL import Image

def convert_to_yolov5():
    # 创建 YOLOv5 目录结构
    base_dir = Path(r"E:\a1234\daima\kozhao\yolov5_dataset")
    (base_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (base_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (base_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # 类别列表
    classes = ["mask", "nomask"]
    
    # 处理训练集
    train_txt = r"E:\a1234\daima\kozhao\MaskDetect-YOLOv4-PyTorch-master (1)\MaskDetect-YOLOv4-PyTorch-master\model_data\mask_train.txt"
    process_dataset(train_txt, base_dir, "train", classes)
    
    # 处理验证集
    val_txt = r"E:\a1234\daima\kozhao\MaskDetect-YOLOv4-PyTorch-master (1)\MaskDetect-YOLOv4-PyTorch-master\model_data\mask_val.txt"
    process_dataset(val_txt, base_dir, "val", classes)
    
    # 创建 data.yaml
    create_data_yaml(base_dir, classes)
    
    print(f"\n转换完成！数据保存在: {base_dir}")

def process_dataset(txt_file, base_dir, split, classes):
    print(f"正在处理 {split} 集...")
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        img_path = parts[0]
        
        # 复制图片
        img_name = Path(img_path).name
        dest_img = base_dir / "images" / split / img_name
        
        if not Path(img_path).exists():
            print(f"警告: 图片不存在 {img_path}")
            continue
            
        shutil.copy2(img_path, dest_img)
        
        # 创建标签文件
        label_name = Path(img_name).stem + '.txt'
        label_path = base_dir / "labels" / split / label_name
        
        # 读取图片尺寸
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except:
            print(f"警告: 无法读取图片 {img_path}")
            continue
        
        # 写入标签
        with open(label_path, 'w') as f:
            for box_str in parts[1:]:
                box = list(map(int, box_str.split(',')))
                x1, y1, x2, y2, cls = box
                
                # 转换为 YOLOv5 格式 (归一化中心坐标和宽高)
                x_center = (x1 + x2) / 2.0 / img_w
                y_center = (y1 + y2) / 2.0 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                # 确保在 [0, 1] 范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        count += 1
    
    print(f"{split} 集: {count} 张图片")

def create_data_yaml(base_dir, classes):
    yaml_content = f'''# 口罩检测数据集
path: {base_dir.as_posix()}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
'''
    with open(base_dir / "data.yaml", 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"配置文件已创建: {base_dir / 'data.yaml'}")

if __name__ == "__main__":
    convert_to_yolov5()

