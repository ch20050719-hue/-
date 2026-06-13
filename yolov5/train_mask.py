
import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from train import parse_opt, main

if __name__ == "__main__":
    opt = parse_opt()
    # 设置默认参数
    opt.data = ROOT / "data/mask.yaml"
    opt.cfg = ROOT / "models/yolov5s.yaml"
    opt.weights = ROOT / "yolov5s.pt"
    opt.img = 640
    opt.batch = 16
    opt.epochs = 100
    opt.device = "0"  # GPU
    opt.project = ROOT / "runs/train"
    opt.name = "mask_detection"
    main(opt)

