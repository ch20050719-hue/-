
@echo off
echo 开始训练口罩检测模型...
echo.
python train.py --data data/mask.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 100 --device 0 --project runs/train --name mask_detection
echo.
echo 训练完成！
pause

