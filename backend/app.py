# -*- coding: utf-8 -*-
"""
口罩检测系统 - Flask Web 后端
"""
import io, os, base64, json, sys
from pathlib import Path
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np

YOLOV5_DIR = Path(__file__).resolve().parents[1] / "yolov5"
sys.path.insert(0, str(YOLOV5_DIR))

MODEL_PATH = YOLOV5_DIR / "runs" / "train" / "mask_detection" / "weights" / "best.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = YOLOV5_DIR / "yolov5s.pt"

CLASS_NAMES = ["mask", "nomask"]
COLORS = {"mask": (0, 255, 0), "nomask": (255, 0, 0)}
MAX_IMAGE_SIZE = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

print(f"正在加载模型: {MODEL_PATH}")
try:
    model = torch.hub.load(str(YOLOV5_DIR), "custom", path=str(MODEL_PATH), source="local", force_reload=False)
    model.conf = 0.4
    model.iou = 0.45
    print("模型加载成功!  类别:", model.names)
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型未加载"}), 500
    if "image" not in request.files:
        return jsonify({"error": "请上传图片文件"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "不支持的文件格式"}), 400
    img_bytes = file.read(MAX_IMAGE_SIZE + 1)
    if len(img_bytes) > MAX_IMAGE_SIZE:
        return jsonify({"error": "图片过大，最大 16MB"}), 413
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "图片文件无效"}), 400

    results = model(img, size=640)
    detections = []
    draw = ImageDraw.Draw(img)
    font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 20)
                break
            except: continue
    if font is None:
        font = ImageFont.load_default()

    stats = {"mask": 0, "nomask": 0, "total": 0}
    for det in results.pandas().xyxy[0].itertuples():
        x1, y1, x2, y2 = int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)
        conf = round(det.confidence, 3)
        cls_name = str(det.name).lower()
        display_name = cls_name
        if cls_name in ["0", "with_mask"]:
            display_name = "mask"
        elif cls_name in ["1", "without_mask", "no_mask"]:
            display_name = "nomask"
        if display_name in stats: stats[display_name] += 1
        stats["total"] += 1
        color = COLORS.get(display_name, (0, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{display_name} {conf:.2f}"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle([x1, y1-th-4, x1+tw+6, y1], fill=color)
        draw.text((x1+3, y1-th-2), label_text, fill=(255,255,255), font=font)
        detections.append({"class": display_name, "confidence": conf, "bbox": {"x1":x1,"y1":y1,"x2":x2,"y2":y2}})

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return jsonify({
        "success": True,
        "image": f"data:image/jpeg;base64,{img_base64}",
        "detections": detections,
        "stats": stats,
        "filename": file.filename
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "model_path": str(MODEL_PATH)})

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "文件过大，最大 16MB"}), 413

if __name__ == "__main__":
    print("="*50)
    print("  口罩检测系统已启动")
    print(f"  模型: {MODEL_PATH.name}")
    print("  访问: http://127.0.0.1:5000")
    print("="*50)
    app.run(host="127.0.0.1", port=5000, debug=False)
