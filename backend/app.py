# -*- coding: utf-8 -*-
"""
口罩检测系统 - Flask Web 后端
支持: 图片检测 / 视频检测 / 摄像头检测
"""
import io, os, base64, json, sys, tempfile, time, uuid, subprocess
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
MAX_VIDEO_SIZE = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "wmv", "flv"}

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_SIZE  # 200MB 统一上限

# 启动时清理过期的检测视频（保留最近 1 小时）
STATIC_VIDEOS_DIR = Path(__file__).parent / "static" / "videos"
STATIC_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
try:
    now = time.time()
    for f in STATIC_VIDEOS_DIR.iterdir():
        if f.is_file() and f.suffix == ".mp4" and (now - f.stat().st_mtime) > 3600:
            f.unlink(missing_ok=True)
except Exception:
    pass

# ────────────── 字体工具 ──────────────
_FONT_CACHE = None

def _get_font(size=20):
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                _FONT_CACHE = ImageFont.truetype(fp, size)
                return _FONT_CACHE
            except Exception:
                continue
    _FONT_CACHE = ImageFont.load_default()
    return _FONT_CACHE


# ────────────── 模型加载 ──────────────
print(f"正在加载模型: {MODEL_PATH}")
try:
    model = torch.hub.load(str(YOLOV5_DIR), "custom", path=str(MODEL_PATH), source="local", force_reload=False)
    model.conf = 0.4
    model.iou = 0.45
    print("模型加载成功!  类别:", model.names)
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None


# ────────────── 工具函数 ──────────────

def allowed_file(filename, extensions=ALLOWED_EXTENSIONS):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def annotate_image(img, results):
    """
    在 PIL Image 上绘制检测框，返回 (annotated_img, detections_list, stats_dict)
    """
    detections = []
    draw = ImageDraw.Draw(img)
    font = _get_font()

    stats = {"mask": 0, "nomask": 0, "total": 0}
    for det in results.pandas().xyxy[0].itertuples():
        x1, y1, x2, y2 = int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)
        conf = round(det.confidence, 3)
        cls_name = str(det.name).lower()

        # 规范化类别名
        display_name = cls_name
        if cls_name in ["0", "with_mask"]:
            display_name = "mask"
        elif cls_name in ["1", "without_mask", "no_mask"]:
            display_name = "nomask"

        if display_name in stats:
            stats[display_name] += 1
        stats["total"] += 1

        color = COLORS.get(display_name, (0, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label_text = f"{display_name} {conf:.2f}"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th - 2), label_text, fill=(255, 255, 255), font=font)

        detections.append({
            "class": display_name,
            "confidence": conf,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    return img, detections, stats


def image_to_base64(img, fmt="JPEG", quality=95):
    """PIL Image → data URI base64 字符串"""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    buf.seek(0)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


# ────────────── 路由 ──────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """图片口罩检测"""
    if model is None:
        return jsonify({"error": "模型未加载"}), 500

    if "image" not in request.files:
        return jsonify({"error": "请上传图片文件"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "不支持的文件格式，支持: " + ", ".join(ALLOWED_EXTENSIONS)}), 400

    img_bytes = file.read(MAX_IMAGE_SIZE + 1)
    if len(img_bytes) > MAX_IMAGE_SIZE:
        return jsonify({"error": "图片过大，最大 16MB"}), 413
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "图片文件无效"}), 400

    results = model(img, size=640)
    img, detections, stats = annotate_image(img, results)
    img_base64 = image_to_base64(img)

    return jsonify({
        "success": True,
        "image": img_base64,
        "detections": detections,
        "stats": stats,
        "filename": file.filename
    })


@app.route("/predict_video", methods=["POST"])
def predict_video():
    """视频口罩检测 — 逐帧处理，返回标注视频 + 统计数据"""
    if model is None:
        return jsonify({"error": "模型未加载"}), 500

    # ── 校验输入 ──
    if "video" not in request.files:
        return jsonify({"error": "请上传视频文件"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename, VIDEO_EXTENSIONS):
        return jsonify({"error": "不支持的视频格式，支持: " + ", ".join(VIDEO_EXTENSIONS)}), 400

    video_bytes = file.read(MAX_VIDEO_SIZE + 1)
    if len(video_bytes) > MAX_VIDEO_SIZE:
        return jsonify({"error": "视频过大，最大 200MB"}), 413

    # ── 写入临时文件 ──
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    try:
        import cv2
    except ImportError:
        os.unlink(tmp_in_path)
        return jsonify({"error": "服务器缺少 OpenCV (cv2)，无法处理视频"}), 500

    frames_dir = None
    try:
        cap = cv2.VideoCapture(tmp_in_path)
        if not cap.isOpened():
            os.unlink(tmp_in_path)
            return jsonify({"error": "无法打开视频文件，可能已损坏"}), 400

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / orig_fps if orig_fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 每间隔 orig_fps 帧采样一次（即 1fps）
        sample_interval = max(1, int(round(orig_fps)))

        frame_idx = 0
        processed_count = 0
        frame_summary = []
        anno_pil_cache = None  # 上次标注的 PIL Image（用于填充间隔帧）

        # 累计统计
        agg_stats = {"mask": 0, "nomask": 0, "total_faces": 0, "frame_count": 0}

        # 临时目录存放 JPEG 帧（用于 FFMPEG 编码）
        frames_dir = tempfile.mkdtemp()

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            should_sample = (frame_idx % sample_interval == 0)

            if should_sample:
                # BGR → RGB → PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # 推理 + 标注
                results = model(pil_img, size=640)
                pil_img, detections, frame_stats = annotate_image(pil_img, results)
                anno_pil_cache = pil_img

                # 累计
                agg_stats["mask"] += frame_stats.get("mask", 0)
                agg_stats["nomask"] += frame_stats.get("nomask", 0)
                agg_stats["total_faces"] += frame_stats.get("total", 0)
                agg_stats["frame_count"] += 1
                processed_count += 1

                # 记录该帧摘要
                frame_summary.append({
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / orig_fps, 1),
                    "mask": frame_stats.get("mask", 0),
                    "nomask": frame_stats.get("nomask", 0),
                    "total": frame_stats.get("total", 0),
                })

                # 标注后的 PIL → 保存为 JPEG
                anno_rgb = np.array(pil_img)
                anno_bgr = cv2.cvtColor(anno_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:08d}.jpg"), anno_bgr)

            elif anno_pil_cache is not None:
                # 非采样帧：重复最近的标注帧
                anno_rgb = np.array(anno_pil_cache)
                anno_bgr = cv2.cvtColor(anno_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:08d}.jpg"), anno_bgr)
            else:
                # 刚开始还未有标注帧，写入原始帧
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:08d}.jpg"), frame_bgr)

            frame_idx += 1

        cap.release()

        # ── 使用 FFMPEG 编码 H.264 MP4 ──
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg_exe = get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = "ffmpeg"  # 回退，假设在 PATH 中

        video_filename = f"{uuid.uuid4().hex}.mp4"
        static_videos_dir = Path(__file__).parent / "static" / "videos"
        static_videos_dir.mkdir(parents=True, exist_ok=True)
        h264_out = str(static_videos_dir / video_filename)

        ffmpeg_cmd = [
            ffmpeg_exe, "-y",
            "-framerate", str(orig_fps),
            "-i", os.path.join(frames_dir, "frame_%08d.jpg"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-crf", "23",
            h264_out
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, timeout=300)

        # ── 计算平均置信度（仅最后一个 sampled frame 的） ──
        avg_conf = 0
        if agg_stats["total_faces"] > 0:
            avg_conf = round(
                sum(d.get("confidence", 0) for d in detections) / max(len(detections), 1),
                3
            )

        return jsonify({
            "success": True,
            "video_url": f"/static/videos/{video_filename}",
            "duration_sec": round(duration_sec, 1),
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "fps": round(orig_fps, 1),
            "aggregate_stats": {
                "mask": agg_stats["mask"],
                "nomask": agg_stats["nomask"],
                "total_faces": agg_stats["total_faces"],
                "frame_count": agg_stats["frame_count"],
            },
            "frame_summary": frame_summary,
            "filename": file.filename,
        })

    except Exception as e:
        return jsonify({"error": f"视频处理失败: {str(e)}"}), 500
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_in_path)
        except Exception:
            pass
        # 清理帧目录
        try:
            import shutil
            shutil.rmtree(frames_dir)
        except Exception:
            pass


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "文件过大，图片最大 16MB，视频最大 200MB"}), 413


if __name__ == "__main__":
    print("=" * 50)
    print("  口罩检测系统已启动")
    print(f"  模型: {MODEL_PATH.name}")
    print("  图片检测: POST /predict")
    print("  视频检测: POST /predict_video")
    print("  访问: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=False)
