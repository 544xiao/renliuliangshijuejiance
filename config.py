# -*- coding: utf-8 -*-
"""全局配置：路径与算法参数集中管理，便于复现实验。"""
from pathlib import Path

# ---- 路径 ----
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
YOLO_WEIGHTS = ROOT / "weights" / "yolov8n.pt"

# ---- 视频与计数线 ----
VIDEO_FPS = 25                  # 视频帧率读取失败时的兜底值
COUNT_LINE_RATIO = 0.5          # 计数线位于画面宽度的比例（0~1）

# ---- 跟踪器 ----
TRACK_MAX_MISSED = 15           # 连续多少帧未检出则删除轨迹
TRACK_MIN_IOU = 0.15            # 新检出与已有轨迹匹配的最小IoU

# ---- 检测结果过滤 ----
MIN_BOX_AREA = 400              # 最小候选框面积（像素）
MAX_BOX_AREA_RATIO = 0.35       # 最大候选框占整帧面积比例
LOW_CONF_THRESHOLD = 0.5        # 低于此置信度标黄，提示人工复核

# ---- YOLO 权重下载镜像（github直连失败时使用） ----
YOLO_WEIGHT_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
]

