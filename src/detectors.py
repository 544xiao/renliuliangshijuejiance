# -*- coding: utf-8 -*-
"""行人检测器模块。

提供三个可互换的检测器，构成实验对照：
1. Mog2Detector  —— MOG2背景减除（运动检测基线，零依赖，速度快）
2. HogDetector   —— HOG+SVM行人检测（传统机器学习方法，OpenCV内置预训练）
3. YoloDetector  —— YOLOv8卷积神经网络（深度学习方法，需额外安装ultralytics）

统一接口：detect(frame) -> [{"box": [x, y, w, h], "score": float}, ...]
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import config


class DetectorUnavailable(RuntimeError):
    """检测器不可用（依赖缺失或权重下载失败）。"""


# ------------------------------------------------------------------ 工具
def box_iou(a, b):
    """计算两个 [x, y, w, h] 框的交并比。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(dets, iou_thr=0.35):
    """简单非极大抑制：按分数降序保留互不重叠的框。"""
    kept = []
    for d in sorted(dets, key=lambda x: -x["score"]):
        if all(box_iou(d["box"], k["box"]) < iou_thr for k in kept):
            kept.append(d)
    return kept


def _download(urls, dest: Path, min_size=1000, timeout=60):
    """多地址依次尝试下载（直连失败时走镜像）。"""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size >= min_size:
        return True
    for url in urls:
        try:
            print(f"  正在下载 {dest.name} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) >= min_size:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                print(f"  下载成功: {dest.name} ({len(data) // 1024} KB)")
                return True
        except Exception as exc:  # noqa: BLE001 逐个镜像容错
            print(f"  下载失败({type(exc).__name__})，尝试下一个地址")
    return False


# ------------------------------------------------------------------ 检测器
class BaseDetector:
    name = "base"

    def detect(self, frame):
        raise NotImplementedError


class Mog2Detector(BaseDetector):
    """基线：MOG2高斯混合背景减除，检测画面中的运动区域。"""

    name = "mog2"

    def __init__(self, min_area=config.MIN_BOX_AREA,
                 max_area_ratio=config.MAX_BOX_AREA_RATIO):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=48, detectShadows=True)
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def detect(self, frame):
        fg = self.subtractor.apply(frame)
        fg[fg == 127] = 0                       # 去掉阴影（127）
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        frame_area = frame.shape[0] * frame.shape[1]
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect = h / max(w, 1)
            if area < self.min_area or area > frame_area * self.max_area_ratio:
                continue
            if not (1.1 <= aspect <= 6.0):      # 行人长宽比过滤
                continue
            # MOG2没有真实置信度，用面积占比给出0.5~0.9的近似分数
            score = round(min(0.9, 0.5 + area / frame_area * 4), 2)
            boxes.append({"box": [int(x), int(y), int(w), int(h)],
                          "score": score})
        return nms(boxes)


class HogDetector(BaseDetector):
    """传统机器学习：HOG方向梯度直方图 + 线性SVM行人分类器（OpenCV内置预训练）。"""

    name = "hog"

    def __init__(self, hit_threshold=0.0, scale=1.05, win_stride=(8, 8)):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hit_threshold = hit_threshold
        self.scale = scale
        self.win_stride = win_stride

    def detect(self, frame):
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            scale=self.scale,
            hitThreshold=self.hit_threshold)
        dets = []
        for (x, y, w, h), weight in zip(rects, weights):
            # 兼容不同OpenCV版本：weights可能是(N,1)数组或标量
            score = float(np.ravel(weight)[0])
            dets.append({"box": [int(x), int(y), int(w), int(h)],
                         "score": round(score, 3)})
        return nms(dets)


class YoloDetector(BaseDetector):
    """深度学习：YOLOv8目标检测，取person类别（coco类别0）。"""

    name = "yolo"

    def __init__(self, weights_path=None, urls=None):
        try:
            from ultralytics import YOLO  # 可选依赖
        except ImportError as exc:
            raise DetectorUnavailable(
                "未安装 ultralytics，可执行: pip install ultralytics") from exc

        weights = Path(weights_path or config.YOLO_WEIGHTS)
        if not weights.exists():
            print("YOLO权重不存在，尝试下载 yolov8n.pt ...")
            urls = urls or config.YOLO_WEIGHT_URLS
            if not _download(urls, weights, min_size=5_000_000):
                raise DetectorUnavailable(
                    "yolov8n.pt 下载失败（网络原因）。"
                    "可手动下载后放到 weights/ 目录，或改用 --detector hog")
        self.model = YOLO(str(weights))

    def detect(self, frame):
        result = self.model(frame, verbose=False)[0]
        dets = []
        for b in result.boxes:
            if int(b.cls[0]) != 0:               # 只保留 person 类别
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dets.append({"box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                         "score": round(float(b.conf[0]), 3)})
        return dets


def create_detector(name: str):
    """工厂函数：按名称创建检测器；auto 优先YOLO，不可用则降级HOG。"""
    name = (name or "auto").lower()
    if name == "mog2":
        return Mog2Detector()
    if name == "hog":
        return HogDetector()
    if name == "yolo":
        return YoloDetector()
    if name == "auto":
        try:
            return YoloDetector()
        except DetectorUnavailable as exc:
            print(f"[auto] YOLO不可用({exc})，降级为 HOG+SVM 检测器")
            return HogDetector()
    raise ValueError(f"未知检测器: {name}（可选 auto/hog/mog2/yolo）")


