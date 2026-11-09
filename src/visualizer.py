# -*- coding: utf-8 -*-
"""可视化模块：绘制检测框、轨迹ID、计数线、统计信息，并输出标注视频。"""
from pathlib import Path

import cv2

import config

COLOR_OK = (80, 200, 80)        # 高置信度框（BGR绿色）
COLOR_LOW = (0, 165, 255)       # 低置信度框（BGR橙色），提示人工复核
COLOR_LINE = (255, 80, 80)      # 计数线（BGR红色）
COLOR_TEXT = (40, 40, 40)       # HUD文字


def open_video_writer(out_dir, stem, fps, size):
    """创建视频写入器，编码失败时自动降级（mp4v -> XVID -> MJPG）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for codec, ext in (("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")):
        path = out_dir / f"{stem}{ext}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec),
                                 fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError("无法创建视频写入器（编码器不可用）")


class Renderer:
    """把检测/跟踪/计数结果绘制到视频帧上。"""

    def __init__(self, line_x, detector_name, fps, width):
        self.line_x = line_x
        self.detector_name = detector_name
        self.fps = fps
        self.width = width

    def draw(self, frame, tracks, counter, frame_idx):
        h, w = frame.shape[:2]
        t_s = frame_idx / self.fps
        minute = int(t_s // 60) + 1

        # 计数线（虚线效果：分段画）
        for y in range(0, h, 16):
            cv2.line(frame, (self.line_x, y),
                     (self.line_x, min(y + 8, h)), COLOR_LINE, 2)
        cv2.putText(frame, "Counting Line", (self.line_x - 60, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LINE, 1,
                    cv2.LINE_AA)

        # 检测框 + 轨迹ID + 置信度
        low_conf = 0
        for tr in tracks:
            x, y, bw, bh = [int(v) for v in tr.box]
            color = COLOR_OK if tr.score >= config.LOW_CONF_THRESHOLD \
                else COLOR_LOW
            if tr.score < config.LOW_CONF_THRESHOLD:
                low_conf += 1
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            label = f"ID{tr.id} {tr.score:.2f}"
            cv2.putText(frame, label, (x, max(y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)

        # HUD统计信息
        lines = [
            f"Detector: {self.detector_name}",
            f"Time: {t_s:6.1f}s   Minute: {minute}",
            f"Total crossings: {counter.total}",
            f"Flow(min {minute}): "
            f"{counter.per_minute.get(minute - 1, 0)} people/min",
        ]
        if low_conf:
            lines.append(f"Low-confidence to review: {low_conf}")
        for i, text in enumerate(lines):
            y = 22 + i * 20
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, COLOR_TEXT, 1, cv2.LINE_AA)
        return frame
