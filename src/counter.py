# -*- coding: utf-8 -*-
"""跨线计数与每分钟人流量统计。

在画面中设置一条竖直计数线，当某条轨迹的质心从线的一侧移动到另一侧时，
记一次"跨线事件"，并按视频时间归入对应分钟，得到每分钟人流量。
"""
import csv
from collections import defaultdict
from pathlib import Path


class FlowCounter:
    def __init__(self, line_x, fps):
        self.line_x = line_x      # 计数线的x坐标
        self.fps = fps            # 视频帧率
        self.sides = {}           # track_id -> 上一帧所在侧(-1左 / +1右)
        self.events = []          # 跨线事件列表
        self.per_minute = defaultdict(int)   # 第N分钟 -> 跨线人次

    @property
    def total(self):
        return len(self.events)

    def update(self, tracks, frame_idx):
        """用当前帧轨迹更新计数状态。"""
        time_s = frame_idx / self.fps
        minute = int(time_s // 60)
        for tr in tracks:
            if tr.prev_centroid is None:
                # 新出现的轨迹只记录所在侧，不计数
                self.sides[tr.id] = -1 if tr.centroid[0] < self.line_x else 1
                continue
            side = -1 if tr.centroid[0] < self.line_x else 1
            prev_side = self.sides.get(tr.id)
            if prev_side is not None and side != prev_side:
                direction = "left_to_right" if side > 0 else "right_to_left"
                self.events.append({
                    "track_id": tr.id,
                    "frame": int(frame_idx),
                    "time_s": round(time_s, 2),
                    "direction": direction,
                })
                self.per_minute[minute] += 1
            self.sides[tr.id] = side

    def flow_rows(self):
        """生成CSV行：分钟、该分钟人流量、累计人数。"""
        rows, cumulative = [], 0
        max_minute = max(self.per_minute.keys(), default=-1)
        for minute in range(max_minute + 1):
            count = self.per_minute.get(minute, 0)
            cumulative += count
            rows.append({"minute": minute + 1,       # 展示为第1分钟、第2分钟…
                         "people_per_minute": count,
                         "cumulative_total": cumulative})
        return rows

    def save_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=["minute", "people_per_minute",
                               "cumulative_total"])
            writer.writeheader()
            writer.writerows(self.flow_rows())
        return path

    def summary(self):
        rows = self.flow_rows()
        flows = [r["people_per_minute"] for r in rows]
        avg = round(sum(flows) / len(flows), 1) if flows else 0.0
        return {
            "total_crossings": self.total,
            "per_minute": {str(r["minute"]): r["people_per_minute"]
                           for r in rows},
            "average_people_per_minute": avg,
        }
