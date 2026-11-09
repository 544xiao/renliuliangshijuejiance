# -*- coding: utf-8 -*-
"""多目标跟踪：基于IoU贪心匹配，为每个行人分配稳定的轨迹ID。"""
import config


def box_iou(a, b):
    """计算两个 [x, y, w, h] 框的交并比。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    """单个行人的轨迹状态。"""

    def __init__(self, track_id, box, score, centroid):
        self.id = track_id
        self.box = box              # 当前帧边界框 [x, y, w, h]
        self.score = score          # 当前帧检测置信度
        self.centroid = centroid    # 当前帧质心 (x, y)
        self.prev_centroid = None   # 上一帧质心（新轨迹为None）
        self.missed = 0             # 连续未检出帧数
        self.age = 0                # 存活帧数


class Tracker:
    """IoU贪心匹配跟踪器。

    工作方式：
    1. 计算所有(已有轨迹, 新检出)的IoU，按IoU降序贪心配对；
    2. 配对成功的轨迹更新位置并清零missed；
    3. 未配对的检出创建新轨迹；
    4. 未配对的轨迹missed加1，超过阈值则删除。
    """

    def __init__(self, max_missed=config.TRACK_MAX_MISSED,
                 min_iou=config.TRACK_MIN_IOU):
        self.tracks = []
        self.max_missed = max_missed
        self.min_iou = min_iou
        self._next_id = 1

    @staticmethod
    def _centroid(box):
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    def update(self, detections):
        """输入当前帧检测结果，返回更新后的活动轨迹列表。"""
        # 1) 生成所有 (轨迹下标, 检出下标, IoU) 候选并降序排列
        candidates = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                iou = box_iou(track.box, det["box"])
                if iou >= self.min_iou:
                    candidates.append((iou, ti, di))
        candidates.sort(reverse=True, key=lambda x: x[0])

        # 2) 贪心配对（每个轨迹/检出只用一次）
        used_tracks, used_dets = set(), set()
        for iou, ti, di in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            track = self.tracks[ti]
            det = detections[di]
            track.prev_centroid = track.centroid
            track.box = det["box"]
            track.score = det["score"]
            track.centroid = self._centroid(det["box"])
            track.missed = 0
            track.age += 1

        # 3) 未匹配的检出 -> 新轨迹
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks.append(Track(tid, det["box"], det["score"],
                                     self._centroid(det["box"])))
            used_tracks.add(len(self.tracks) - 1)

        # 4) 未匹配的轨迹 -> missed+1，超限删除
        kept = []
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.missed += 1
                track.age += 1
            if track.missed <= self.max_missed:
                kept.append(track)
        self.tracks = kept
        return self.tracks
