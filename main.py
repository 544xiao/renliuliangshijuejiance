# -*- coding: utf-8 -*-
"""校园人流量视觉检测 —— 端到端入口（任务入口 / 流程编排 / 结果输出）。
任务定义：
    输入：一段监控视频（文件或摄像头）
    输出：跨线人流量统计、每分钟人流量CSV与曲线、标注视频、预测结果JSON
    成功判定：完成视频处理并产出上述文件，计数结果可人工复核

运行示例：
    python main.py --video data/test1.mp4          # 处理自己的视频
    python main.py --video data/test1.mp4 --detector hog
    python main.py --video 0 --detector mog2       # 摄像头
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

import config
from src.counter import FlowCounter
from src.detectors import DetectorUnavailable, Mog2Detector, create_detector
from src.tracker import Tracker
from src.visualizer import Renderer, open_video_writer


def setup_matplotlib():
    """无图形界面环境（如CI）使用Agg后端，只保存图片不弹窗。"""
    import matplotlib
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")


def step(no, total, text):
    print(f"\n[{no}/{total}] {text}")


def parse_video_source(arg):
    """'0'等纯数字 -> 摄像头；否则按文件路径打开。"""
    if arg.isdigit():
        cap = cv2.VideoCapture(int(arg))
        desc = f"摄像头 {arg}"
    else:
        path = Path(arg)
        if not path.exists():
            sys.exit(f"错误：视频文件不存在 -> {path}\n"
                     f"请检查路径是否正确，例如: "
                     f"python main.py --video data/test1.mp4")
        cap = cv2.VideoCapture(str(path))
        desc = str(path)
    if not cap.isOpened():
        sys.exit(f"错误：无法打开视频 -> {desc}\n"
                 "请检查文件格式（推荐mp4/avi）或摄像头是否被占用。")
    return cap, desc


def main():
    parser = argparse.ArgumentParser(
        description="校园人流量视觉检测（图像识别 + 跨线计数 + 每分钟流量）")
    parser.add_argument("--video", default=None,
                        help="视频路径或摄像头编号（如0）")
    parser.add_argument("--detector", default="auto",
                        choices=["auto", "hog", "mog2", "yolo"],
                        help="检测器：auto=优先YOLO不可用则HOG；"
                             "hog=HOG+SVM；mog2=运动基线；yolo=YOLOv8")

    parser.add_argument("--output", default=str(config.RESULTS_DIR),
                        help="结果输出目录（默认 results/）")
    parser.add_argument("--no-video", action="store_true",
                        help="不保存标注视频（加快处理）")
    args = parser.parse_args()

    base_out = Path(args.output)          # results 根目录
    base_out.mkdir(parents=True, exist_ok=True)
    total_steps = 6

    # ---- [1/6] 任务解析 ----
    step(1, total_steps, "解析任务参数")
    if args.video is None:
        parser.print_help()
        sys.exit("\n请指定 --video <视频文件或摄像头号>，"
                 "例如: python main.py --video data/test1.mp4")
    print(f"  检测器: {args.detector}")

    # ---- [2/6] 准备数据 ----
    step(2, total_steps, "准备数据")
    video_source = args.video
    print(f"  视频源: {video_source}")

    # ---- [3/6] 打开视频 ----
    step(3, total_steps, "打开视频")
    cap, desc = parse_video_source(str(video_source))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  已打开: {desc}")
    print(f"  分辨率 {width}x{height}，帧率 {fps:.1f} fps，"
          f"总帧数 {total_frames if total_frames > 0 else '未知'}")

    # ---- [4/6] 加载检测器 ----
    step(4, total_steps, "加载检测器")
    try:
        detector = create_detector(args.detector)
    except DetectorUnavailable as exc:
        print(f"  {exc}")
        print("  自动降级为 mog2 运动检测基线")
        detector = Mog2Detector()
    print(f"  当前检测器: {detector.name}")

    tracker = Tracker()
    line_x = int(width * config.COUNT_LINE_RATIO)
    counter = FlowCounter(line_x=line_x, fps=fps)
    renderer = Renderer(line_x, detector.name, fps, width)

    # 每个视频+检测器组合的结果放进 results/<视频文件名>_<检测器>/ 子文件夹
    # 例如 results/test1_hog、results/test1_mog2，三种检测器结果互不覆盖
    stem = Path(desc).stem if not desc.startswith("摄像头") else "camera"
    out_dir = base_out / f"{stem}_{detector.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  结果目录: {out_dir}")

    writer = None
    if not args.no_video:
        writer, video_out = open_video_writer(out_dir, "annotated", fps,
                                              (width, height))
        print(f"  标注视频将保存到: {video_out}")

    # ---- [5/6] 逐帧：检测 -> 跟踪 -> 计数 ----
    step(5, total_steps, "逐帧检测、跟踪与计数")
    frames_pred = {}
    zero_streak, switched = 0, False
    frame_idx, start = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)

        # 失败降级：连续60帧零检出且当前为hog/yolo -> 切换mog2并提示
        if detections:
            zero_streak = 0
        else:
            zero_streak += 1
        if (not switched and zero_streak >= 60
                and detector.name in ("hog", "yolo")):
            print(f"  [降级] {detector.name} 连续60帧未检出目标，"
                  "自动切换为 mog2 运动检测（建议改用真实行人视频）")
            detector = Mog2Detector()
            renderer.detector_name = detector.name
            tracker = Tracker()
            zero_streak, switched = 0, True

        tracks = tracker.update(detections)
        counter.update(tracks, frame_idx)
        frames_pred[str(frame_idx)] = [d["box"] for d in detections]

        if writer is not None:
            vis = renderer.draw(frame.copy(), tracks, counter, frame_idx)
            writer.write(vis)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  已处理 {frame_idx} 帧，累计跨线 {counter.total} 人次"
                  f"（{time.time() - start:.0f}s）")

    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.time() - start

    if frame_idx == 0:
        sys.exit("错误：视频中没有读到任何帧。")

    # ---- [6/6] 输出结果 ----
    step(6, total_steps, "输出结果")
    flow_csv = counter.save_csv(out_dir / "flow_log.csv")

    # 每分钟人流量曲线
    setup_matplotlib()
    import matplotlib.pyplot as plt
    rows = counter.flow_rows()
    if rows:
        minutes = [r["minute"] for r in rows]
        flows = [r["people_per_minute"] for r in rows]
        plt.figure(figsize=(7, 4))
        plt.bar(minutes, flows, color="#4C78A8", label="people / minute")
        plt.plot(minutes, flows, "o-", color="#E45756", label="trend")
        plt.xlabel("Minute")
        plt.ylabel("People per minute")
        plt.title("Campus People Flow per Minute")
        plt.xticks(minutes)
        plt.legend()
        plt.tight_layout()
        chart = out_dir / "flow_per_minute.png"
        plt.savefig(chart, dpi=150)
        plt.close()
        print(f"  每分钟流量曲线: {chart}")

    summary = counter.summary()
    summary.update({
        "detector": detector.name,
        "video": desc,
        "fps": round(fps, 2),
        "frames": frame_idx,
        "elapsed_s": round(elapsed, 1),
        "output_csv": str(flow_csv),
    })
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pred_path = out_dir / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump({"detector": detector.name, "fps": fps,
                   "total": counter.total,
                   "per_minute": {str(k + 1): v for k, v in
                                  sorted(counter.per_minute.items())},
                   "events": counter.events,
                   "frames": frames_pred},
                  f, ensure_ascii=False)

    print("\n" + "=" * 52)
    print("校园人流量检测 —— 运行摘要")
    print("=" * 52)
    print(f"检测器        : {detector.name}")
    print(f"视频          : {desc}")
    print(f"处理帧数/耗时 : {frame_idx} 帧 / {elapsed:.1f} 秒")
    print(f"跨线总人次    : {summary['total_crossings']}")
    print(f"每分钟人流量  : {summary['per_minute']} 人/分钟")
    print(f"平均人流量    : {summary['average_people_per_minute']} 人/分钟")
    print(f"结果目录      : {out_dir.resolve()}")
    print("=" * 52)


if __name__ == "__main__":
    main()





