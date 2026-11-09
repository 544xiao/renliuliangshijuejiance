# 校园人流量视觉检测（renliuliangshijuejiance）

> 基于计算机视觉的校园走廊人流量计量系统：从监控视频中检测行人、跨线计数，并按时间统计**每分钟人流量**。

## 1. 需求与目标

**目标用户**：校园后勤/保卫部门的管理人员。

**真实场景**：教学楼走廊、食堂门口、图书馆入口的高峰人流量，目前依靠人工现场计数或事后回看监控，耗时且容易漏记。

**现有痛点**：
- 人工计数无法覆盖长时间监控，成本高、不可复核；
- 简单的"运动人数统计"无法区分方向、无法给出按分钟的时间序列；
- 现有商用方案闭源、难以按校园场景调整。

**本项目目标**：给定一段监控视频（或摄像头），自动输出
1. 跨越计数线的总人次（区分方向）；
2. **每分钟人流量**（CSV + 曲线图）；
3. 带检测框/轨迹ID的标注视频（可人工复核）。

**成功标准**：一条命令完成端到端处理并产出上述文件；通过切换检测器（MOG2/HOG/YOLOv8）与人工计数对照，量化各方法的计数差异与误差。

## 2. 系统设计与工作流

```
任务入口(main.py)
   │  解析参数：视频源 / 检测器 / 输出目录
   ▼
数据准备 ── 用户视频文件 或 摄像头
   ▼
行人检测(ML组件) ── 三种可互换检测器：
   │   · HOG+SVM      传统机器学习（OpenCV 内置预训练）
   │   · MOG2         运动检测基线（背景减除）
   │   · YOLOv8       CNN 深度学习（可选，需 ultralytics）
   ▼
多目标跟踪 ── IoU 贪心匹配，为行人分配稳定轨迹ID
   ▼
跨线计数 ── 质心跨越竖直计数线 → 记一次事件(方向/时刻)
   ▼
时间聚合 ── 按视频时间归入分钟桶 → 每分钟人流量
   ▼
结果输出 ── results/<视频名>/ 下的 CSV / 曲线 / 标注视频 / 摘要
```

**失败与安全处理**：
- 视频文件不存在 / 无法打开 → 给出明确错误与修复建议后退出；
- `auto` 模式下 YOLO 权重或依赖不可用 → 自动降级 HOG 并打印原因；
- 检测器连续 60 帧零检出 → 自动切换 MOG2 基线并提示；
- 置信度低于 0.5 的框以橙色标注，HUD 显示"待人工复核"数量，不做无依据的自动决策。

## 3. 快速开始

环境：Python 3.10（Windows / Linux / macOS）

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 处理自己的视频（把视频放到 data/ 或任意路径）
python main.py --video data/test1.mp4

# 可指定检测器（默认 auto：优先YOLO，不可用则HOG）
python main.py --video data/test1.mp4 --detector hog
python main.py --video data/test1.mp4 --detector mog2

# 使用摄像头
python main.py --video 0 --detector mog2
```

启用 YOLOv8 深度学习检测器（可选，需联网下载约2GB依赖与权重）：

```bash
pip install ultralytics
python main.py --video data/test1.mp4 --detector yolo
```

### 输出文件

**每个"视频+检测器"组合的结果独立存放在 `results/<视频文件名>_<检测器>/` 子文件夹中**，
文件夹名带检测器后缀，方便区分不同测试方式的结果：

```
results/
├── test1_hog/                    ← HOG+SVM 检测器的结果
│   ├── annotated.mp4             # 标注视频：检测框、轨迹ID、计数线、实时统计HUD
│   ├── flow_log.csv              # 每分钟人流量、累计人数
│   ├── flow_per_minute.png       # 每分钟人流量柱状+趋势图
│   ├── summary.json              # 运行摘要：检测器、帧数、总人次、平均人流量
│   └── predictions.json          # 逐帧预测框与跨线事件
├── test1_mog2/                   ← MOG2 检测器的结果（运行后生成）
├── test1_yolo/                   ← YOLOv8 检测器的结果（运行后生成）
└── 其他视频_检测器/               ← 同一规则
```

### 对照实验（基线比较）

对同一段视频切换检测器分别运行，结果分别存入 `results/test1_hog/`、`results/test1_mog2/`、`results/test1_yolo/`，可逐文件夹对比 `summary.json` 中的总人次与每分钟流量：

```bash
python main.py --video data/test1.mp4 --detector mog2   # 运动基线
python main.py --video data/test1.mp4 --detector hog    # 传统ML
python main.py --video data/test1.mp4 --detector yolo   # 深度学习（可选）
```

## 4. 目录结构

```
renliuliangshijuejiance/
├── README.md                  # 本说明
├── requirements.txt           # 依赖清单
├── LICENSE                    # MIT
├── config.py                  # 路径与算法参数（集中配置，便于复现）
├── main.py                    # 端到端入口（任务流程编排）
├── src/
│   ├── detectors.py           # 检测器：MOG2 / HOG+SVM / YOLOv8
│   ├── tracker.py             # IoU贪心多目标跟踪
│   ├── counter.py             # 跨线计数 + 每分钟流量统计
│   └── visualizer.py          # 标注绘制与视频输出
├── data/                      # 输入视频（如 test1.mp4，不入库）
└── results/                   # 运行输出，按视频名分文件夹（不入库）
```

## 5. 复用组件清单（来源 / 许可证 / 用途）

| 组件 | 版本 | 来源 | 许可证 | 在本项目中的用途 | 本项目改动 |
|------|------|------|--------|------------------|------------|
| OpenCV (opencv-python) | ≥4.8 | https://github.com/opencv/opencv | Apache-2.0 | 视频读写、HOG+SVM行人检测、MOG2背景减除、绘图 | 仅调用，未修改 |
| NumPy | ≥1.24 | https://github.com/numpy/numpy | BSD-3-Clause | 数组运算、图像处理 | 仅调用 |
| Matplotlib | ≥3.7 | https://github.com/matplotlib/matplotlib | PSF-based | 人流量曲线 | 仅调用 |
| ultralytics (YOLOv8) | 可选 ≥8.0 | https://github.com/ultralytics/ultralytics | AGPL-3.0 | 可选的CNN行人检测（`--detector yolo`） | 仅调用预训练yolov8n；**若启用须遵守AGPL-3.0** |

> 说明：本仓库核心代码（检测流程、跟踪、计数、可视化）均为本项目编写；第三方库仅按文档调用，未修改其源码。输入视频由使用者自行提供并自行确保授权，系统不做人脸识别与身份关联。

## 6. 已知局限与改进方向

- 跟踪采用IoU贪心匹配，密集遮挡下可能交换ID，可升级为 DeepSORT/ByteTrack；
- 每分钟流量按视频时间戳统计，未做时钟对齐与多摄像头汇聚；
- 计数线为单线，可扩展为多区域/多入口的区域计数；
- MOG2对静止行人会漏检，HOG对非真实行人泛化弱，真实场景建议 `--detector yolo`。

## 7. 运行环境与复现

- Python 3.10；依赖见 `requirements.txt`
- 复现两步：`pip install -r requirements.txt` → `python main.py --video 你的视频.mp4`
- 所有输出集中于 `results/<视频名>/`，可直接对照复核

## 8. 许可证

本项目代码采用 [MIT License](LICENSE)。第三方组件遵循其各自许可证（见第5节）。


