# 项目简介

本项目为“教室视频监控智能分析系统”，旨在实现对教室等场所的实时视频监控、运动与危险行为检测、AI识别、告警联动与Web可视化。系统集成了多种视频分析算法与现代Web技术，适用于智慧教室、实验室、会议室等场景的安全管理与行为分析。

## 技术栈与架构

- **后端主语言**：Python 3
- **主要依赖/框架**：
  - OpenCV（视频采集、图像处理、运动分析）
  - Flask（Web服务与API接口）
  - Ultralytics YOLO/torch（AI目标检测）
  - Numpy（数值计算）
  - 多线程与队列（高效解耦采集与处理）
- **系统架构**：
  - **模块化设计**：各功能（采集、运动分析、行为识别、AI检测、告警、Web等）均为独立模块，便于扩展与维护。
  - **多线程解耦**：采集、处理、Web服务等各自独立线程，互不阻塞，保障实时性。
  - **数据流驱动**：帧数据通过队列流转，分析与告警结果通过内存与API同步。
  - **前后端分离**：后端Flask负责API与视频流，前端HTML+JS定时拉取数据、动态展示。
  - **配置驱动**：支持通过命令行和JSON配置灵活调整参数、规则、区域等。

## 主要特性
- 实时视频采集与处理，支持摄像头/视频流/文件
- 运动特征提取（光流、轮廓、运动历史等）
- AI目标检测（支持YOLO等模型）
- 多目标跟踪与轨迹分析
- 行为识别（静止、行走、奔跑、徘徊、打架等）
- 危险行为检测与区域告警（如大范围运动、摔倒、闯入、危险区域停留）
- 告警管理与统计，支持前端处理/标记
- Web可视化界面，展示视频、统计、告警、系统状态
- 配置灵活，易于扩展和二次开发

# 项目模块结构与调用关系总览

## 1. 主入口模块

### src/all_in_one_system.py
- **功能**：系统主入口，整合视频采集、运动分析、危险行为识别、AI识别、告警、Web界面等所有核心功能。
- **主要职责**：
  - 解析命令行参数，初始化各子模块
  - 启动视频采集、特征提取、行为识别、AI推理、告警处理等多线程流程
  - 提供Web服务（Flask），支持前端实时展示与控制
  - 管理系统状态、统计信息、告警历史等

## 2. 运动与行为分析模块

### src/models/motion/motion_manager.py
- **功能**：负责从视频帧中提取运动特征（如光流、运动历史、轮廓等），为后续分析提供基础数据。
- **调用关系**：被主系统和行为识别模块调用。

### src/models/behavior/behavior_recognition.py
- **功能**：基于轨迹和运动特征，识别个体行为（如静止、行走、奔跑、徘徊、打架等）和对象间交互。
- **调用关系**：依赖轨迹管理、运动特征，结果用于告警和前端展示。

### src/models/trajectory/trajectory_manager.py
- **功能**：管理多目标的轨迹，分析对象移动、消失、交互等。
- **调用关系**：为行为识别和告警等模块提供轨迹数据。

## 3. 危险行为与告警模块

### src/danger_recognizer.py
- **功能**：检测视频中的危险行为（如大范围运动、摔倒、闯入、危险区域停留等），并生成告警。
- **调用关系**：主系统定期调用其process_frame方法，结果用于告警统计和前端展示。

### src/models/alert/alert_system.py
- **功能**：统一管理告警规则、事件、通知、处理状态等，支持多种告警类型和处理流程。
- **调用关系**：可被主系统或危险识别模块调用，支持与前端交互。

## 4. 目标检测与跟踪模块

### src/models/object_detection/object_tracker.py
- **功能**：集成YOLO等模型，实现多目标检测与ID跟踪，为行为分析和危险检测提供对象信息。
- **调用关系**：主系统和行为识别模块调用。

## 5. 视频采集与处理模块

### src/models/video/video_capture.py
- **功能**：统一管理摄像头/视频流采集，支持断线重连、测试模式、帧统计等。
- **调用关系**：主系统直接调用，作为数据源。

## 6. 配置与工具模块

### src/config/*.json
- **功能**：系统规则、通知等配置文件。

### src/utils/
- **功能**：帧预处理、运动工具、帧有效性检测等通用工具。

## 7. Web前端与API

### templates/index.html
- **功能**：前端主页面，展示视频流、系统状态、统计、告警等。
- **调用关系**：由Flask后端渲染，JS定时拉取后端API数据。

## 8. 测试与脚本

### test/
- **功能**：包含各子模块和系统级的测试脚本、批处理脚本。

---

# 模块间调用与交互流程

1. **主系统启动**（all_in_one_system.py）
   - 解析命令行参数，初始化视频采集、运动特征、危险识别、AI模型、Web服务等。
2. **视频采集**（video_capture.py）
   - 持续采集摄像头/视频流帧，送入处理队列。
3. **运动特征提取**（motion_manager.py）
   - 对每帧提取光流、轮廓等特征。
4. **目标检测与跟踪**（object_tracker.py）
   - AI模型检测人/物体，分配ID，输出检测结果。
5. **轨迹管理**（trajectory_manager.py）
   - 跟踪每个对象的移动轨迹，分析交互。
6. **行为识别**（behavior_recognition.py）
   - 结合轨迹和运动特征，识别行为类型（如打架、徘徊等）。
7. **危险行为检测**（danger_recognizer.py）
   - 检查大范围运动、摔倒、危险区域等，生成告警。
8. **告警系统**（alert_system.py）
   - 统一管理告警事件、规则、通知、处理状态。
9. **Web前端交互**
   - Flask后端提供API，前端定时拉取统计、告警、状态等数据，实时展示。

---

# 实时视频监测实现流程（命令行启动）

以如下命令为例：

```bash
python src/all_in_one_system.py --source 0 --width 640 --height 480 --process_every 1 --max_fps 30 --enable_ai --vision_model yolov8n --ai_interval 5 --ai_confidence 0.4 --feature_threshold 80 --area_threshold 0.05 --alert_cooldown 10 --min_confidence 0.5 --distance_threshold 50 --dwell_time_threshold 1.0 --alert_region "[(10,10), (260,10), (260,380), (10,380)]" --record --save_alerts --web_interface
```

- **参数说明**：
  - `--source 0`：摄像头编号（或视频文件路径）
  - `--width/--height`：采集分辨率
  - `--process_every`：每隔多少帧处理一次
  - `--max_fps`：最大处理帧率
  - `--enable_ai`：启用AI目标检测
  - `--vision_model yolov8n`：使用YOLOv8n模型
  - `--ai_interval`：AI检测间隔帧数
  - `--ai_confidence`：AI检测置信度阈值
  - `--feature_threshold`、`--area_threshold`：运动特征与区域阈值
  - `--alert_cooldown`：告警冷却时间
  - `--min_confidence`：最小置信度
  - `--distance_threshold`、`--dwell_time_threshold`：危险区域相关参数
  - `--alert_region`：危险区域多边形
  - `--record`：保存处理后视频
  - `--save_alerts`：保存告警
  - `--web_interface`：启用Web前端

- **流程**：
  1. 启动后，系统初始化所有模块。
  2. 持续采集视频帧，依次进行运动特征提取、AI检测、轨迹跟踪、行为识别、危险检测。
  3. 检测到危险行为时，生成告警并通过Web界面实时展示。
  4. 前端页面可查看视频、统计、告警详情，并通过API与后端交互。 

---

# 详细模块结构与调用链说明

## 1. 主入口 AllInOneSystem（src/all_in_one_system.py）

- **主要类**：
  - `AllInOneSystem`：系统核心类，负责参数解析、各子模块初始化、线程管理、Web服务、主流程调度。
- **关键方法/线程**：
  - `start()`：启动系统，分别启动视频采集线程（`capture_thread_func`）和处理线程（`process_thread_func`）。
  - `capture_thread_func()`：持续采集视频帧，放入队列。
  - `process_thread_func()`：从队列取帧，依次进行运动特征提取、AI检测、危险行为识别、告警生成、可视化、统计更新。
  - `init_web_server()`：初始化Flask Web服务，注册API（视频流、统计、告警、控制等）。
  - `generate_frames()`：为Web端持续推送处理后的视频帧。
  - `visualize_frame()`：叠加运动/行为/告警等可视化信息。
- **数据流**：
  1. 视频帧采集 → 运动特征提取 → AI检测 → 轨迹/行为分析 → 危险检测 → 告警生成 → 可视化 → Web/API输出。
  2. 多线程+队列解耦采集与处理，Web端通过API/流式接口获取最新状态。
- **与子模块交互**：
  - 运动特征：`self.motion_manager.extract_features()`
  - 危险识别：`self.danger_recognizer.process_frame()`
  - AI检测：`self.ai_model()`（YOLO等）
  - 告警管理：内部维护all_alerts/recent_alerts，支持前端处理/标记

## 2. 运动特征管理 MotionFeatureManager（src/models/motion/motion_manager.py）

- **主要类**：
  - `MotionFeatureManager`：负责光流、运动历史、轮廓、关键点等多种运动特征的提取与可视化。
- **关键方法**：
  - `extract_features(frame, prev_frame)`：输入当前帧和前一帧，输出特征字典（光流、轮廓、关键点等）。
  - `visualize_features(frame, features)`：在帧上叠加特征可视化信息。
  - `reset()`：重置内部状态。
- **输入输出**：
  - 输入：原始帧、前一帧
  - 输出：特征字典（含光流、轮廓、关键点、统计量等）
- **与其他模块关系**：
  - 被主系统和行为识别模块调用，特征结果用于危险检测、行为分析、可视化。

## 3. 行为识别 BehaviorRecognizer（src/models/behavior/behavior_recognition.py）

- **主要类**：
  - `BehaviorRecognizer`：基于轨迹和运动特征，识别静止、行走、奔跑、徘徊、打架等行为。
- **关键方法**：
  - `analyze(trajectories, motion_features, interaction_detector)`：综合分析轨迹和运动特征，输出行为和交互列表。
  - `analyze_trajectories(trajectories, motion_features)`：单体行为识别。
  - `detect_interactions(trajectories, interaction_detector)`：对象间交互识别。
  - `visualize_behaviors(frame, behaviors, interactions)`：在帧上叠加行为/交互可视化。
- **输入输出**：
  - 输入：轨迹列表、运动特征、交互检测器
  - 输出：行为对象列表、交互对象列表
- **与其他模块关系**：
  - 依赖轨迹管理、运动特征，结果用于危险检测、告警、前端展示。

## 4. 轨迹管理 TrajectoryManager（src/models/trajectory/trajectory_manager.py）

- **主要类**：
  - `TrajectoryManager`：管理多目标轨迹，分析对象移动、消失、交互。
- **关键方法**：
  - `update(tracked_objects)`：输入跟踪对象，更新轨迹、消失计数、交互。
  - `get_active_trajectories()`：获取所有活跃轨迹。
  - `get_interaction_detector()`：获取交互检测器。
  - `reset()`：重置所有轨迹。
- **输入输出**：
  - 输入：跟踪对象列表（含ID、位置、类别等）
  - 输出：轨迹字典、活跃对象ID、交互信息
- **与其他模块关系**：
  - 为行为识别、危险检测等模块提供轨迹数据。

## 5. 目标检测与跟踪 ObjectTracker（src/models/object_detection/object_tracker.py）

- **主要类**：
  - `ObjectTracker`：集成YOLO等模型，实现多目标检测与ID跟踪。
  - `OpenCVDetector`：基于OpenCV DNN的检测器，支持YOLOv4/SSD/FasterRCNN。
  - `CentroidTracker`：基于质心的ID分配与跟踪。
- **关键方法**：
  - `detect(frame)`：输入帧，输出检测对象列表（含类别、置信度、坐标、ID）。
  - `update(detections)`：输入检测结果，输出带ID的跟踪对象。
  - `reset()`：重置跟踪状态。
- **输入输出**：
  - 输入：视频帧
  - 输出：检测对象列表、跟踪对象列表
- **与其他模块关系**：
  - 主系统和行为识别模块调用，结果用于轨迹管理、行为分析、危险检测。

## 6. 危险行为识别 DangerRecognizer（src/danger_recognizer.py）

- **主要类**：
  - `DangerRecognizer`：检测大范围运动、摔倒、闯入、危险区域停留等危险行为。
- **关键方法**：
  - `process_frame(frame, features, object_detections)`：输入帧、特征、检测结果，输出告警列表。
  - `add_alert_region(region, name)`：添加危险区域。
  - `_analyze_danger()`、`_track_danger_zone_dwell()`等：内部分析逻辑。
  - `get_alert_stats()`：获取告警统计。
- **输入输出**：
  - 输入：帧、运动特征、检测对象
  - 输出：告警字典列表（含类型、置信度、帧号、描述等）
- **与其他模块关系**：
  - 主系统定期调用，结果用于告警统计、前端展示。

## 7. 告警系统 AlertSystem（src/models/alert/alert_system.py）

- **主要类**：
  - `AlertSystem`：统一管理告警规则、事件、通知、处理状态。
- **关键方法**：
  - `process_frame(frame_idx, frame, behavior_results, tracks, motion_features, scene_data, process_now)`：处理帧及分析结果，生成告警。
  - `get_recent_alerts(count)`：获取最近告警。
  - `get_alert_stats()`：获取告警统计。
  - `acknowledge_alert(alert_id)`：标记告警为已处理。
  - `set_rule_enabled(rule_id, enabled)`、`update_rule(rule)`：管理告警规则。
- **输入输出**：
  - 输入：帧、行为/轨迹/运动特征等分析结果
  - 输出：告警事件对象、统计、规则
- **与其他模块关系**：
  - 可被主系统或危险识别模块调用，支持与前端交互。

## 8. 视频采集 VideoCaptureManager（src/models/video/video_capture.py）

- **主要类**：
  - `VideoCaptureManager`：统一管理摄像头/视频流采集，支持断线重连、测试模式、帧统计。
- **关键方法**：
  - `open_source(source)`：打开视频源。
  - `read()`：读取单帧。
  - `read_frames(validate)`：帧生成器，支持校验。
  - `get_video_properties()`、`get_processing_stats()`：获取视频属性和统计。
  - `release()`、`close()`：释放资源。
- **输入输出**：
  - 输入：摄像头编号/视频路径
  - 输出：帧、帧号、统计信息
- **与其他模块关系**：
  - 主系统直接调用，作为数据源。

---

# 典型时序与数据流动

1. **系统启动**：命令行参数传递给AllInOneSystem，初始化所有子模块。
2. **采集线程**：持续采集视频帧，放入队列。
3. **处理线程**：
   - 取出帧，调用MotionFeatureManager提取运动特征。
   - 若启用AI，调用YOLO等模型检测目标，ObjectTracker分配ID。
   - TrajectoryManager更新轨迹，BehaviorRecognizer识别行为/交互。
   - DangerRecognizer分析危险行为，生成告警。
   - 告警信息存入all_alerts/recent_alerts，统计同步更新。
   - 可视化结果叠加运动/行为/告警等信息，推送Web端。
4. **Web前端**：Flask API持续提供视频流、统计、告警、控制等接口，前端定时拉取并展示。
5. **告警处理**：前端可通过API标记告警为已处理，后端同步状态。

---

# 典型调用链举例

- `capture_thread_func` → `process_thread_func` → `motion_manager.extract_features` → `ai_model()` → `object_tracker.update` → `trajectory_manager.update` → `behavior_recognizer.analyze` → `danger_recognizer.process_frame` → `all_alerts`/`recent_alerts` → `visualize_frame` → `generate_frames`/Web API 