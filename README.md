#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全功能视频监控系统 - 核心代码包
"""

# 视频监控危险行为检测系统

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](LICENSE)

基于计算机视觉的实时视频监控系统，能够自动检测和报警危险行为。该系统利用光流和运动特征提取技术，在不需要复杂深度学习模型的情况下实现高效、低延迟的危险行为检测。

![系统预览](docs/system_preview.png)

## 主要特性

- **多类型危险行为检测**：
  - 突然剧烈运动
  - 大范围异常运动
  - 摔倒检测
  - 异常移动模式（如徘徊）
  - 警戒区域入侵检测

- **高性能设计**：
  - 双线程架构（视频捕获与处理分离）
  - 帧缓冲管理和选择性处理
  - 支持GPU加速（OpenCV CUDA后端）

- **实用功能**：
  - 自定义警戒区域
  - 可视化告警信息
  - 告警帧自动保存
  - 运行报告生成
  - 视频录制

## 安装要求

### 依赖包

```bash
pip install -r requirements.txt
```

主要依赖：
- OpenCV >= 4.0.0
- NumPy >= 1.19.0
- Python >= 3.7

## 快速开始

使用预设配置启动系统：

```bash
# Windows
run_all_in_one.bat

# Linux/Mac
python src/all_in_one_system.py --source 0 --width 640 --height 480 --save_alerts
```

## 系统架构

系统由以下主要组件构成：

1. **运动特征提取模块**：
   - 光流提取器（OpticalFlowExtractor）
   - 运动历史提取器（MotionHistoryExtractor）
   - 特征管理器（MotionFeatureManager）

2. **危险行为识别模块**：
   - 行为分析器（DangerRecognizer）
   - 多种行为检测算法

3. **视频处理核心**：
   - 视频捕获线程
   - 处理分析线程
   - 显示和输出管理

## 使用指南

### 命令行参数

```
python src/all_in_one_system.py [参数]
```

#### 主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 视频源（0表示摄像头，也可以是视频文件路径） | 0 |
| `--width` | 视频宽度 | 640 |
| `--height` | 视频高度 | 480 |
| `--process_every` | 每N帧处理一次（提高性能） | 3 |
| `--max_fps` | 最大帧率 | 30 |
| `--feature_threshold` | 特征点数量阈值（影响检测灵敏度） | 100 |
| `--area_threshold` | 运动区域阈值（占比） | 0.05 |
| `--alert_cooldown` | 告警冷却帧数 | 10 |
| `--min_confidence` | 最小置信度 | 0.5 |
| `--alert_region` | 警戒区域，例如：`"[(100,100), (300,100), (300,300), (100,300)]"` | 无 |
| `--output` | 输出目录 | danger_system_output |
| `--record` | 记录视频 | False |
| `--save_alerts` | 保存告警帧 | False |
| `--use_gpu` | 使用GPU加速 | False |
| `--use_motion_history` | 使用运动历史 | False |

### 交互快捷键

在程序运行时，可使用以下快捷键：
- `ESC`：退出程序
- `s`：保存当前帧
- `r`：重置统计信息

## 自定义警戒区域

可以通过`--alert_region`参数定义多边形警戒区域，当检测到物体或运动在该区域内时会触发报警。示例：

```bash
python src/all_in_one_system.py --alert_region "[(100,100), (300,100), (300,300), (100,300)]"
```

## 调整检测灵敏度

可以通过调整以下参数来改变系统的灵敏度：

1. **降低检测阈值**，使系统更敏感：
   ```bash
   python src/all_in_one_system.py --feature_threshold 50 --area_threshold 0.03
   ```

2. **提高检测阈值**，减少误报：
   ```bash
   python src/all_in_one_system.py --feature_threshold 150 --area_threshold 0.1
   ```

## 优化性能

针对不同硬件环境，可以通过以下方式优化系统性能：

1. **提高处理间隔**，减少CPU负载：
   ```bash
   python src/all_in_one_system.py --process_every 5
   ```

2. **降低分辨率**，提高处理速度：
   ```bash
   python src/all_in_one_system.py --width 320 --height 240
   ```

3. **启用GPU加速**（如果支持）：
   ```bash
   python src/all_in_one_system.py --use_gpu
   ```

## 常见问题

1. **系统运行缓慢**
   - 降低分辨率或增加处理间隔
   - 减少输出（不录制视频或不保存告警帧）
   - 使用更强大的硬件

2. **误报过多**
   - 增加特征点数量阈值
   - 增加告警冷却时间
   - 调整区域检测阈值

3. **摄像头无法启动**
   - 检查摄像头设备ID是否正确
   - 确保其他程序未占用摄像头
   - 尝试重新插拔USB摄像头

## 扩展功能

### 添加新的危险行为检测

可以在`src/danger_recognizer.py`中的`_analyze_danger`方法中添加新的检测逻辑。例如：

```python
# 7. 新的行为检测
if [检测条件]:
    alerts.append({
        'type': self.DANGER_TYPES['new_behavior_type'],
        'confidence': 0.8,
        'frame': self.current_frame
    })
```

### 集成对象检测

系统支持集成第三方对象检测结果，可通过`process_frame`方法的`object_detections`参数传入。对象检测结果应当包含以下格式：

```python
[
    {
        'class': '人',
        'confidence': 0.95,
        'bbox': [x1, y1, x2, y2]
    },
    # 更多对象...
]
```

## 协议

[MIT License](LICENSE)

## 贡献

欢迎提交问题报告、功能请求和贡献代码。

---

## 系统演示

![告警示例](docs/alert_example.png)

运行报告示例：
```
==== 系统报告 ====
运行时间: 120.45 秒
总帧数: 3589
处理帧数: 1196 (33.3%)
平均帧率: 29.8 FPS
告警总数: 12

告警分类统计:
  - 突然剧烈运动: 7
  - 大范围异常运动: 3
  - 可能摔倒: 1
  - 入侵警告区域: 1
```

---

# 项目结构

## 目录结构

```
.
├── run_all_in_one.bat                # 启动系统的批处理文件
├── requirements.txt                  # 项目依赖库列表
├── README.md                         # 项目说明文档
├── alerts/                           # 告警保存目录
├── docs/                             # 文档目录
├── output/                           # 输出文件和结果目录
├── templates/                        # Web界面模板
│   └── index.html                    # 主页模板
├── test/                             # 测试文件目录
└── src/                              # 源代码目录
    ├── __init__.py                   # 包初始化文件
    ├── all_in_one_system.py          # 全功能系统主程序
    ├── danger_recognizer.py          # 危险行为识别模块
    ├── config/                       # 配置文件目录
    │   └── __init__.py               # 配置管理模块
    ├── models/                       # 模型目录
    │   ├── __init__.py               # 模型包初始化文件
    │   ├── alert/                    # 告警系统模块
    │   │   ├── __init__.py           # 告警模块初始化文件
    │   │   ├── alert_event.py        # 告警事件定义
    │   │   ├── alert_rule.py         # 告警规则定义
    │   │   ├── alert_system.py       # 告警系统核心
    │   │   ├── alert_processor.py    # 告警处理器
    │   │   ├── alert_plugins.py      # 告警插件
    │   │   ├── notification_manager.py # 通知管理器
    │   │   └── rule_analyzer.py      # 规则分析器
    │   ├── behavior/                 # 行为分析模块
    │   │   ├── __init__.py           # 行为模块初始化文件
    │   │   ├── behavior_analysis.py  # 行为分析器
    │   │   ├── behavior_recognition.py # 行为识别器
    │   │   └── behavior_types.py     # 行为类型定义
    │   ├── motion/                   # 运动特征相关模块
    │   │   ├── __init__.py           # 运动模块初始化文件
    │   │   ├── motion_manager.py     # 运动特征管理器
    │   │   ├── optical_flow.py       # 光流提取器
    │   │   ├── motion_history.py     # 运动历史提取器
    │   │   └── motion_feature_base.py # 运动特征基类
    │   ├── object_detection/         # 物体检测模块
    │   │   ├── __init__.py           # 物体检测模块初始化文件
    │   │   └── object_tracker.py     # 物体跟踪器
    │   ├── trajectory/               # 轨迹分析模块
    │   │   ├── __init__.py           # 轨迹模块初始化文件
    │   │   ├── trajectory.py         # 轨迹类
    │   │   ├── trajectory_manager.py # 轨迹管理器
    │   │   ├── trajectory_analysis.py # 轨迹分析器
    │   │   └── interaction_detector.py # 交互检测器
    │   ├── video/                    # 视频处理模块
    │   │   ├── __init__.py           # 视频模块初始化文件
    │   │   ├── video_capture.py      # 视频捕获类
    │   │   └── frame_processor.py    # 帧处理器
    │   └── visual_language/          # 视觉语言模型模块
    │       ├── __init__.py           # 视觉语言模块初始化文件
    │       ├── qwen_vl.py            # 通义千问视觉语言模型
    │       └── rga.py                # RGA模型
    ├── utils/                        # 工具函数目录
    │   ├── __init__.py               # 工具包初始化文件
    │   ├── frame_validation.py       # 帧验证工具
    │   ├── motion_utils.py           # 运动分析工具
    │   └── preprocessing.py          # 图像预处理工具
    ├── preprocessing_samples/        # 预处理样本目录
    └── video_samples/                # 视频样本目录
```

## 核心文件说明

### 主要程序文件

- **src/all_in_one_system.py**: 系统的主程序，整合了运动特征提取、危险行为识别和Web界面等功能。包含双线程架构（视频捕获与处理分离）、告警处理和可视化功能。

- **src/danger_recognizer.py**: 危险行为识别模块，用于检测视频中的异常行为。包含多种行为检测逻辑，如突然剧烈运动、大范围异常运动、摔倒检测等。

### 运动特征提取模块

- **src/models/motion/motion_manager.py**: 运动特征管理器，负责协调和管理各种运动特征提取器。支持光流、运动历史、背景减除、轮廓检测等多种特征提取方法。

- **src/models/motion/optical_flow.py**: 光流提取器，用于计算连续帧之间的像素运动。支持Farneback和Lucas-Kanade算法，可选GPU加速。

- **src/models/motion/motion_history.py**: 运动历史提取器，记录和分析视频中的运动历史。用于检测长时间的运动模式和行为。

- **src/models/motion/motion_feature_base.py**: 运动特征基类，定义了所有运动特征提取器的共同接口和基础功能。

### 其他模块文件

- **src/models/alert/**: 告警系统模块，包含告警管理、规则分析和通知功能。
  
- **src/models/behavior/**: 行为分析模块，用于识别和分类各种行为模式。

- **src/models/object_detection/**: 物体检测模块，集成了目标跟踪功能。

- **src/models/trajectory/**: 轨迹分析模块，用于分析和预测运动轨迹。

- **src/models/video/**: 视频处理模块，包含帧处理和视频捕获功能。

- **src/models/visual_language/**: 视觉语言模型集成，用于高级场景理解。

### Web界面

- **templates/index.html**: Web界面的主页模板，提供视频流显示、系统状态监控和控制功能。

## 启动脚本

- **run_all_in_one.bat**: Windows启动脚本，配置了推荐的系统参数，包括视频源、分辨率、处理参数和告警设置。

## 使用建议

1. 首次运行系统建议使用`run_all_in_one.bat`脚本，它已配置好推荐参数。

2. 如需自定义系统配置，可直接编辑脚本或使用命令行参数启动：
   ```bash
   python src/all_in_one_system.py --width 640 --height 480 --feature_threshold 60 --enable_ai
   ```

3. 对于开发者，可以通过扩展`src/danger_recognizer.py`中的检测逻辑来添加新的危险行为检测类型。

---

# 大模型集成指南

本系统支持与AI大模型集成，以增强危险行为识别能力、提供更深层次的场景理解，并实现更智能的告警分析。

## 大模型集成架构

![大模型集成架构](docs/ai_integration.png)

集成大模型后的系统架构如下：

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  视频输入模块  │───▶│  特征提取模块  │───▶│  行为分析模块  │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        │            ┌───────▼───────┐    ┌───────▼───────┐
        └──────────▶│   大模型模块   │◀───┤   告警系统    │
                    └───────┬───────┘    └───────────────┘
                            │
                    ┌───────▼───────┐
                    │  结果解释模块  │
                    └───────────────┘
```

## 支持的大模型类型

系统支持以下类型的AI大模型集成：

1. **视觉大模型**：用于高级场景理解和物体识别
   - YOLO系列 (YOLOv8, YOLOv9)
   - SAM (Segment Anything Model)
   - CLIP (Contrastive Language-Image Pre-training)

2. **多模态大模型**：用于视觉-语言理解
   - GPT-4V
   - Gemini
   - Qwen-VL
   - LLaVA
   - Claude 3 Vision

3. **大语言模型**：用于告警分析和决策支持
   - GPT-4/3.5
   - Claude
   - Qwen/Qwen2
   - Llama 2/3
   - Mistral

## 安装大模型依赖

### 添加依赖

编辑requirements.txt文件，添加以下依赖：

```
# 大模型依赖
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
openai>=1.0.0
anthropic>=0.7.0
llama-cpp-python>=0.2.0
ultralytics>=8.0.0  # YOLOv8支持
huggingface_hub>=0.16.0
sentence-transformers>=2.2.0
```

安装依赖：
```bash
pip install -r requirements.txt
```

### 模型下载

对于本地运行的模型，您需要下载相应的权重文件：

```bash
# 下载YOLO模型
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 下载CLIP模型
python -c "from transformers import CLIPProcessor, CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"

# 下载轻量级LLM模型(可选)
python download_models.py --model tiny-llama
```

## 配置大模型

在`config/models.yaml`中配置您的模型选项：

```yaml
vision_models:
  default: "yolov8n"
  options:
    - name: "yolov8n"
      path: "models/yolov8n.pt"
      confidence: 0.5
    - name: "clip"
      path: "openai/clip-vit-base-patch32"
      device: "cuda"

multimodal_models:
  default: "openai"
  options:
    - name: "openai"
      type: "api"
      model: "gpt-4-vision-preview"
      api_key_env: "OPENAI_API_KEY"
    - name: "qwen-vl"
      type: "local"
      path: "Qwen/Qwen-VL-Chat"
      device: "cuda"

llm_models:
  default: "openai"
  options:
    - name: "openai"
      type: "api"
      model: "gpt-4-turbo"
      api_key_env: "OPENAI_API_KEY"
    - name: "llama"
      type: "local"
      path: "meta-llama/Llama-3-8B-Instruct"
      device: "cuda"
```

## 命令行参数

新增的命令行参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_ai` | 启用AI大模型支持 | False |
| `--vision_model` | 使用的视觉模型名称 | "yolov8n" |
| `--multimodal_model` | 使用的多模态模型名称 | "openai" |
| `--llm_model` | 使用的LLM模型名称 | "openai" |
| `--ai_interval` | AI处理间隔帧数 | 30 |
| `--ai_confidence` | AI检测最小置信度 | 0.5 |
| `--scene_analysis` | 启用场景分析 | False |
| `--behavior_analysis` | 启用行为理解分析 | False |
| `--local_models_only` | 仅使用本地模型 | False |

## 使用示例

### 基础集成

```bash
python integrate_danger_detection.py --source 0 --enable_ai --vision_model yolov8n
```

### 完整AI集成

```bash
python integrate_danger_detection.py --source 0 --enable_ai --vision_model yolov8n --multimodal_model openai --llm_model openai --scene_analysis --behavior_analysis --ai_interval 20
```

### 仅使用本地模型

```bash
python integrate_danger_detection.py --source 0 --enable_ai --local_models_only --vision_model yolov8n
```

## 批处理快速启动

Windows:
```bash
run_ai_monitor.bat
```

Linux/Mac:
```bash
./run_ai_monitor.sh
```

## 实现细节

### 1. 视觉模型集成

将视觉大模型集成到危险行为检测系统需要修改`integrate_danger_detection.py`文件，并添加新的模块：

```python
# models/ai/vision_model.py

class VisionModelManager:
    def __init__(self, config):
        self.model_name = config.get('model_name', 'yolov8n')
        self.confidence = config.get('confidence', 0.5)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
    def _load_model(self):
        if self.model_name.startswith('yolo'):
            from ultralytics import YOLO
            return YOLO(f"{self.model_name}.pt")
        elif self.model_name == 'clip':
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            model.to(self.device)
            return model
        # 添加其他模型支持...
            
    def process_frame(self, frame):
        # 处理帧并返回检测结果
        if self.model_name.startswith('yolo'):
            results = self.model(frame)
            return self._parse_yolo_results(results)
        # 其他模型处理...
    
    def _parse_yolo_results(self, results):
        # 将YOLO结果转换为标准格式
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf.item() > self.confidence:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': r.names[cls],
                        'confidence': conf
                    })
        return detections
```

### 2. 多模态和LLM集成

创建处理模块以整合多模态模型和LLM：

```python
# models/ai/multimodal_manager.py

class MultimodalManager:
    def __init__(self, config):
        self.model_type = config.get('model_type', 'api')
        self.model_name = config.get('model_name', 'openai')
        self.client = self._setup_client()
        
    def _setup_client(self):
        if self.model_type == 'api' and self.model_name == 'openai':
            import openai
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            return openai.Client()
        elif self.model_type == 'local':
            # 加载本地多模态模型
            # ...
            
    def analyze_scene(self, frame, detections=None):
        # 使用多模态模型分析场景
        if self.model_type == 'api' and self.model_name == 'openai':
            # 将帧转换为base64
            import base64
            from io import BytesIO
            from PIL import Image
            buffered = BytesIO()
            Image.fromarray(frame).save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 调用API
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": "分析视频画面中的潜在危险行为和异常情况。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "分析这个视频帧中是否存在危险行为或异常情况:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        # 其他模型处理...
```

### 3. 告警分析与强化

创建LLM分析器来增强告警：

```python
# models/ai/llm_analyzer.py

class LLMAnalyzer:
    def __init__(self, config):
        self.model_type = config.get('model_type', 'api')
        self.model_name = config.get('model_name', 'openai')
        self.client = self._setup_client()
        
    def _setup_client(self):
        if self.model_type == 'api' and self.model_name == 'openai':
            import openai
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            return openai.Client()
        # 其他模型设置...
            
    def analyze_alerts(self, alerts, context=None):
        """分析告警并提供深度解释"""
        if not alerts:
            return None
            
        alert_descriptions = "\n".join([f"- {a['type']} (置信度: {a['confidence']:.2f})" for a in alerts])
        prompt = f"""
        分析以下视频监控告警:
        {alert_descriptions}
        
        提供:
        1. 告警严重程度评估
        2. 可能的原因分析
        3. 建议采取的行动
        4. 是否可能存在误报
        """
        
        if self.model_type == 'api' and self.model_name == 'openai':
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "你是视频监控系统的AI分析师，专注于解释和分析视频监控告警。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        # 其他模型处理...
```

## 性能优化建议

集成大模型会增加系统资源占用，以下是优化建议：

1. **降低AI处理频率**：使用`--ai_interval`调整大模型处理频率，避免每帧都使用AI
2. **使用轻量级模型**：针对边缘设备，考虑使用YOLOv8n等轻量级模型
3. **混合精度推理**：启用FP16/INT8推理以加速大模型
4. **模型量化**：对模型进行量化以减少内存占用
5. **模型裁剪**：针对特定场景裁剪模型
6. **使用边缘计算设备**：考虑Intel NCS2、Google Coral等边缘AI加速器

## 硬件要求

使用大模型的推荐硬件配置：

- **CPU**: 6核心以上，推荐Intel i7/i9或AMD Ryzen 7/9
- **RAM**: 最低16GB，推荐32GB以上
- **GPU**: 
  - 本地大模型: NVIDIA RTX 3060(8GB+)以上
  - 基础视觉模型: NVIDIA GTX 1650(4GB+)以上
- **存储**: SSD 256GB以上
- **网络**: 使用云端API时，稳定的网络连接

## 注意事项

1. **API密钥管理**：
   - 不要在代码中硬编码API密钥
   - 使用环境变量或密钥管理系统
   - 为API设置使用限制

2. **隐私考虑**：
   - 使用云API时，视频数据会传输到第三方服务器
   - 考虑敏感场景使用本地模型
   - 遵守相关法规和隐私政策

3. **版权与许可**：
   - 使用开源模型需遵循其许可协议
   - 商业使用需确认模型许可是否允许

4. **错误处理**：
   - 系统应能在AI模块失败的情况下继续运行基本功能
   - 实现超时机制防止AI处理阻塞主系统

## 大模型集成开发路线图

1. **基础集成阶段**：
   - 集成YOLO用于物体检测
   - 添加基础的API调用功能
   - 实现结果显示和保存

2. **功能增强阶段**：
   - 添加多种模型支持
   - 实现本地模型推理
   - 增加场景理解功能

3. **高级分析阶段**：
   - 行为序列分析
   - 多摄像头联动分析
   - 实现异常预测功能

4. **性能优化阶段**：
   - 模型量化和优化
   - 硬件加速支持
   - 分布式处理 