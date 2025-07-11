# 快速开始指南 - Video Surveillance AI Integration

本指南将帮助您快速启动 video-surveillance-ai-integration 项目，包括环境准备、依赖安装、项目启动及常见问题处理。

## 一、前期准备

### 1. 获取项目代码

克隆仓库到本地：
```bash
git clone https://github.com/zhizhu505/video-surveillance-ai-integration.git
cd video-surveillance-ai-integration
```

切换到目标分支（如 zsq 分支）：
```bash
git checkout zsq
```

解决分支冲突（若有）：
```bash
git pull origin zsq  # 拉取远程更新
# 手动合并冲突文件（如 all_in_one_system.log）
git add .
git commit -m "Resolve merge conflicts"
git push
```

### 2. 创建并激活虚拟环境

新建虚拟环境：
```bash
python -m venv venv  # 创建名为 venv 的虚拟环境
```

激活环境（PowerShell）：
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  # 临时允许脚本执行
.\venv\Scripts\Activate.ps1  # 激活后命令行前缀显示 (venv)
```

激活环境（Command Prompt）：
```bash
venv\Scripts\activate.bat
```

激活环境（Linux/Mac）：
```bash
source venv/bin/activate
```

## 二、安装依赖

### 1. 安装基础依赖

项目依赖记录在 `requirements.txt` 中，执行：
```bash
pip install -r requirements.txt
```

### 2. 补充缺失依赖

#### 解决 cv2 导入错误：安装 OpenCV
```bash
pip install opencv-python opencv-contrib-python
```

#### 解决 AI 功能禁用：安装 YOLOv8 依赖
```bash
pip install ultralytics  # YOLOv8 官方库
```

#### 解决声学检测缺失：安装音频监控模块
```bash
pip install sounddevice tensorflow librosa
```

### 3. 验证依赖安装

检查关键依赖是否正确安装：
```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import ultralytics; print('Ultralytics installed successfully')"
python -c "import sounddevice; print('Sounddevice installed successfully')"
```

## 三、启动项目

### 1. 运行启动脚本

通过项目提供的批处理文件启动系统：
```bash
.\run_all_in_one.bat
```

或使用危险等级测试脚本：
```bash
.\run_danger_level_test.bat
```

或直接执行主程序：
```bash
python src/all_in_one_system.py --source 0 --width 640 --height 480 --process_every 1 --max_fps 30 --enable_ai --vision_model yolov8n --ai_interval 5 --ai_confidence 0.4 --feature_threshold 80 --area_threshold 0.05 --alert_cooldown 10 --min_confidence 0.5 --distance_threshold 50 --dwell_time_threshold 1.0 --alert_region "[(10,10), (260,10), (260,380), (10,380)]" --record --save_alerts --web_interface
```

### 2. 验证启动状态

成功启动后，您应该看到以下日志信息：
```
2025-07-11 12:06:49,025 - AllInOneSystem - INFO - 成功导入MotionFeatureManager
2025-07-11 12:06:49,025 - AllInOneSystem - INFO - 成功导入DangerRecognizer
2025-07-11 12:06:49,245 - AllInOneSystem - INFO - 成功导入Flask Web模块
2025-07-11 12:06:49,255 - DangerRecognizer - INFO - 危险行为识别器已初始化
2025-07-11 12:06:49,258 - AllInOneSystem - INFO - Web服务器已启动，访问 http://localhost:5000/
2025-07-11 12:06:49,264 - AllInOneSystem - INFO - 教室视频监控系统初始化完成
2025-07-11 12:06:49,266 - AllInOneSystem - INFO - 教室视频监控系统已启动
```

### 3. 访问 Web 界面

Web 服务器启动后，可通过以下地址访问监控界面：

- **本地访问**：http://127.0.0.1:5000
- **局域网访问**：http://10.61.115.116:5000（根据日志中实际 IP 调整）

## 四、系统功能验证

### 1. 核心功能检查

启动后系统应具备以下功能：
- ✅ 视频捕获和处理
- ✅ 运动检测和分析
- ✅ 危险行为识别
- ✅ 告警生成和保存
- ✅ Web 界面显示
- ✅ 危险等级分类（低/中/高）

### 2. 功能模块状态

检查日志中的模块状态：
- **MotionFeatureManager**：运动特征管理器
- **DangerRecognizer**：危险行为识别器
- **Flask Web模块**：Web 界面服务
- **AI功能**：YOLOv8 对象检测
- **音频监控**：声学事件检测

## 五、常见问题处理

### 1. 虚拟环境激活失败

若 `Activate.ps1` 未找到，删除并重建虚拟环境：
```bash
Remove-Item -Recurse -Force venv  # 删除损坏环境
python -m venv venv  # 重新创建
```

### 2. Web 界面访问错误

- 检查 URL 拼写（确保端口为 5000）
- 确认系统启动日志中 werkzeug 已显示服务器运行（Running on ...）
- 若提示"网页解析失败"，检查 `templates` 目录下是否有 HTML 文件

### 3. 功能模块禁用

日志中若提示以下警告，重新安装对应依赖后重启系统：
```
WARNING - 未找到必要的AI依赖，AI功能将被禁用
WARNING - 未找到audio_monitor，声学检测功能将被禁用
```

### 4. cv2 导入错误（IDE 显示）

这是 IDE 的 Python 语言服务器问题，不影响实际运行：
- 确保 IDE 使用正确的 Python 解释器
- 在 VS Code 中：`Ctrl+Shift+P` → "Python: Select Interpreter" → 选择虚拟环境中的 Python
- 重启 IDE 或重新加载窗口

### 5. 权限问题

在 Windows 上可能遇到执行策略限制：
```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 六、系统配置

### 1. 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 视频源（0=摄像头，文件路径=视频文件） | 0 |
| `--width/--height` | 视频分辨率 | 640x480 |
| `--enable_ai` | 启用 AI 检测 | False |
| `--vision_model` | AI 模型类型 | yolov8n |
| `--feature_threshold` | 运动特征阈值 | 80 |
| `--area_threshold` | 运动面积阈值 | 0.05 |
| `--web_interface` | 启用 Web 界面 | False |

### 2. 告警区域配置

在 `src/config/rules.json` 中配置告警规则：
```json
{
  "large_area_motion": {
    "enabled": true,
    "area_threshold": 0.05,
    "danger_level": "low"
  },
  "fall_detection": {
    "enabled": true,
    "confidence_threshold": 0.7,
    "danger_level": "high"
  }
}
```

## 七、测试和验证

### 1. 运行测试脚本

```bash
# 测试危险等级系统
python test_danger_levels.py

# 测试摔倒检测
python test_fall_detection.py

# 测试运动检测
python test_motion_detection.py
```

### 2. 验证告警功能

- 在摄像头前进行大幅度运动触发"大面积运动"告警
- 模拟摔倒动作触发"摔倒检测"告警
- 检查 `alerts/` 目录中是否生成告警图片
- 在 Web 界面查看告警列表和危险等级颜色

## 八、总结

核心流程：
1. **克隆代码** → 获取项目源码
2. **配置虚拟环境** → 创建隔离的 Python 环境
3. **安装依赖** → 安装所有必要的 Python 包
4. **启动系统** → 运行主程序或批处理脚本
5. **访问 Web 界面** → 通过浏览器查看监控结果

通过逐步排查依赖和环境问题，可确保系统正常运行，启用视频监控、运动检测、危险行为识别等核心功能。

## 九、下一步

- 查看 [API 参考文档](API_REFERENCE.md) 了解详细接口
- 阅读 [开发者指南](DEVELOPER_GUIDE.md) 进行二次开发
- 参考 [模块概览](MODULE_OVERVIEW.md) 了解系统架构
- 查看 [危险等级指南](DANGER_LEVEL_GUIDE.md) 了解告警系统

---

**注意**：首次启动可能需要较长时间下载 AI 模型文件，请耐心等待。 