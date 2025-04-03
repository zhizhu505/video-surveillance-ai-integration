# 视频监控危险行为检测系统文档

欢迎使用视频监控危险行为检测系统文档！本文档提供了系统的详细信息，包括安装说明、使用指南、API参考和开发者文档。

## 文档索引

- [快速入门指南](docs/QUICK_START.md)
- [API参考文档](docs/API_REFERENCE.md)
- [开发者指南](docs/DEVELOPER_GUIDE.md)
- [大模型集成指南](docs/AI_INTEGRATION_GUIDE.md)

## 系统架构

### 基础架构

![系统架构图](docs/images/architecture.png)

### AI增强架构

我们的系统集成了先进的AI大模型能力，以提高危险行为识别的准确性和智能性：

```mermaid
flowchart TD
    %% 核心组件
    VideoInput[视频输入模块] --> FeatureExtraction[特征提取模块]
    FeatureExtraction --> BehaviorAnalysis[行为分析模块]
    BehaviorAnalysis --> AlertSystem[告警系统]
    
    %% AI模块
    AILayer[大模型集成层]
    
    %% 连接关系
    VideoInput --> AILayer
    FeatureExtraction --> AILayer
    AILayer --> BehaviorAnalysis
    AILayer --> AlertSystem
    
    %% AI子组件
    subgraph AIModels[大模型模块]
        VisionModels[视觉模型\nYOLO/SAM/CLIP]
        MultimodalModels[多模态模型\nGPT-4V/Claude 3]
        LLMs[大语言模型\nGPT-4/Llama 3]
    end
    
    AILayer --- AIModels
    
    %% 输出接口
    AlertSystem --> UI[实时监控界面]
    AlertSystem --> Notification[通知系统]
    AILayer --> Reports[分析报告]
    
    %% 样式定义
    classDef core fill:#d5e8d4,stroke:#82b366,stroke-width:2px
    classDef ai fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px
    classDef output fill:#ffe6cc,stroke:#d79b00,stroke-width:2px
    
    class VideoInput,FeatureExtraction,BehaviorAnalysis,AlertSystem core
    class AILayer,AIModels,VisionModels,MultimodalModels,LLMs ai
    class UI,Notification,Reports output
```

AI增强架构将视觉大模型、多模态大模型和大语言模型无缝集成到现有的视频监控系统中，提供以下增强功能：

- 精确物体识别与跟踪
- 深度场景理解与上下文分析
- 智能告警分析与误报过滤
- 自然语言告警解释与建议

如需了解更多有关AI集成的详细信息，请参阅[大模型集成指南](docs/AI_INTEGRATION_GUIDE.md)。

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 联系方式

如有任何问题或建议，请联系开发团队或提交issue。 