# AI监控日报生成指南

本文档详细介绍如何使用CoT思维链 + Prompt Engineering实现AI自动生成监控日报功能。

## 功能概述

AI监控日报生成系统基于以下技术栈实现：

- **CoT思维链（Chain-of-Thought）**：分步骤推理，确保分析逻辑清晰
- **Prompt Engineering**：精心设计的提示词模板，提升生成质量
- **多模态集成**：结合视觉-语言模型，提供更丰富的场景描述
- **RESTful API**：提供完整的API接口，便于集成

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   数据收集模块   │───▶│   CoT分析模块   │───▶│   日报生成模块   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   告警数据API   │    │   统计分析      │    │   Markdown输出  │
│   统计数据API   │    │   事件分析      │    │   HTML输出      │
│   来源类型API   │    │   风险分析      │    │   API接口       │
└─────────────────┘    │   建议生成      │    └─────────────────┘
                       └─────────────────┘
```

## CoT思维链实现

### 1. 统计分析步骤
```python
def _analyze_statistics(self, data):
    """第一步：统计分析"""
    prompt = f"""
    作为安防AI分析师，请分析以下监控统计数据：
    
    数据概览：
    - 总告警数：{data['total_alerts']}
    - 已处理：{data['handled_count']}
    - 未处理：{data['unhandled_count']}
    - 处理率：{data['handled_count']/data['total_alerts']*100:.1f}%
    
    请从以下角度分析：
    1. 告警数量是否正常？
    2. 处理效率如何？
    3. 主要告警类型是什么？
    4. 有什么异常趋势？
    """
    return self._call_llm(prompt, "statistics_analysis")
```

### 2. 事件分析步骤
```python
def _analyze_events(self, data, multimodal_content):
    """第二步：事件分析（依赖第一步结果）"""
    prompt = f"""
    作为安防AI分析师，请分析以下监控事件：
    
    事件列表：{events_summary}
    场景描述：{scene_descriptions}
    
    请从以下角度分析：
    1. 最严重的事件是什么？
    2. 事件的时间分布有什么特点？
    3. 哪些事件需要重点关注？
    4. 有什么异常模式？
    """
    return self._call_llm(prompt, "events_analysis")
```

### 3. 风险分析步骤
```python
def _analyze_risks(self, data, events_analysis):
    """第三步：风险分析（依赖第二步结果）"""
    prompt = f"""
    作为安防AI分析师，请评估以下风险指标：
    
    风险指标：
    - 高风险告警：{high_risk_count}个
    - 未处理高风险：{unhandled_high_risk}个
    - 总告警数：{total_alerts}个
    - 处理率：{handling_rate:.1f}%
    
    事件分析结果：{events_analysis}
    
    请评估：
    1. 当前风险等级（低/中/高）
    2. 主要风险点是什么？
    3. 需要立即关注的问题？
    4. 风险趋势如何？
    """
    return self._call_llm(prompt, "risk_analysis")
```

### 4. 建议生成步骤
```python
def _generate_recommendations(self, data, risk_analysis):
    """第四步：建议生成（依赖第三步结果）"""
    prompt = f"""
    作为安防AI分析师，基于以下信息生成改进建议：
    
    数据概览：{data_summary}
    风险分析：{risk_analysis}
    
    请从以下方面提供建议：
    1. 立即行动项（24小时内）
    2. 短期改进（1周内）
    3. 长期优化（1个月内）
    4. 预防措施
    """
    return self._call_llm(prompt, "recommendations")
```

## Prompt Engineering 设计

### 1. 系统角色定义
```python
system_role = "你是一名专业的安防AI分析师，擅长分析监控数据和生成安全报告。请用简洁、专业的语言进行分析和总结。"
```

### 2. 模板化Prompt设计
```json
{
  "statistics_analysis": {
    "template": "作为安防AI分析师，请分析以下监控统计数据：\n\n数据概览：\n- 总告警数：{total_alerts}\n- 已处理：{handled_count}\n- 未处理：{unhandled_count}\n- 处理率：{handling_rate:.1f}%\n\n告警类型：{source_types}\n\n请从以下角度分析：\n1. 告警数量是否正常？\n2. 处理效率如何？\n3. 主要告警类型是什么？\n4. 有什么异常趋势？\n\n请用简洁的语言总结统计发现。",
    "variables": ["total_alerts", "handled_count", "unhandled_count", "handling_rate", "source_types"]
  }
}
```

### 3. 多模态Prompt集成
```python
def _generate_multimodal_content(self, data):
    """生成多模态内容"""
    if not self.qwen_vl:
        return {}
    
    multimodal_content = {
        'scene_descriptions': [],
        'key_events': []
    }
    
    # 选择典型告警进行场景描述
    typical_alerts = self._select_typical_alerts(data['alerts'])
    
    for alert in typical_alerts[:5]:
        if alert.get('images', {}).get('frame'):
            scene_desc = self.qwen_vl.generate_caption(alert['frame'])
            multimodal_content['scene_descriptions'].append({
                'alert_id': alert['id'],
                'description': scene_desc,
                'level': alert.get('danger_level', 'unknown'),
                'time': alert.get('datetime', '')
            })
    
    return multimodal_content
```

## 使用方法

### 1. 命令行使用
```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"

# 运行日报生成
python src/daily_report_generator.py

# 或使用批处理脚本（Windows）
run_daily_report.bat
```

### 2. API接口使用
```bash
# 生成日报
curl -X POST http://localhost:5000/api/daily-report/generate \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-01-15",
    "format": "markdown",
    "include_multimodal": true
  }'

# 获取状态
curl http://localhost:5000/api/daily-report/status

# 测试功能
curl -X POST http://localhost:5000/api/daily-report/test \
  -H "Content-Type: application/json" \
  -d '{"test_mode": "cot_analysis"}'
```

### 3. 测试脚本使用
```bash
# 运行完整测试
python test/test_daily_report.py

# 测试特定功能
python -c "
from test.test_daily_report import test_cot_analysis
test_cot_analysis()
"
```

## 配置说明

### 1. 基础配置
```json
{
  "api_settings": {
    "base_url": "http://localhost:5000",
    "timeout": 30
  },
  "llm_settings": {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "max_tokens": 1000,
    "temperature": 0.7,
    "api_key_env": "OPENAI_API_KEY"
  }
}
```

### 2. 多模态配置
```json
{
  "multimodal_settings": {
    "enable": true,
    "model": "Qwen/Qwen-VL-Chat",
    "device": "cpu",
    "max_scene_descriptions": 5
  }
}
```

### 3. 输出配置
```json
{
  "report_settings": {
    "output_dir": "reports",
    "format": "markdown",
    "include_images": false,
    "max_alerts_per_report": 100
  }
}
```

## 输出示例

### Markdown格式日报
```markdown
# 监控日报 - 2024-01-15

**生成时间**: 2024-01-16 09:30:00

---

## 执行摘要

今日监控系统共检测到15起告警事件，其中高风险事件3起，已处理12起，处理率80%。主要风险集中在人员摔倒检测和危险区域停留。

## 详细分析

### 统计分析
- 总告警数：15起
- 高风险告警：3起
- 中风险告警：8起
- 低风险告警：4起
- 处理率：80%

### 事件分析
1. **最严重事件**：上午10:30检测到人员摔倒，已及时处理
2. **时间分布**：告警主要集中在上午9-11点和下午2-4点
3. **重点关注**：危险区域停留事件频发，需要加强监控

### 风险分析
- **风险等级**：中等
- **主要风险点**：人员摔倒、危险区域停留
- **需要关注**：未处理的3起高风险事件
- **风险趋势**：相比昨日有所下降

## 行动建议

### 立即行动项（24小时内）
1. 处理剩余的3起高风险告警
2. 检查危险区域标识是否清晰

### 短期改进（1周内）
1. 优化危险区域监控算法
2. 加强人员培训，提高安全意识

### 长期优化（1个月内）
1. 升级监控设备，提升检测精度
2. 建立定期安全评估机制

### 预防措施
1. 定期检查监控设备状态
2. 建立应急预案，提高响应速度

---

*本报告由AI自动生成，仅供参考。如有疑问，请联系安防管理人员。*
```

## 扩展功能

### 1. 自定义Prompt模板
```python
# 在配置文件中添加自定义模板
"custom_templates": {
    "security_analysis": {
        "template": "请从安全角度分析以下数据：{data}",
        "variables": ["data"]
    }
}
```

### 2. 多语言支持
```python
# 支持中英文切换
def set_language(self, language):
    if language == 'en':
        self.system_role = "You are a professional security AI analyst..."
    else:
        self.system_role = "你是一名专业的安防AI分析师..."
```

### 3. 定时任务集成
```python
# 使用cron或APScheduler实现定时生成
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()
scheduler.add_job(generate_daily_report, 'cron', hour=9, minute=0)
scheduler.start()
```

## 故障排除

### 1. 常见问题
- **API密钥未设置**：确保设置了`OPENAI_API_KEY`环境变量
- **网络连接失败**：检查API基础URL是否正确
- **多模态模型加载失败**：检查模型路径和设备配置

### 2. 调试方法
```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 测试数据收集
curl http://localhost:5000/api/daily-report/test \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"test_mode": "data_collection"}'

# 测试CoT分析
curl http://localhost:5000/api/daily-report/test \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"test_mode": "cot_analysis"}'
```

## 性能优化

### 1. 缓存机制
```python
# 缓存LLM调用结果
@lru_cache(maxsize=100)
def cached_llm_call(self, prompt_hash):
    return self._call_llm(prompt_hash)
```

### 2. 并发处理
```python
# 并行处理多个分析步骤
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(self._analyze_statistics, data),
        executor.submit(self._analyze_events, data, multimodal_content)
    ]
    results = [future.result() for future in futures]
```

### 3. 批量处理
```python
# 批量生成多日报告
def generate_weekly_reports(self, start_date, end_date):
    for date in date_range(start_date, end_date):
        self.generate_daily_report(date.strftime('%Y-%m-%d'))
```

## 总结

通过CoT思维链 + Prompt Engineering的实现，AI监控日报生成系统能够：

1. **提供结构化分析**：分步骤推理，确保分析逻辑清晰
2. **生成高质量内容**：精心设计的Prompt模板，提升生成质量
3. **支持多模态集成**：结合视觉-语言模型，提供更丰富的场景描述
4. **易于扩展和维护**：模块化设计，便于功能扩展

该系统为监控安全管理提供了智能化的决策支持，大大提升了工作效率和报告质量。 