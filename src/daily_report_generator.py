#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI自动生成监控日报模块
使用CoT思维链 + Prompt Engineering实现智能日报生成
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DailyReportGenerator:
    """AI监控日报生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化日报生成器
        
        Args:
            config: 配置字典，包含API密钥、模型设置等
        """
        self.config = config
        self.api_base_url = config.get('api_base_url', 'http://localhost:5000')
        
        # 初始化大模型客户端
        self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化大语言模型客户端"""
        provider = self.config.get('llm_settings', {}).get('provider', 'deepseek').lower()
        self.llm_provider = provider
        if provider == 'deepseek':
            self.deepseek_api_key = os.getenv(self.config['llm_settings'].get('api_key_env', 'DEEPSEEK_API_KEY'))
            self.deepseek_api_url = "https://api.deepseek.com/chat/completions"
        else:
            raise ValueError(f"不支持的LLM provider: {provider}")
    
    def generate_daily_report(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        生成指定日期的监控日报
        
        Args:
            target_date: 目标日期，格式为'YYYY-MM-DD'，默认为昨天
            
        Returns:
            包含日报内容的字典
        """
        if target_date is None:
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"开始生成 {target_date} 的监控日报")
        
        # Step 1: 数据收集
        data = self._collect_daily_data(target_date)
        
        # Step 2: 多模态内容生成（如果有图片）
        multimodal_content = self._generate_multimodal_content(data)
        
        # Step 3: CoT思维链分析
        analysis_results = self._chain_of_thought_analysis(data, multimodal_content)
        
        # Step 4: 生成最终日报
        final_report = self._generate_final_report(analysis_results, target_date)
        
        return final_report
    
    def _collect_daily_data(self, target_date: str) -> Dict[str, Any]:
        """收集日报所需的数据"""
        try:
            # 获取告警数据
            alerts_response = requests.get(
                f"{self.api_base_url}/api/alerts/history",
                params={
                    'start_date': f"{target_date} 00:00:00",
                    'end_date': f"{target_date} 23:59:59",
                    'limit': 1000
                }
            )
            alerts_data = alerts_response.json() if alerts_response.status_code == 200 else {'alerts': []}
            
            # 获取统计数据
            stats_response = requests.get(f"{self.api_base_url}/api/alerts/statistics")
            stats_data = stats_response.json() if stats_response.status_code == 200 else {}
            
            # 获取来源类型
            source_types_response = requests.get(f"{self.api_base_url}/api/alerts/source_types")
            source_types = source_types_response.json() if source_types_response.status_code == 200 else {'types': []}
            
            return {
                'date': target_date,
                'alerts': alerts_data.get('alerts', []),
                'statistics': stats_data.get('stats', {}),
                'source_types': source_types.get('types', []),
                'total_alerts': len(alerts_data.get('alerts', [])),
                'unhandled_count': len([a for a in alerts_data.get('alerts', []) 
                                      if not a.get('acknowledged', False)]),
                'handled_count': len([a for a in alerts_data.get('alerts', []) 
                                    if a.get('acknowledged', False)])
            }
        except Exception as e:
            logger.error(f"数据收集失败: {e}")
            return {'date': target_date, 'alerts': [], 'statistics': {}, 'source_types': []}
    
    def _generate_multimodal_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成多模态内容（场景描述等）"""
        # 移除多模态相关内容
        # if not self.qwen_vl:
        #     return {}
        
        multimodal_content = {
            'scene_descriptions': [],
            'key_events': []
        }
        
        try:
            # 选择典型告警进行场景描述
            typical_alerts = self._select_typical_alerts(data['alerts'])
            
            for alert in typical_alerts[:5]:  # 限制数量避免处理过慢
                if alert.get('images', {}).get('frame'):
                    try:
                        # 这里需要根据实际图片路径获取图片
                        # 暂时用文本描述代替
                        scene_desc = f"告警ID {alert['id']}: {alert.get('message', '无描述')}"
                        multimodal_content['scene_descriptions'].append({
                            'alert_id': alert['id'],
                            'description': scene_desc,
                            'level': alert.get('danger_level', 'unknown'),
                            'time': alert.get('datetime', '')
                        })
                    except Exception as e:
                        logger.warning(f"生成场景描述失败: {e}")
        
        except Exception as e:
            logger.error(f"多模态内容生成失败: {e}")
        
        return multimodal_content
    
    def _select_typical_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """选择典型告警"""
        if not alerts:
            return []
        
        # 按危险等级和重要性排序
        sorted_alerts = sorted(alerts, 
                             key=lambda x: (
                                 {'high': 3, 'medium': 2, 'low': 1}.get(x.get('danger_level', 'low'), 0),
                                 not x.get('acknowledged', False)  # 未处理的优先
                             ), 
                             reverse=True)
        
        return sorted_alerts[:10]  # 返回前10个典型告警
    
    def _chain_of_thought_analysis(self, data: Dict[str, Any], 
                                  multimodal_content: Dict[str, Any]) -> Dict[str, Any]:
        """CoT思维链分析"""
        
        # Step 1: 统计分析
        stats_analysis = self._analyze_statistics(data)
        
        # Step 2: 事件分析
        events_analysis = self._analyze_events(data, multimodal_content)
        
        # Step 3: 风险分析
        risk_analysis = self._analyze_risks(data, events_analysis)
        
        # Step 4: 建议生成
        recommendations = self._generate_recommendations(data, risk_analysis)
        
        return {
            'statistics': stats_analysis,
            'events': events_analysis,
            'risks': risk_analysis,
            'recommendations': recommendations
        }
    
    def _analyze_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """统计分析"""
        prompt = f"""
作为安防AI分析师，请分析以下监控统计数据：

数据概览：
- 总告警数：{data['total_alerts']}
- 已处理：{data['handled_count']}
- 未处理：{data['unhandled_count']}
- 处理率：{data['handled_count']/data['total_alerts']*100:.1f}% (如果总数>0)

告警类型：{data['source_types']}

请从以下角度分析：
1. 告警数量是否正常？
2. 处理效率如何？
3. 主要告警类型是什么？
4. 有什么异常趋势？

请用简洁的语言总结统计发现。
"""
        
        return self._call_llm(prompt, "statistics_analysis")
    
    def _analyze_events(self, data: Dict[str, Any], 
                       multimodal_content: Dict[str, Any]) -> Dict[str, Any]:
        """事件分析"""
        
        # 准备事件数据
        events_summary = []
        for alert in data['alerts'][:20]:  # 分析前20个告警
            events_summary.append({
                'id': alert['id'],
                'type': alert.get('source_type', 'unknown'),
                'level': alert.get('danger_level', 'unknown'),
                'time': alert.get('datetime', ''),
                'message': alert.get('message', ''),
                'handled': alert.get('acknowledged', False)
            })
        
        # 场景描述
        scene_descriptions = multimodal_content.get('scene_descriptions', [])
        
        prompt = f"""
作为安防AI分析师，请分析以下监控事件：

事件列表（前20个）：
{json.dumps(events_summary, ensure_ascii=False, indent=2)}

场景描述：
{json.dumps(scene_descriptions, ensure_ascii=False, indent=2)}

请从以下角度分析：
1. 最严重的事件是什么？
2. 事件的时间分布有什么特点？
3. 哪些事件需要重点关注？
4. 有什么异常模式？

请用简洁的语言总结事件分析。
"""
        
        return self._call_llm(prompt, "events_analysis")
    
    def _analyze_risks(self, data: Dict[str, Any], 
                      events_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """风险分析"""
        
        # 计算风险指标
        risk_metrics = {
            'high_risk_count': len([a for a in data['alerts'] if a.get('danger_level') == 'high']),
            'unhandled_high_risk': len([a for a in data['alerts'] 
                                       if a.get('danger_level') == 'high' and not a.get('acknowledged', False)]),
            'total_alerts': data['total_alerts'],
            'handling_rate': data['handled_count'] / max(data['total_alerts'], 1) * 100
        }
        
        prompt = f"""
作为安防AI分析师，请评估以下风险指标：

风险指标：
- 高风险告警：{risk_metrics['high_risk_count']}个
- 未处理高风险：{risk_metrics['unhandled_high_risk']}个
- 总告警数：{risk_metrics['total_alerts']}个
- 处理率：{risk_metrics['handling_rate']:.1f}%

事件分析结果：
{events_analysis.get('summary', '无')}

请评估：
1. 当前风险等级（低/中/高）
2. 主要风险点是什么？
3. 需要立即关注的问题？
4. 风险趋势如何？

请用简洁的语言总结风险评估。
"""
        
        return self._call_llm(prompt, "risk_analysis")
    
    def _generate_recommendations(self, data: Dict[str, Any], 
                                risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成建议"""
        
        prompt = f"""
作为安防AI分析师，基于以下信息生成改进建议：

数据概览：
- 总告警：{data['total_alerts']}个
- 未处理：{data['unhandled_count']}个
- 处理率：{data['handled_count']/max(data['total_alerts'], 1)*100:.1f}%

风险分析：
{risk_analysis.get('summary', '无')}

请从以下方面提供建议：
1. 立即行动项（24小时内）
2. 短期改进（1周内）
3. 长期优化（1个月内）
4. 预防措施

请用简洁的语言总结建议。
"""
        
        return self._call_llm(prompt, "recommendations")
    
    def _call_llm(self, prompt: str, task_name: str) -> Dict[str, Any]:
        if self.llm_provider == 'deepseek':
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-chat",  # 强制使用官方推荐模型名
                "messages": [
                    {"role": "system", "content": self.config['prompt_templates']['system_role']},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config['llm_settings'].get('max_tokens', 1000),
                "temperature": self.config['llm_settings'].get('temperature', 0.7),
                "stream": False
            }
            try:
                resp = requests.post(self.deepseek_api_url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                result = resp.json()
                content = result['choices'][0]['message']['content']
                return {'summary': content}
            except Exception as e:
                logger.error(f"DeepSeek LLM调用失败: {e}")
                return {'summary': f'DeepSeek LLM调用失败: {str(e)}'}
        else:
            return {'summary': f'不支持的LLM客户端: {self.llm_provider}'}
    
    def _generate_final_report(self, analysis_results: Dict[str, Any], 
                             target_date: str) -> Dict[str, Any]:
        """生成最终日报"""
        
        prompt = f"""
作为安防AI分析师，请基于以下分析结果生成一份完整的监控日报：

日期：{target_date}

分析结果：
- 统计分析：{analysis_results['statistics'].get('summary', '无')}
- 事件分析：{analysis_results['events'].get('summary', '无')}
- 风险分析：{analysis_results['risks'].get('summary', '无')}
- 改进建议：{analysis_results['recommendations'].get('summary', '无')}

请生成一份结构化的日报，包含：
1. 日报标题和日期
2. 执行摘要（2-3句话总结）
3. 详细分析（分点列出）
4. 风险等级评估
5. 行动建议
6. 后续跟进事项

格式要求：使用Markdown格式，结构清晰，语言专业简洁。
"""
        
        final_content = self._call_llm(prompt, "final_report")
        
        return {
            'date': target_date,
            'generated_at': datetime.now().isoformat(),
            'content': final_content.get('summary', '日报生成失败'),
            'analysis_results': analysis_results
        }
    
    def save_report(self, report: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """保存日报到文件"""
        if output_path is None:
            output_path = f"reports/daily_report_{report['date']}.md"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        content = report['content']
        # 去除最外层的 ```markdown ... ``` 包裹
        content = re.sub(r'^```markdown\s*([\s\S]*?)\s*```$', r'\1', content.strip(), flags=re.MULTILINE)
        # 生成完整的Markdown内容
        markdown_content = f"""# 监控日报 - {report['date']}

**生成时间**: {report['generated_at']}

---

{content}

---

*本报告由AI自动生成，仅供参考。如有疑问，请联系安防管理人员。*
"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"日报已保存到: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"保存日报失败: {e}")
            return ""

def main():
    """主函数 - 示例用法"""
    config_path = "src/config/daily_report_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    generator = DailyReportGenerator(config)
    # 生成昨天的日报
    report = generator.generate_daily_report()
    # 保存日报
    output_path = generator.save_report(report)
    print(f"日报生成完成: {output_path}")
    print("日报内容预览:")
    print("-" * 50)
    print(report['content'][:500] + "...")

if __name__ == "__main__":
    main() 