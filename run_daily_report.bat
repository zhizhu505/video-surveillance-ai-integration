@echo off
echo 启动AI监控日报生成系统...

:: 设置环境变量
set PYTHONPATH=%PYTHONPATH%;%CD%

:: 检查OpenAI API密钥
if "%OPENAI_API_KEY%"=="" (
    echo 警告: 未设置OPENAI_API_KEY环境变量
    echo 请设置您的OpenAI API密钥:
    echo setx OPENAI_API_KEY "your-api-key-here"
    echo.
)

:: 运行日报生成
echo 正在生成监控日报...
python src/daily_report_generator.py

echo.
echo 日报生成完成！
echo 生成的报告保存在 reports/ 目录下
pause 