@echo off
echo 启动带有危险等级告警系统的视频监控
echo.

cd /d "%~dp0"

echo 检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python，请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

echo.
echo 启动全功能视频监控系统（危险等级告警）...
echo 系统将启动Web界面，访问 http://localhost:5000 查看告警
echo.

python src/all_in_one_system.py ^
    --source 0 ^
    --width 640 ^
    --height 480 ^
    --max-fps 30 ^
    --web-interface ^
    --web-port 5000 ^
    --feature-threshold 30 ^
    --area-threshold 0.2 ^
    --alert-cooldown 5 ^
    --min-confidence 0.5 ^
    --enable-ai ^
    --ai-confidence 0.5 ^
    --ai-interval 10 ^
    --use-motion-history ^
    --save-alerts ^
    --output system_output ^
    --alert-region "[[100,100],[300,100],[300,300],[100,300]]"

echo.
echo 系统已停止
pause 