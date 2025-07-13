@echo off
chcp 65001 >nul
echo ========================================
echo 视频监控系统 - Web界面版本
echo ========================================
echo.

echo 正在启动Web界面系统...
echo 系统将在浏览器中打开，地址: http://localhost:5000
echo.

echo 功能说明:
echo - 右侧面板可以设置停留时间阈值
echo - 点击"开始框选"后在视频上拖拽鼠标框选警戒区域
echo - 点击"重置区域"可以清除警戒区域
echo - 按Ctrl+C停止系统
echo.

python src/all_in_one_system.py --web_interface --source 0 --width 640 --height 480 --max_fps 30

echo.
echo 系统已停止
pause 