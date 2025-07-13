@echo off
chcp 65001 >nul
echo ========================================
echo 多边形警戒区域功能测试
echo ========================================
echo.

echo 正在运行多边形区域测试...
python test_polygon_region.py

echo.
echo 测试完成！
echo 请查看生成的测试图像文件：test_polygon_region.jpg
echo.
pause 