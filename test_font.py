# test_font.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


# --- 这里是我们唯一要测试的函数 ---
def draw_text_with_chinese(frame, text, position, font_path, font_size, color):
    # 检查字体文件是否存在
    if not os.path.exists(font_path):
        print(f"!!! 致命错误: 在路径 '{font_path}' 未找到字体文件。")
        return frame

    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        print(">>> 成功: 字体已加载, 文字已绘制。")
        return frame
    except Exception as e:
        print(f"!!! 致命错误: 绘制中文时出错: {e}")
        return frame


# --- 这里设置要测试的字体路径 (请确保它和您主程序中的路径完全一样) ---
TEST_FONT_PATH = "src/assets/msyh.ttc"

# --- 测试主逻辑 ---
if __name__ == '__main__':
    print("开始字体功能独立测试...")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"将要测试的字体路径: {TEST_FONT_PATH}")

    # 创建一个黑色的背景图
    test_image = np.zeros((200, 700, 3), dtype=np.uint8)

    # 尝试用函数绘制中文
    test_image = draw_text_with_chinese(
        frame=test_image,
        text="中文测试：如果能看到我，说明字体和路径都正确。",
        position=(10, 50),
        font_path=TEST_FONT_PATH,
        font_size=24,
        color=(0, 255, 0)  # 绿色
    )

    # 显示结果图像
    cv2.imshow("Font Test Result", test_image)
    print("请查看弹出的'Font Test Result'窗口。按任意键关闭。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()