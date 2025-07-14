import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
import queue
import threading
import time
import os
import sys
import importlib
import tensorflow_hub as hub

# 路径配置（请根据实际情况调整）
YAMNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yamnet.h5')  # 需提前下载YAMNet权重文件
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), 'yamnet_class_map.csv')  # 需提前下载类别映射表

# 读取类别映射
def load_class_map(path):
    import csv
    class_names = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2])
    return class_names

yamnet_classes = load_class_map(CLASS_MAP_PATH)

# 优先本地加载YAMNet模型
YAMNET_LOCAL_DIR = os.path.join(os.path.dirname(__file__), 'yamnet_1')
try:
    if os.path.exists(YAMNET_LOCAL_DIR):
        yamnet_model = hub.load(YAMNET_LOCAL_DIR)
        print(f"[音频监控] 已从本地加载YAMNet模型: {YAMNET_LOCAL_DIR}")
    else:
        print("[音频监控] 未找到本地YAMNet模型，尝试在线加载...")
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("[音频监控] 已在线加载YAMNet模型")
except Exception as e:
    print(f"[音频监控] 加载YAMNet模型失败: {e}\n声学检测功能将被禁用。")
    yamnet_model = None

# 你关心的类别
TARGET_KEYWORDS = ['Scream', 'Shout', 'Yell', 'Fight', 'Argument', 'Siren', 'Emergency vehicle', 'Whistle', 'Speech', 'Child speech', 'Children shouting', 'Screaming']

# 声学事件检测
def detect_audio_event(audio_data, sr):
    # yamnet_model 为空时直接返回
    if yamnet_model is None:
        return None, None
    # 预处理到16kHz单通道
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    # 保证waveform是一维float32
    waveform = audio_data.astype(np.float32)
    try:
        scores, embeddings, spectrogram = yamnet_model(waveform)
        mean_scores = np.mean(scores, axis=0)
        top_class = np.argmax(mean_scores)
        top_score = mean_scores[top_class]
        top_label = yamnet_classes[top_class]
        # 检查是否为目标事件
        for kw in TARGET_KEYWORDS:
            if kw.lower() in top_label.lower() and top_score > 0.2:
                return top_label, float(top_score)
        return None, None
    except Exception as e:
        print("[音频监控] 声学事件检测错误:", str(e))
        return None, None

# 音频监控主循环
def audio_monitor_callback(alert_callback, duration=1, samplerate=16000):
    q = queue.Queue()
    def audio_callback(indata, frames, time, status):
        q.put(indata.copy())
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate):
        print("音频监控中...")
        while True:
            audio_chunk = q.get()
            audio_data = audio_chunk.flatten()
            label, score = detect_audio_event(audio_data, samplerate)
            if label:
                print(f"检测到异常声音: {label} (置信度: {score:.2f})")
                alert_callback(label, score)

# 集成到主系统：自动查找AllInOneSystem实例
main_system = None
try:
    sys.path.append('src')
    all_in_one = importlib.import_module('all_in_one_system')
    # 假设主系统实例为all_in_one_system.AllInOneSystem._instance或全局变量system
    if hasattr(all_in_one, 'system'):
        main_system = all_in_one.system
    elif hasattr(all_in_one, 'AllInOneSystem'):
        # 可根据实际情况获取主系统实例
        pass
except Exception as e:
    print(f"[音频监控] 未能自动获取主系统实例: {e}")

# 修改alert_callback，推送到主系统

def alert_callback(label, score):
    if main_system and hasattr(main_system, 'add_audio_alert'):
        main_system.add_audio_alert(label, score)
    else:
        print(f"[ALERT] 声学异常: {label} (置信度: {score:.2f})")

if __name__ == '__main__':
    audio_thread = threading.Thread(target=audio_monitor_callback, args=(alert_callback,))
    audio_thread.daemon = True
    audio_thread.start()
    while True:
        time.sleep(1) 