import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
import queue
import threading
import time as time_module
import os
import sys
import importlib
import tensorflow_hub as hub
import collections

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

# 声学检测参数
NOISE_DETECT_THRESHOLD = 0.01  # 声音置信度阈值（极低，调试用）
NOISE_ACCUM_SECONDS = 10.0     # 喧哗累计时长阈值（秒）
NOISE_VOLUME_THRESHOLD = 0.15  # 音量阈值（需根据实际环境调整）
NOISE_REQUIRED_RATIO = 0.5     # 10秒内有一半时间为噪音
NOISE_WINDOW_SIZE = 30         # 30秒滑动窗口，假设每秒采样一次
NOISE_REQUIRED_SECONDS = 25    # 30秒内有25秒为噪音才告警
noise_window = collections.deque(maxlen=NOISE_WINDOW_SIZE)

# 声学事件检测
def detect_audio_event(audio_data, sr):
    # yamnet_model 为空时直接返回
    if yamnet_model is None:
        return [], []
    # 预处理到16kHz单通道
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    # 保证waveform是一维float32
    waveform = audio_data.astype(np.float32)
    if yamnet_model is None:
        return [], []
    try:
        scores, embeddings, spectrogram = yamnet_model(waveform)
        mean_scores = np.mean(scores, axis=0)
        detected_labels = []
        detected_scores = []
        for i, score in enumerate(mean_scores):
            label = yamnet_classes[i]
            for kw in TARGET_KEYWORDS:
                if kw.lower() in label.lower() and score > 0.2:
                    detected_labels.append(label)
                    detected_scores.append(float(score))
        return detected_labels, detected_scores
    except Exception as e:
        print("[音频监控] 声学事件检测错误:", str(e))
        return [], []

# 音频监控主循环
def audio_monitor_callback(alert_callback, duration=1, samplerate=16000):
    q = queue.Queue()
    cooldown_seconds = 25  # 声学告警冷却时间（秒），与30秒窗口配合，避免频繁告警
    last_alert_time = 0
    _noise_accum_active = False
    _noise_accum_start = None
    _noise_accum_seconds = 0.0
    _last_noise_time = None
    global noise_window
    def audio_callback(indata, frames, t, status):  # 避免参数名time与time_module冲突
        q.put(indata.copy())
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate):
        print("音频监控中...")
        while True:
            audio_chunk = q.get()
            audio_data = audio_chunk.flatten()
            labels, scores = detect_audio_event(audio_data, samplerate)
            now = time_module.time()  # 避免与sounddevice的time冲突
            # 计算音量（分贝）
            rms = np.sqrt(np.mean(audio_data ** 2))
            max_db = 20 * np.log10(np.max(np.abs(audio_data)) + 1e-8)
            min_db = 20 * np.log10(np.min(np.abs(audio_data)) + 1e-8)
            avg_db = 20 * np.log10(rms + 1e-8)
            # 兼容原有音量阈值
            volume = float(np.max(np.abs(audio_data)))
            is_noisy = volume > NOISE_VOLUME_THRESHOLD
            noise_window.append(is_noisy)
            # 每秒采样一次，30秒窗口内有25秒为噪音则告警
            if len(noise_window) == NOISE_WINDOW_SIZE and sum(noise_window) >= NOISE_REQUIRED_SECONDS:
                if now - last_alert_time > cooldown_seconds:
                    print(f"检测到30秒内有25秒为噪音，触发Classroom Noise告警，音量={volume:.3f}")
                    alert_callback(['Classroom Noise'], [volume], {
                        'avg_db': round(float(avg_db), 1),
                        'max_db': round(float(max_db), 1),
                        'min_db': round(float(min_db), 1)
                    })
                    last_alert_time = now
                    noise_window.clear()
            # 兼容原有Speech累计逻辑（可选）
            # if 'Speech' in labels:
            #     ...
            if labels and (now - last_alert_time > cooldown_seconds):
                print(f"检测到异常声音: {labels} (置信度: {scores})")
                alert_callback(labels, scores, {
                    'avg_db': round(float(avg_db), 1),
                    'max_db': round(float(max_db), 1),
                    'min_db': round(float(min_db), 1)
                })
                last_alert_time = now

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

def alert_callback(labels, scores, audio_db_stats=None):
    if main_system and hasattr(main_system, 'add_audio_alert'):
        main_system.add_audio_alert(labels, scores, audio_db_stats)
    else:
        print(f"[ALERT] 声学异常: {labels} (置信度: {scores}), 分贝统计: {audio_db_stats}")

if __name__ == '__main__':
    audio_thread = threading.Thread(target=audio_monitor_callback, args=(alert_callback,))
    audio_thread.daemon = True
    audio_thread.start()
    while True:
        time_module.sleep(1) 