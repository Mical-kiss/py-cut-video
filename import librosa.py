import librosa
import numpy as np

# 加载音频文件
audio_path = './test.mp3'
audio, sr = librosa.load(audio_path, sr=None)

# 计算音频的能量
energy = librosa.feature.rms(audio, frame_length=1024, hop_length=512)[0]

# 设置能量阈值和回合最小持续时间
energy_threshold = np.mean(energy) + np.std(energy)
min_round_duration = 0.5  # 最小回合持续时间（以秒为单位）

# 检测回合的开始和结束时间
round_start_time = []
round_end_time = []
is_round = False
round_start = 0

for i in range(len(energy)):
    if energy[i] >= energy_threshold and not is_round:
        is_round = True
        round_start = i * 512 / sr
    elif energy[i] < energy_threshold and is_round:
        is_round = False
        round_end = i * 512 / sr
        if round_end - round_start >= min_round_duration:
            round_start_time.append(round_start)
            round_end_time.append(round_end)
        else:
            round_start = round_end  # 重置回合开始时间为当前时间点

# 打印回合的开始和结束时间
for start, end in zip(round_start_time, round_end_time):
    print("Round Start:", start, "Round End:", end)