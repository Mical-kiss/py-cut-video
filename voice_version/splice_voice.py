import librosa
import soundfile as sf
import numpy as np
import os

def splice_voice(filename):
  # 加载音频文件
  audio_file = os.path.join(os.path.dirname(__file__), "data", f"voice/{filename}.wav")
  splice_file = os.path.join(os.path.dirname(__file__), "data", f"splice_voice/{filename}.wav")
  if os.path.exists(splice_file):
    return

  y, sr = librosa.load(audio_file)

  # 应用频率特性分析（例如傅里叶变换）
  D = librosa.stft(y)

  # 分离不同的音色
  # 在这里，你可以根据需要使用不同的分离技术，例如基于频率的滤波、独立成分分析（ICA）等

  # 获取频率数组
  n_fft = D.shape[0]
  frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

  # 创建用于掩码的零数组
  mask = np.zeros_like(D)

  # 示例：选择乒乓球声音的频率范围（根据实际情况调整）
  pingpong_freq_range = (4000, 4200)  # 1000 Hz 到 3000 Hz
  pingpong_indices = np.where((frequencies >= pingpong_freq_range[0]) & (frequencies <= pingpong_freq_range[1]))[0]
  mask[pingpong_indices, :] = 1.0

  # 应用掩码
  masked_D = D * mask

  # 逆向傅里叶变换，将分离的音频信号转换回时域
  separated_audio = librosa.istft(masked_D)

  sf.write(splice_file, separated_audio, sr)
