import matplotlib.pyplot as plt
import librosa
import numpy as np
import math
import os

def splice_video(filename):
  # 加载音频文件
  audio_path = os.path.join(os.path.dirname(__file__), "data", f"splice_voice/{filename}.wav")
  audio, sr = librosa.load(audio_path, sr=None)

  print("sr:", len(audio), sr, len(audio) / sr)

  ration = sr # 帧长度

  # 计算音频的能量
  energy = librosa.feature.rms(y=audio, frame_length=math.floor(ration), hop_length=math.floor(ration / 2))[0]
  print("energy_length:", len(energy))

  # 计算最大值和最小值
  min_val = np.min(energy)
  max_val = np.max(energy)

  # 将最大值和最小值之间的范围等分成 每0.005额度一份
  # num = (max_val - min_val) / 0.005
  bin_edges = np.linspace(min_val, max_val, num=math.floor((max_val - min_val) / 0.0025))
  # 计算每个区间的数据个数
  hist, _ = np.histogram(energy, bins=bin_edges)
  # 找到包含最多数据的区间
  max_freq_index = np.argmax(hist)
  most_frequent_bin = (bin_edges[max_freq_index + 0] + bin_edges[max_freq_index + 1]) / 2

  energy_threshold = most_frequent_bin

  print("energy_threshold:", energy_threshold)
  min_round_duration = 4.5  # 最小回合持续时间（以秒为单位）
  consecutive_below_threshold = 0  # 连续低于阈值的计数

  # 检测回合的开始和结束时间
  round_start_time = []
  round_end_time = []
  is_round = False
  round_start = 0

  for i in range(len(energy)):
      if energy[i] >= energy_threshold and not is_round:
          is_round = True
          round_start = (i - 3) * (ration / 2) / sr
          if round_start < 0:
            round_start = 0
          consecutive_below_threshold = 0
      elif energy[i] < energy_threshold and is_round:
          consecutive_below_threshold += 1
          if consecutive_below_threshold >= 3:
              is_round = False
              round_end = (i - 1) * (ration / 2) / sr
              if round_end - round_start >= min_round_duration:
                  round_start_time.append(round(round_start, 2))
                  round_end_time.append(round(round_end, 2))
      elif energy[i] >= energy_threshold and consecutive_below_threshold > 0:
          consecutive_below_threshold = 0
      

  def seconds_to_minutes(seconds):
      minutes = seconds // 60  # 计算分钟数
      seconds_remainder = seconds % 60  # 计算剩余的秒数

      result = f"{minutes}m{seconds_remainder}s"  # 格式化输出

      return result

  # 打印回合的开始和结束时间
  for start, end in zip(round_start_time, round_end_time):
      print("Round Start:", seconds_to_minutes(start), "Round End:", seconds_to_minutes(end))

  """ # 绘制折线图
  subset = energy[:60]
  plt.plot(range(len(subset)), subset)

  # 设置图表标题和轴标签
  plt.title('line')
  plt.xlabel('time')
  plt.ylabel('energy')
  plt.show() """

  from moviepy.editor import VideoFileClip, concatenate_videoclips

  # 原始视频文件路径
  original_video_path = os.path.join(os.path.dirname(__file__), "data", f"video/{filename}.mp4")

  clips = []
  clip = []
  clip_index = 0
  video = VideoFileClip(original_video_path)

  for start, end in zip(round_start_time, round_end_time):
    video_clip = video.subclip(start, end)
    # audio_clip = video_clip.audio
    # video_clip.set_audio(audio_clip)
    clip.append(video_clip)
    if len(clip) == 5:
      clip_index += 1
      final_clip = concatenate_videoclips(clip)
      output_path = os.path.join(os.path.dirname(__file__), "data", f"output/{filename}_{clip_index}.mp4")
      final_clip.write_videofile(output_path)
      # clips.append(clip)
      clip = []
  if clip:
    clip_index += 1
    clips.append(clip)
    final_clip = concatenate_videoclips(clip)
    output_path = os.path.join(os.path.dirname(__file__), "data", f"output/{filename}_{clip_index}.mp4")
    final_clip.write_videofile(output_path)
  
  """ for clip1 in range(len(clips)):
    print('test: ', clip1, len(clips[clip1]))
    # 按顺序合成剪辑后的视频
    final_clip = concatenate_videoclips(clips[clip1])

    print("w&&h:", video_clip.size[1], video_clip.size[0])
    final_clip = final_clip.resize((video_clip.size[0], video_clip.size[1]))
    # 输出合成后的视频文件
    output_path = os.path.join(os.path.dirname(__file__), "data", f"output/{filename}_{clip1}.mp4")
    final_clip.write_videofile(output_path) """
