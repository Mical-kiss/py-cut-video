from moviepy.editor import VideoFileClip
import os

def get_voice(filename):
  # 指定输入视频文件路径
  video_path = os.path.join(os.path.dirname(__file__), "data", f"video/{filename}.mp4")
  audio_file = os.path.join(os.path.dirname(__file__), "data", f"voice/{filename}.wav")
  if os.path.exists(audio_file):
    return

  video_file = video_path

  # 加载视频文件
  clip = VideoFileClip(video_file)

  # 提取音频并保存为.wav格式
  audio = clip.audio
  audio.write_audiofile(audio_file)
