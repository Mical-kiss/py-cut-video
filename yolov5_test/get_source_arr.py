import torch
# import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
# import ultralytics_yolov5_master
# import numpy as np
# from pathlib import Path
# from moviepy.editor import VideoFileClip, concatenate_videoclips

def get_source_arr():
  # 加载权重文件
  model = torch.hub.load(os.path.join(os.path.dirname(__file__), f'', './ultralytics_yolov5_master'), 'custom', path=os.path.join(os.path.dirname(__file__), f'', 'best (5).pt'), source='local')

  # 视频路径
  video_path = os.path.join(os.path.dirname(__file__), f'', '222222_1.mp4')

  # 视频读取器
  cap = cv2.VideoCapture(video_path)

  # 读取视频帧并进行目标检测
  frame_interval = 0.2  # 截取帧的时间间隔
  ball_results = []  # 存储球的检测结果
  net_results = []  # 存储球网的检测结果
  frame_count = 0  # 帧计数器

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break

      # 转换为 PIL 图像
      image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

      # 图像预处理
      # transform = transforms.Compose([
      #     transforms.Resize((640, 640)),  # 调整图像大小
      #     transforms.ToTensor()  # 转换为张量
      # ])
      # image = transform(image)

      # 使用模型进行目标检测
      results = model(image)
      print(results)

      # 处理检测结果
      ball_detection = []
      net_detection = []
      print(results.xyxy[0])
      for detection in results.xyxy[0]:
          bbox = detection[:4]  # 边界框坐标
          conf = detection[4]  # 置信度
          label = int(detection[5])  # 类别标签

          if label == 0:  # 类别为球
              ball_detection.append((bbox, conf))
          elif label == 1:  # 类别为球网
              net_detection.append((bbox, conf))

      # 根据置信度排序
      ball_detection.sort(key=lambda x: x[1], reverse=True)
      net_detection.sort(key=lambda x: x[1], reverse=True)

      # 存储结果
      ball_results.append(ball_detection[0] if ball_detection else [])
      net_results.append(net_detection[0] if net_detection else [])

      # 跳到下一个要截取的帧
      fps = cap.get(cv2.CAP_PROP_FPS)
      frame_skip = int(fps * frame_interval)
      print(f'ing... {cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_skip - 1}')
      cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_skip - 1)

  print('ball_results:', ball_results)
  print('net_results:', net_results)
  # 释放资源
  cap.release()
  cv2.destroyAllWindows()

  ball_boxes = []  # 存放球的边界框信息
  net_boxes = []  # 存放球网的边界框信息

  for ball_result in ball_results:
      if ball_result:  # 检查是否有球的检测结果
          bbox, conf = ball_result
          bbox_list = bbox.tolist()  # 将边界框张量转换为列表
          ball_boxes.append(bbox_list if conf > 0.5 else [])
      else:
          ball_boxes.append([])  # 使用空数组填充边界框信息

  for net_result in net_results:
      if net_result:  # 检查是否有球网的检测结果
          bbox, conf = net_result
          bbox_list = bbox.tolist()  # 将边界框张量转换为列表
          net_boxes.append(bbox_list)
      else:
          net_boxes.append([])  # 使用空数组填充边界框信息

  print('ball_results:', ball_boxes)
  print('net_results:', net_boxes)
  return [ball_boxes, net_boxes]

def pre_cal(ball_boxes, net_boxes):
  finialResults = []
  for ball_box, net_box in zip(ball_boxes, net_boxes):
      if ball_box and net_box:  # 检查是否有球和网的位置信息
          ball_center_x = (ball_box[0] + ball_box[2]) / 2  # 球的中心点 x 坐标
          net_left = net_box[0]  # 网的左边界 x 坐标
          net_right = net_box[2]  # 网的右边界 x 坐标
          net_bottom = net_box[3]  # 网的下边界 y 坐标
          # net_left = net_box[1]  # 网的左边界 x 坐标
          # net_right = net_box[3]  # 网的右边界 x 坐标
          # net_bottom = net_box[2]  # 网的下边界 y 坐标


          if net_left < ball_center_x < net_right and ball_box[3] < net_bottom:
              finialResults.append(1)
          else:
              finialResults.append(0)
      else:
          finialResults.append(0)

  print('finialResults:', len(finialResults), finialResults)
  return finialResults

""" video = VideoFileClip(video_path)
clip_index = 0
clip = []
for start, end in zip(start_indices, end_indices):
  video_clip = video.subclip(start, end)
  clip.append(video_clip)
  if len(clip) == 5:
    clip_index += 1
    final_clip = concatenate_videoclips(clip)
    output_path = os.path.join(os.path.dirname(__file__), "output", f"test_{clip_index}.mp4")
    final_clip.write_videofile(output_path)
    # clips.append(clip)
    clip = []
if clip:
  clip_index += 1
  final_clip = concatenate_videoclips(clip)
  output_path = os.path.join(os.path.dirname(__file__), "output", f"test_{clip_index}.mp4")
  final_clip.write_videofile(output_path)
 """

""" # 加载权重文件
model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.path.dirname(__file__), f'', 'best (4).pt'))

# 视频路径
video_path = os.path.join(os.path.dirname(__file__), f'', '231223_1.mp4')

# 视频读取器
cap = cv2.VideoCapture(video_path)

# 读取视频并进行目标检测
frame_interval = 0.2  # 截取帧的时间间隔
ball_results = []  # 存储球的检测结果
net_results = []  # 存储球网的检测结果

def process_frame(t, frame):
    if frame is None:
        return None  # 空帧
    # 转换为 PIL 图像
    image = Image.fromarray(frame)

    # 图像预处理
    # transform = transforms.Compose([
    #     transforms.Resize((640, 640)),  # 调整图像大小
    #     transforms.ToTensor()  # 转换为张量
    # ])
    # image = transform(image)

    # 使用模型进行目标检测
    results = model(image)

    # 处理检测结果
    ball_detection = []
    net_detection = []
    for detection in results.xyxy[0]:
        bbox = detection[:4]  # 边界框坐标
        conf = detection[4]  # 置信度
        label = int(detection[5])  # 类别标签

        if label == 0:  # 类别为球
            ball_detection.append((bbox, conf))
        elif label == 1:  # 类别为球网
            net_detection.append((bbox, conf))

    # 根据置信度排序
    ball_detection.sort(key=lambda x: x[1], reverse=True)
    net_detection.sort(key=lambda x: x[1], reverse=True)

    # 存储结果
    ball_results.append(ball_detection[0] if ball_detection else [])
    net_results.append(net_detection[0] if net_detection else [])

    return frame

# 使用 VideoFileClip 读取视频并处理每一帧
clip = VideoFileClip(video_path)
# processed_clip = clip.fl_time(process_frame, frame_interval)
processed_clip = clip.fl(process_frame)

# 保存处理后的视频
processed_video_path = os.path.join(os.path.dirname(__file__), f'', '231223_1_copy.mp4')
processed_clip.write_videofile(processed_video_path, codec="libx264")
print('ball_results:', ball_results)
print('net_results:', net_results)

# 释放资源
clip.reader.close()
clip.audio.reader.close_proc() """