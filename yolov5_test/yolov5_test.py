import torch
from PIL import Image
import os
from pathlib import Path

# 加载模型及权重文件
model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.path.dirname(__file__), f'', 'best (4).pt'))

# 加载图片
img_path = os.path.join(os.path.dirname(__file__), f'', 't1.jpg')
image = Image.open(img_path)

# 进行目标检测
results = model(image)

# 处理检测结果
for detection in results.xyxy[0]:
    bbox = detection[:4]  # 边界框坐标
    conf = detection[4]  # 置信度
    label = int(detection[5])  # 类别标签
    print("Bounding Box:", bbox)
    print("Confidence:", conf)
    print("Label:", label)

# 可选：保存检测结果图像
out_path = os.path.join(os.path.dirname(__file__), f'result')
results.save(save_dir=out_path)
