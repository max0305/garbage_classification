import cv2
import os

# 確認檔案是否存在
video_path = 'C:/garbage_classification/Garbage.mp4'
if os.path.exists(video_path):
    print("y")
else:
    print("n")

cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    print("s")
else:
    print("f")