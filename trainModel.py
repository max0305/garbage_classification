from ultralytics import YOLO

# 載入YOLOv8模型
model = YOLO("yolov10m.pt")

# 開始訓練
if __name__ == '__main__':
    model.train(
        data = "data.yaml", 
        epochs = 60, 
        imgsz = 416, 
        augment = True
    )
