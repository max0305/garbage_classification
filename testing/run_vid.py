import cv2
from ultralytics import YOLO
# 加載訓練好的 YOLOv8 模型
model = YOLO("runs/detect/train4/weights/best.pt")  # 使用訓練產出的權重

# 載入影片
video_path = "C:/garbage_classification/Garbage.mp4"
cap = cv2.VideoCapture(video_path)

# 設置顯示窗口的大小
cv2.namedWindow("YOLOv8 Video Detection", cv2.WINDOW_NORMAL)
while cap.isOpened():
    print('open')
    ret, frame = cap.read()
    if not ret:
        break

    # 使用模型進行推論
    results = model(frame)
    
    # 繪製檢測結果
    annotated_frame = results[0].plot()  # 將結果繪製在圖片上

    # 顯示結果
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)
    
    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
