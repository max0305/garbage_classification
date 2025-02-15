import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

# 加載訓練好的 YOLO 模型
model = YOLO("runs/detect/train7/weights/best.pt")  # 使用訓練後的權重

# 設置 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()

# 設定要啟用的流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # RGB 影像流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度流

# 開始管道
pipeline.start(config)

# 創建對齊物件，對齊 **深度影像到 RGB 影像**
align = rs.align(rs.stream.color)

# 創建 OpenCV 顯示窗口
cv2.namedWindow("RealSense YOLO Detection", cv2.WINDOW_NORMAL)

try:
    while True:
        # 獲取影像幀（彩色與深度）
        frames = pipeline.wait_for_frames()

        # 使用 `rs.align` 讓深度影像與 RGB 影像對齊
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # depth_intrinsics 中就包含 fx, fy, ppx, ppy, distortion 等資訊 
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

        if not color_frame or not depth_frame:
            continue

        # 將影像轉換為 NumPy 陣列
        frame = np.asanyarray(color_frame.get_data())

        # 使用 YOLO 模型進行推論
        results = model(frame)

        # 拿到推論結果（第一張圖對應 results[0]）
        det = results[0]
        boxes = det.boxes  # YOLO v8 中會以 boxes 屬性存放偵測到的物件

        # 繪製檢測結果 (YOLO 內建函式先畫出 Bounding Box 與標籤)
        annotated_frame = det.plot()
        # 轉成可寫
        annotated_frame = annotated_frame.copy()

        # 自行在 Bounding Box 上方加上深度資訊
        # boxes 中的每個元素皆包含 [x1, y1, x2, y2, conf, cls] (若使用默認設定)
        for box in boxes:
            # box.xyxy[0] 會是 [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0]

            # 取得 Bounding Box 的中心點 (取整數避免 get_distance 出錯)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # 從 depth_frame 取得該像素的深度 (單位：公尺)
            distance = depth_frame.get_distance(center_x, center_y)

            # 取得3D座標
            point_3d = rs.rs2_deproject_pixel_to_point(
                depth_intrinsics, 
                [center_x, center_y], 
                distance
            )

            # 以文字顯示 (距離取到小數點第二位)
            text = f"{distance:.2f} m"
            # 決定文字要顯示的位置
            text_pos = (int(x1), int(y1) - 25)

            # 在畫面上繪製深度資訊
            cv2.putText(
                annotated_frame, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
            )

            X, Y, Z = point_3d  # 單位：公尺
            Y = -Y

            # 3. 組合顯示的字串，可依需求調整小數位數
            text_3d = f"X={X:.2f}, Y={Y:.2f}, Z={Z:.2f} m"

            # 4. 繪製文字到影像上
            cv2.putText(
                annotated_frame, 
                text_3d, 
                (int(x1), int(y1) - 40),  # 可以放在深度資訊上方或下方
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 255), 
                2
            )

        # 顯示結果
        cv2.imshow("RealSense YOLO Detection", annotated_frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 釋放資源
    pipeline.stop()
    cv2.destroyAllWindows()

