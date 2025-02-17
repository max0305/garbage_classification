import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

# ========== (A) 簡易的 2D Bounding Box 卡曼濾波器 ==========
class KalmanFilterBBox:
    def __init__(self, dt=1.0):
        # 狀態向量: [x, y, w, h, vx, vy, vw, vh]
        self.dt = dt
        self.dim_state = 8

        # 狀態轉移矩陣 F (8x8)
        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1]
        ], dtype=np.float32)

        # 量測矩陣 H (4x8)，只量測 [x, y, w, h]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        # 狀態向量 X (8x1)
        self.X = np.zeros((8,1), dtype=np.float32)

        # 預估誤差協方差 P (8x8)
        self.P = np.eye(8, dtype=np.float32)

        # 過程雜訊協方差 Q
        self.Q = np.eye(8, dtype=np.float32) * 0.01

        # 量測雜訊協方差 R (4x4)
        self.R = np.eye(4, dtype=np.float32) * 0.1

        # 追蹤品質計數器 (可用來判斷追蹤器是否可信，或失效等)
        self.lost_frames = 0

    def init_state(self, bbox):
        # bbox: (x, y, w, h)
        self.X[0] = bbox[0]
        self.X[1] = bbox[1]
        self.X[2] = bbox[2]
        self.X[3] = bbox[3]
        # 速度初始化暫設 0
        self.X[4] = 0
        self.X[5] = 0
        self.X[6] = 0
        self.X[7] = 0

    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X

    def update(self, meas):
        # meas: (x, y, w, h)
        z = np.array(meas, dtype=np.float32).reshape((4,1))

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        y = z - (self.H @ self.X)
        self.X = self.X + K @ y

        I = np.eye(self.dim_state, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        # 每次更新到量測，可視為追蹤成功 => 重置 lost_frames
        self.lost_frames = 0

        return self.X

    def get_bbox(self):
        # 取出 [x, y, w, h] 來繪製框，並確保不為負
        x, y, w, h = self.X[0], self.X[1], self.X[2], self.X[3]
        w = max(w, 1)
        h = max(h, 1)
        return (int(x), int(y), int(w), int(h))

# ========== (B) 簡易 IOU 函數，做關聯用 ==========
def iou(bbox1, bbox2):
    # bbox: (x, y, w, h)
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 轉為 x1y1x2y2，(x1, y1)為左上角、(x2, y2)為右下角
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2

    #交集區域的左上角座標，對應於「兩個框的左上角中 x、y 最大的那個」
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    #交集區域的右下角座標，對應於「兩個框的右下角中 x、y 最小的那個」
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0
    return inter_area / union_area

# ========== (C) 主程式：結合 runRealsense.py + 多物件追蹤 ==========
def main():
    # YOLO + RealSense 初始化
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    model = YOLO("runs/detect/train7/weights/best.pt")  # 替換為自己的權重

    cv2.namedWindow("RealSense YOLO Detection", cv2.WINDOW_NORMAL)

    # 建立追蹤器列表
    trackers = []

    # 自訂參數
    dt = 1/30
    MAX_LOST = 10  # 容許多少幀未匹配就刪除追蹤器
    IOU_THRESHOLD = 0.3

    last_distance = None  # frame未初始化
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # depth_intrinsics 中就包含 fx, fy, ppx, ppy, distortion 等資訊 
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

            frame = np.asanyarray(color_frame.get_data())
            results = model(frame)
            det = results[0]
            boxes = det.boxes
            #annotated_frame = det.plot().copy()
            annotated_frame = frame.copy()

            # 1) 先對所有 tracker 做 predict
            for kf in trackers:
                kf.predict()
                # 若本迴圈沒更新到量測，lost_frames 會在下面++一次

            # 2) 蒐集所有新的偵測框
            det_bboxes = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                w = x2 - x1
                h = y2 - y1
                x = x1
                y = y1
                # 這裡使用左上 (x, y, w, h) 作為卡曼濾波器量測
                # 也可改成中心 (cx, cy, w, h) 方式
                det_bboxes.append((int(x), int(y), int(w), int(h)))

            # 3) 簡易的資料關聯: 對每個偵測框找最適合的 tracker
            matched_trackers = set()  # 用來標記哪些 tracker 已被匹配
            for dbbox in det_bboxes:
                best_iou = 0
                best_tracker = None
                for kf_idx, kf in enumerate(trackers):
                    # 計算與預測框的IOU
                    pred_bbox = kf.get_bbox()  # 拿到平滑後的預測框
                    current_iou = iou(pred_bbox, dbbox)
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_tracker = kf_idx

                # 若 IOU 大於閾值，視為同一物件 => update
                if best_iou > IOU_THRESHOLD and best_tracker is not None:
                    # 用量測更新
                    trackers[best_tracker].update(dbbox)
                    matched_trackers.add(best_tracker)
                else:
                    # 找不到合適的 => 新增一個新的追蹤器
                    new_kf = KalmanFilterBBox(dt)
                    new_kf.init_state(dbbox)
                    trackers.append(new_kf)

            # 4) 處理「沒有被匹配的 tracker」 => lost_frames 累加
            for idx, kf in enumerate(trackers):
                if idx not in matched_trackers:
                    kf.lost_frames += 1

            # 5) 移除 lost_frames 過多的 tracker
            trackers = [kf for kf in trackers if kf.lost_frames <= MAX_LOST]

            # 6) 繪製結果：用每個 tracker 的平滑後 bbox 在影像上畫框
            # 加上「追蹤ID」和「3D座標」(可參考 runRealsense.py 中的 depth_frame 與深度內參)
            for idx, kf in enumerate(trackers):  # 用 enumerate 取得追蹤器索引
                x, y, w, h = kf.get_bbox()
                x2, y2 = x + w, y + h

                # 在 2D 畫面上繪製方框
                cv2.rectangle(annotated_frame, (x, y), (x2, y2), (0, 255, 0), 2)
                
                # === (A) 加入 3D 座標計算 ===
                # 1) 取出 Bounding Box 中心，通常 (cx, cy) = (x + w/2, y + h/2)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 2) 由 depth_frame 取得該中心像素的深度 (公尺)
                distance = depth_frame.get_distance(center_x, center_y)
                if last_distance is None:
                    # 第一幀，直接用當前值初始化
                    last_distance = distance
                elif distance <= 0 or abs(distance - last_distance) > 0.05:
                    distance = last_distance
                    
                last_distance = distance

                
                # 3) 使用 RealSense 的函式，將像素座標 + 深度轉成 3D 座標 (X, Y, Z)
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics,
                    [center_x, center_y],
                    distance
                )
                # 輸出的座標依 RealSense 機型通常是 (X:右正, Y:下正, Z:前正)
                # 若你需要 Y:上正，可自行調整 Y = -Y 等
                
                X, Y, Z = point_3d
                
                # === (B) 顯示追蹤ID 與 3D座標文字 ===
                # 這裡將 ID 顯示於框的上方
                text_id = f"ID={idx}"
                cv2.putText(annotated_frame, text_id, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 顯示 3D 座標資訊，可放在框內或旁邊
                text_3d = f"3D=({X:.2f}, {Y:.2f}, {Z:.2f})m"
                cv2.putText(annotated_frame, text_3d, (x, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("RealSense YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
