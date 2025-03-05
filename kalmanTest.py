import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import time

CLASS_NAMES = [
    "Plastic Bottle",     # 寶特瓶
    "Can",     # 鋁罐
    "tissue",    # 衛生紙
    "carton"    # 紙盒
]

# ========== 初始化 YOLO 與 RealSense ==========
def init_realsense_yolo():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # 深度影像對齊 RGB 影像
    align = rs.align(rs.stream.color)
    # YOLO權重
    model = YOLO("runs/detect/train7/weights/best.pt")

    return pipeline, align, model

# ========== 讀取相機校正參數 ==========
def load_camera_calibration():
    fs = cv2.FileStorage("camera_calibration_result.yaml", cv2.FILE_STORAGE_READ)
    handEyeRotation = fs.getNode("handEyeRotation").mat()
    handEyeTranslation = fs.getNode("handEyeTranslation").mat()
    return handEyeRotation, handEyeTranslation

# ========== 獲取RealSense影像與處理 ==========
def get_realsense_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None, None

    depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    return color_frame, depth_frame, depth_intrinsics

# ========== (A) 簡易的 2D Bounding Box 卡曼濾波器 ==========
class KalmanFilterBBox:
    def __init__(self, dt=1.0, class_id=None):
        # 狀態向量: [x, y, w, h, vx, vy, vw, vh]
        self.dt = dt
        self.dim_state = 8

        # 追蹤器對應的類別，直接存 class_id，或可存 class_name
        self.class_id = class_id

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

        # 保存最近 10 幀的 3D 座標紀錄 
        self.history = []

        # 原先只存 (X, Y, Z)，現在加入 confidence，
        # 故改用 (X, Y, Z, conf) 四元組。
        self.current_conf = 0.0  # 用於記錄本幀偵測信心

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
    
    def add_3d_history(self, xyz, conf):
        """
        xyz: (X, Y, Z) 3D 座標
        conf: float, 本次追蹤/偵測的信心度
        僅保留最近 10 筆歷史
        """
        self.history.append((xyz[0], xyz[1], xyz[2], conf))
        if len(self.history) > 10:
            self.history.pop(0)

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

# ========== (c) 3D座標平均函數 ==========
def average_3d_coordinates(history):
    if not history:  # 若 history 為空
        return None  

    history_array = np.array(history)  # 轉為 NumPy 陣列，形狀為 (N, 4)
    mean_xyz = np.mean(history_array, axis=0)  # 計算每個維度的平均
    return tuple(mean_xyz)  # 轉回 tuple 較易讀

# ========== (C) 主程式：結合 runRealsense.py + 多物件追蹤 ==========
def main():
    # YOLO + RealSense 初始化
    # 初始化 YOLO 與 RealSense
    pipeline, align, model = init_realsense_yolo()

    cv2.namedWindow("RealSense YOLO Detection", cv2.WINDOW_NORMAL)

    # 建立追蹤器列表
    trackers = []

    # 自訂參數
    dt = 1/30
    MAX_LOST = 10  # 容許多少幀未匹配就刪除追蹤器
    IOU_THRESHOLD = 0.3

    # 讀取座標轉換參數
    handEyeRotation, handEyeTranslation = load_camera_calibration()

    # 記錄開始時間，用來計算 3 秒
    start_time = time.time()

    last_distance = None  # frame未初始化
    try:
        while True:
            # 取得對齊後的影像與深度
            # depth_intrinsics 中就包含 fx, fy, ppx, ppy, distortion 等資訊
            color_frame, depth_frame, depth_intrinsics = get_realsense_frames(pipeline, align)
            if not color_frame or not depth_frame:
                continue

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
            # 紀錄偵測物件id
            class_ids = []
            # 新增 confs 用來保存對應的信心度
            confs = []  
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                w = x2 - x1
                h = y2 - y1
                x = x1
                y = y1
                # 這裡使用左上 (x, y, w, h) 作為卡曼濾波器量測
                # 也可改成中心 (cx, cy, w, h) 方式
                det_bboxes.append((int(x), int(y), int(w), int(h)))

                # 取得對應的 class id（即索引）
                cls_id = int(box.cls[0])
                class_ids.append(cls_id)

                 # 取得及紀錄 YOLO 的 confidence 
                conf = float(box.conf[0])
                confs.append(conf)

            # 3) 簡易的資料關聯: 對每個偵測框找最適合的 tracker
            matched_trackers = set()  # 用來標記哪些 tracker 已被匹配
            for i, dbbox in enumerate(det_bboxes):
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
                    # 把此偵測框的 confidence 記錄到該追蹤器
                    trackers[best_tracker].current_conf = confs[i]
                    matched_trackers.add(best_tracker)
                else:
                    # 找不到合適的 => 新增一個新的追蹤器
                    cls_id = class_ids[i]       # 對應到當前的物件類別
                    new_kf = KalmanFilterBBox(dt,  class_id=cls_id)
                    new_kf.init_state(dbbox)
                    new_kf.current_conf = confs[i]
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

                # 顯示該 tracker 之類別 (若為 None 表示沒偵測到就不顯示)
                if kf.class_id is not None:
                    class_name = CLASS_NAMES[kf.class_id]   # 從 CLASS_NAMES 表拿類別字串
                    '''
                    正常應從model.names中拿字串，但cv2無法顯示中文字，故另設計英文對照表 CLASS_NAMES 替代
                    # class_name = model.names[kf.class_id]  # 從模型 names 表拿字串
                    '''
                    # 文字位置可自行調整
                    cv2.putText(annotated_frame, class_name, (x + 2, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # === (A) 加入 3D 座標計算 ===
                # 1) 取出 Bounding Box 中心，通常 (cx, cy) = (x + w/2, y + h/2)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 2) 由 depth_frame 取得該中心像素的深度 (公尺)
                distance = depth_frame.get_distance(center_x, center_y)
                if last_distance is None:
                    # 第一幀，直接用當前值初始化
                    last_distance = distance
                elif distance <= 0 or (abs(distance - last_distance) > 0.05 and last_distance != 0):
                    distance = last_distance

                last_distance = distance

                
                # 3) 使用 RealSense 的函式，將像素座標 + 深度轉成 3D 座標 (X, Y, Z)
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics,
                    [center_x, center_y],
                    distance
                )
                # 輸出的座標依 RealSense 機型通常是 (X:右正, Y:下正, Z:前正)

                point_3d_arr = np.array(point_3d).reshape(3, 1)  # shape = (3,1)
                
                # 座標轉換
                point_arm = handEyeRotation @ point_3d_arr + handEyeTranslation

                # 解包成純量
                X, Y, Z = point_arm.ravel()  # ravel() 會把 (3,1) 攤平成 (3,)

                # === (B) 顯示追蹤ID 與 3D座標文字 ===
                # 這裡將 ID 顯示於框的上方
                text_id = f"ID={idx}"
                cv2.putText(annotated_frame, text_id, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 顯示 3D 座標資訊，可放在框內或旁邊
                text_3d = f"3D=({X:.3f}, {Y:.3f}, {Z:.3f})m"
                cv2.putText(annotated_frame, text_3d, (x, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 把該追蹤器當前 3D 座標加到 history
                kf.add_3d_history((X, Y, Z), kf.current_conf)

                # 顯示信心度(Optional)
                text_conf = f"Conf={kf.current_conf:.2f}"
                cv2.putText(annotated_frame, text_conf, (x + 2, y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("RealSense YOLO Detection", annotated_frame)

            # 檢查是否超過 3 秒
            elapsed_time = time.time() - start_time
            if elapsed_time > 10:
                break
            # 手動退出，press q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # 程式結束後，可將每個 KF 的 history 輸出
    for i, kf in enumerate(trackers):
        print(f"Tracker ID = {i}, class_ = {model.names[kf.class_id]}, history(len = {len(kf.history)}) =")
        for idx, (X, Y, Z, conf) in enumerate(kf.history):
            print(f"  Frame {idx} coordinate: ({X:.3f}, {Y:.3f}, {Z:.3f}), confidence: {conf:.2f}")
        (AVG_X, AVG_Y, AVG_Z, AVG_CONF) = average_3d_coordinates(kf.history)
        if AVG_X is not None:
            print(f"平均座標: ({AVG_X:.3f}, {AVG_Y:.3f}, {AVG_Z:.3f}), 平均信心: {AVG_CONF:.3f}")
        else: 
            print("平均座標: None")
        

if __name__ == "__main__":
    main()
