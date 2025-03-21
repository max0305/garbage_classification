import rclpy
from rclpy.node import Node

import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import time
from tm12_amm_interfaces.srv import AiAction

CLASS_NAMES = [
    "Plastic Bottle",
    "Can",
    "tissue",
    "carton"
]

# ========== 複製自 kalmanTest.py ==========
def init_realsense_yolo():
    """
    原本 kalmanTest.py 的 init_realsense_yolo 函式。
    回傳 pipeline, align, model
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    model = YOLO("/home/jun/Downloads/best.pt") 
    return pipeline, align, model


#def load_camera_calibration():
    """
    原本 kalmanTest.py 的 load_camera_calibration 函式。
    """
    fs = cv2.FileStorage("/home/jun/Downloads/camera_calibration_result.yaml", cv2.FILE_STORAGE_READ)
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

    return color_frame, depth_frame

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



class DetectionNode(Node):
    def __init__(self):
        """
        這個建構子負責將您在 kalmanTest.py 裡「最先做」的動作都整合進來，
        包含 RealSense + YOLO 初始化、追蹤參數設定、手眼標定參數讀取等。
        """
        super().__init__('detection_node')  # Node 名稱 = detection
        self.get_logger().info('Detection node started')

        # 建立 AiAction 的 Service Client
        self.cli = self.create_client(AiAction, 'ai_action')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for AiAction service...')


        # 1) 初始化 YOLO 與 RealSense (原 kalmanTest.py 裏面 main() 的前半段)
        self.pipeline, self.align, self.model = init_realsense_yolo()
        self.get_logger().info("RealSense pipeline & YOLO model initialized.")

        # 2) 建立 OpenCV 顯示視窗 (若您在無頭環境則可忽略，或自行關閉)
        cv2.namedWindow("RealSense YOLO Detection", cv2.WINDOW_NORMAL)

        # 3) 建立追蹤器相關參數 (依照原程式)
        self.trackers = []           # 用來裝 KalmanFilterBBox 物件
        self.dt = 1/30               # frame rate
        self.MAX_LOST = 10
        self.IOU_THRESHOLD = 0.3

        # 4) 讀取手眼標定參數
        #self.handEyeRotation, self.handEyeTranslation = load_camera_calibration()
        #self.get_logger().info("Hand-eye calibration loaded.")

        # 5) 初始化一些控制變數
        self.last_distance = None
        self.start_time = time.time()

        # 6) (可選) 若要每幀讀取深度內參，可先讀一次
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        if depth_frame:
            self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            self.get_logger().info("Depth intrinsics captured.")
        else:
            self.depth_intrinsics = None
            self.get_logger().warn("No depth frame available at init; intrinsics set to None.")

        # 7) 設置 Timer，取代 while True
        #    每 0.033 秒 (~30fps) 自動呼叫 self.timer_callback()
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("DetectionNode __init__() complete. Start detection loop.")

    def timer_callback(self):
        """
        取代原 kalmanTest.py 裏的 while True。每次計時器觸發時執行一次偵測 + 卡曼多物件追蹤。
        """
        # 1) 時間判斷：若超過一定秒數，或長時間都偵測不到目標，就結束節點
        elapsed_time = time.time() - self.start_time
        # 若超過3秒仍沒追到任何東西，可視需求直接結束
        if elapsed_time > 3 and len(self.trackers) == 0:
            self.get_logger().info("未穩定偵測到任何目標，程式結束。")
            self.destroy_node()
            return
        # 若超過10秒就自動結束
        if elapsed_time > 10:
            self.get_logger().info("偵測超過10秒，程式結束。")
            cv2.destroyAllWindows()
            self.print_final_results()
            self.start_time = time.time()
            self.trackers = [] 
            #self.destroy_node()
            return

        # 2) 取得對齊之 color 與 depth frame
        # depth_intrinsics 中就包含 fx, fy, ppx, ppy, distortion 等資訊
        color_frame, depth_frame = get_realsense_frames(self.pipeline, self.align)
        if not color_frame or not depth_frame:
            return

        frame = np.asanyarray(color_frame.get_data())

        # 3) YOLO 推論
        results = self.model(frame)
        det = results[0]
        boxes = det.boxes  # YOLO 會將預測結果放在 det.boxes

        # 4) 先對現有追蹤器做 predict()
        for kf in self.trackers:
            kf.predict()  # 卡曼預測
            # 若本迴圈沒匹配到量測，後續會 lost_frames += 1

        # 5) 取得新的偵測框 (det_bboxes) 與對應的類別/信心
        det_bboxes = []
        class_ids = []
        confs = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1
            det_bboxes.append((int(x), int(y), int(w), int(h)))

            cls_id = int(box.cls[0])
            class_ids.append(cls_id)

            conf = float(box.conf[0])
            confs.append(conf)

        # 6) 進行追蹤資料關聯：IOU 配對
        matched_trackers = set()
        for i, dbbox in enumerate(det_bboxes):
            best_iou = 0.0
            best_tracker_idx = None
            for kf_idx, kf in enumerate(self.trackers):
                pred_bbox = kf.get_bbox()
                current_iou = iou(pred_bbox, dbbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_tracker_idx = kf_idx
            
            # 若大於 IOU 閾值 => 更新該 tracker
            if best_iou > self.IOU_THRESHOLD and best_tracker_idx is not None:
                self.trackers[best_tracker_idx].update(dbbox)
                self.trackers[best_tracker_idx].current_conf = confs[i]
                matched_trackers.add(best_tracker_idx)
            else:
                # 找不到合適 => 新增 tracker
                cls_id = class_ids[i]
                new_kf = KalmanFilterBBox(self.dt, class_id=cls_id)
                new_kf.init_state(dbbox)
                new_kf.current_conf = confs[i]
                self.trackers.append(new_kf)

        # 7) 沒被配對到的 tracker => lost_frames + 1
        for idx, kf in enumerate(self.trackers):
            if idx not in matched_trackers:
                kf.lost_frames += 1

        # 8) 移除失效追蹤器
        self.trackers = [kf for kf in self.trackers if kf.lost_frames <= self.MAX_LOST]

        # 9) 繪製結果
        annotated_frame = frame.copy()
        for idx, kf in enumerate(self.trackers):
            x, y, w, h = kf.get_bbox()
            x2, y2 = x + w, y + h

            # 畫 2D 方框
            cv2.rectangle(annotated_frame, (x, y), (x2, y2), (0, 255, 0), 2)

            # 顯示類別名稱
            if kf.class_id is not None and 0 <= kf.class_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[kf.class_id]
                cv2.putText(annotated_frame, class_name, (x + 2, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 取得該 tracker 中心像素
            center_x = x + w // 2
            center_y = y + h // 2

            # 取得深度
            distance = depth_frame.get_distance(center_x, center_y)
            if self.last_distance is None:
                self.last_distance = distance
            # 若本次距離 0 或跳動過大，可酌情平滑處理
            elif distance <= 0 or (abs(distance - self.last_distance) > 0.05 and self.last_distance != 0):
                distance = self.last_distance
            self.last_distance = distance

            # 反投影到 3D 座標
            point_3d = rs.rs2_deproject_pixel_to_point(
                self.depth_intrinsics, [center_x, center_y], distance
            )
            X, Y, Z = point_3d  # 單位：公尺
            #point_3d_arr = np.array(point_3d).reshape(3, 1)
            
            # 手眼座標轉換
            #point_arm = self.handEyeRotation @ point_3d_arr + self.handEyeTranslation
            #X, Y, Z = point_arm.ravel()

            # 在畫面上顯示 tracker ID 與 3D 座標
            text_id = f"ID={idx}"
            cv2.putText(annotated_frame, text_id, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            text_3d = f"3D=({X:.3f}, {Y:.3f}, {Z:.3f})"
            cv2.putText(annotated_frame, text_3d, (x, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 記錄 3D 歷史
            kf.add_3d_history((X, Y, Z), kf.current_conf)

            # 顯示 confidence
            text_conf = f"Conf={kf.current_conf:.2f}"
            cv2.putText(annotated_frame, text_conf, (x + 2, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("RealSense YOLO Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()

    def print_final_results(self): 
        # 假設您 node 內的追蹤器叫做 self.trackers
        # 同時您也有 self.model (YOLO 模型)，或有 self.model.names
        max_conf = 0
        target_index = None

        # 過濾掉 history 少於 10 筆的 tracker
        self.trackers = [kf for kf in self.trackers if len(kf.history) >= 10]

        for i, kf in enumerate(self.trackers):
            print(f"Tracker ID = {i}, class_ = {self.model.names[kf.class_id]}, history(len = {len(kf.history)}) =")

            for idx, (X, Y, Z, conf) in enumerate(kf.history):
                print(f"  Frame {idx} coordinate: ({X:.3f}, {Y:.3f}, {Z:.3f}), confidence: {conf:.2f}")

            (AVG_X, AVG_Y, AVG_Z, AVG_CONF) = average_3d_coordinates(kf.history)
            if AVG_X is not None:
                print(f"  平均座標: ({AVG_X:.3f}, {AVG_Y:.3f}, {AVG_Z:.3f}), 平均信心: {AVG_CONF:.2f}")
            else: 
                print("平均座標: None")

            if AVG_CONF is not None and AVG_CONF > max_conf: 
                max_conf = AVG_CONF
                target_index = i

        if target_index is not None and max_conf != 0:
            (TAR_X, TAR_Y, TAR_Z, TAR_CONF) = average_3d_coordinates(self.trackers[target_index].history)
            print(f"目標id: {target_index}, 平均信心: {TAR_CONF:.2f}")
            print(f"目標座標: ({TAR_X:.3f}, {TAR_Y:.3f}, {TAR_Z:.3f})")
            #呼叫ai_action node
            self.call_ai_action(
                self.trackers[target_index].class_id, 
                1, 
                TAR_X, 
                TAR_Y, 
                TAR_Z)
            
    def call_ai_action(self, class_id, repeat_times, x, y, z):
        request = AiAction.Request()
        request.class_id = class_id
        request.repeat_times = repeat_times
        request.x = x
        request.y = y
        request.z = z


        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f"AI 動作成功: {response.message}")
            else:
                self.get_logger().error(f"AI 動作失敗: {response.message}")
        else:
            self.get_logger().error("服務呼叫失敗")

    def destroy_node(self):
        """
        當 Node 被銷毀 (結束) 時，可以在此釋放資源 (pipeline.stop, cv2.destroyAllWindows)。
        """
        if hasattr(self, "destroyed") and self.destroyed:  # ✅ 若已經執行過，則跳過
            return
        self.get_logger().info("Shutting down DetectionNode...")
        self.destroyed = True

        # ✅ 停止 RealSense pipeline（避免 stop() 在未啟動狀態執行）
        if hasattr(self, "pipeline") and self.pipeline is not None:
            try:
                self.pipeline.stop()
                self.get_logger().info("✅ RealSense pipeline stopped.")
            except RuntimeError as e:
                self.get_logger().error(f"⚠️ Failed to stop pipeline: {e}")

        # ✅ 釋放 OpenCV 視窗
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # 確保 OpenCV 正確關閉
        self.get_logger().info("✅ OpenCV windows closed.")

        super().destroy_node()
        self.get_logger().info("✅ DetectionNode terminated successfully.")



def main(args=None):
    """
    這是 ROS2 Python 套件的執行入口點，對應 setup.py 裡 entry_points['console_scripts']。
    """
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)  # ROS2 事件迴圈，會自動呼叫 node.timer_callback()
    except KeyboardInterrupt:
        node.get_logger().info("User interrupted. Shutting down.")
    finally:
        if hasattr(node, "destroyed") and not node.destroyed:  # ✅ 只銷毀一次
            node.destroy_node()
        if rclpy.ok():  # ✅ 確保 `shutdown()` 只執行一次
            rclpy.shutdown()
        print("✅ ROS2 node fully shut down.")

if __name__ == "__main__":
    main()
