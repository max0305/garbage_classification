import pybullet as p
import pybullet_data
import time

# 1. 連線到 PyBullet (GUI 模式)
physicsClient = p.connect(p.GUI)

# 2. 設定搜尋路徑，可以使用 pybullet_data 裏頭提供的範例
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. 載入地面
planeId = p.loadURDF("plane.urdf")

# 4. 設定重力
p.setGravity(0, 0, -9.8)

# 5. 載入機器手臂 (此處示範 UR5，需確保有對應的 URDF)
#    也可以自行下載其他手臂模型 URDF 並指定檔案路徑
ur5Id = p.loadURDF("tmr_ros2/tm_description/urdf/tm12-nominal.urdf", [0, 0, 0], useFixedBase=True)

# 6. 進行模擬迴圈
while True:
    # 每一次迭代步進模擬
    p.stepSimulation()
    time.sleep(1./240.)  # 控制模擬速度，PyBullet 預設約 240Hz