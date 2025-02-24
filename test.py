import cv2
import numpy as np

# 1. 讀取檔案
fs = cv2.FileStorage("camera_calibration_result.yaml", cv2.FILE_STORAGE_READ)

# 2. 讀取 cameraMatrix
cameraMatrix_node = fs.getNode("cameraMatrix")
cameraMatrix = cameraMatrix_node.mat()

# 3. 讀取 distCoeffs
distCoeffs_node = fs.getNode("distCoeffs")
distCoeffs = distCoeffs_node.mat()

# 4. 讀取 handEyeRotation
handEyeRotation_node = fs.getNode("handEyeRotation")
handEyeRotation = handEyeRotation_node.mat()

# 5. 讀取 handEyeTranslation
handEyeTranslation_node = fs.getNode("handEyeTranslation")
handEyeTranslation = handEyeTranslation_node.mat()

fs.release()  # 關閉 FileStorage

# 檢查讀取結果
print("cameraMatrix:")
print(cameraMatrix)

print("distCoeffs:")
print(distCoeffs)

print("handEyeRotation:")
print(handEyeRotation)

print("handEyeTranslation:")
print(handEyeTranslation)
