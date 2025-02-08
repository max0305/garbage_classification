import pandas as pd
import matplotlib.pyplot as plt

# 讀取訓練結果數據
results_path = 'runs/detect/train6/results.csv'  # 修改為你的路徑
data = pd.read_csv(results_path)

# 繪製損失曲線
plt.figure(figsize=(10, 6))
plt.plot(data['epoch'], data['train/box_loss'], label='Train Box Loss')
plt.plot(data['epoch'], data['val/box_loss'], label='Validation Box Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
#plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data['epoch'], data['train/cls_loss'], label='Train Classification Loss')
plt.plot(data['epoch'], data['val/cls_loss'], label='Validation Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# 繪製準確率曲線
plt.figure(figsize=(10, 6))
plt.plot(data['epoch'], data['metrics/precision(B)'], label='Precision')
plt.plot(data['epoch'], data['metrics/recall(B)'], label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Precision and Recall')
plt.legend()
plt.show()
