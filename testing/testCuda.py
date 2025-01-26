import torch
from torchvision.ops import nms
import torchvision
print(torch.__version__)
print(torchvision.__version__)
# 測試樣本
boxes = torch.tensor([[100, 100, 210, 210], [105, 105, 215, 215], [100, 100, 200, 200]], dtype=torch.float32).cuda()
scores = torch.tensor([0.8, 0.75, 0.9]).cuda()
iou_threshold = 0.5

try:
    selected_indices = nms(boxes, scores, iou_threshold)
    print("NMS function executed successfully on GPU.")
except Exception as e:
    print("NMS function failed:", e)
