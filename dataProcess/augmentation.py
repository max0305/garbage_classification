import cv2
import albumentations as A
import os
import numpy as np
import matplotlib.pyplot as plt


# 数据集路径
images_dir = r"C:\Users\a0938\Desktop\training data\carton\train\images"  # 替换为图像文件夹路径
labels_dir = r"C:\Users\a0938\Desktop\training data\carton\train\labels"  # 替换为标签文件夹路径
output_images_dir = r"C:\Users\a0938\Desktop\training data\carton\train\aug_img"
output_labels_dir = r"C:\Users\a0938\Desktop\training data\carton\train\aug_lab"

# 确保输出目录存在
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# 定义增强操作
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 绘制增强后的图像和边界框
def plot_image_with_boxes(image, bboxes, classes):
    for bbox, cls in zip(bboxes, classes):
        x_center, y_center, width, height = bbox
        h, w, _ = image.shape
        # 将 YOLO 格式转换为像素坐标
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)
        # 绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, str(cls), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 使用 matplotlib 显示增强后的图像
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# 设置展示增强图像的数量
visualized_count = 0
max_visualized = 6  # 展示三张增强后的图像

# 每张原始图像生成的增强图像数量
augmentations_per_image = 5

# 遍历所有图片
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 图像路径
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

        # 读取图像
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # 读取标签
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    cls, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                    bboxes.append([x_center, y_center, bbox_width, bbox_height])
                    class_labels.append(int(cls))

        # 多次增强
        for i in range(augmentations_per_image):
            # 应用增强
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_classes = transformed['class_labels']

            # 增强后的文件名
            augmented_filename = f"aug_{i}_{filename}"
            output_image_path = os.path.join(output_images_dir, augmented_filename)
            cv2.imwrite(output_image_path, transformed_image)

            output_label_path = os.path.join(output_labels_dir, augmented_filename.replace(".jpg", ".txt").replace(".png", ".txt"))
            with open(output_label_path, "w") as f:
                for cls, bbox in zip(transformed_classes, transformed_bboxes):
                    f.write(f"{cls} {' '.join(map(str, bbox))}\n")

            # 仅展示三张增强后的图像
            if visualized_count < max_visualized:
                plot_image_with_boxes(transformed_image.copy(), transformed_bboxes, transformed_classes)
                visualized_count += 1

print("finished")