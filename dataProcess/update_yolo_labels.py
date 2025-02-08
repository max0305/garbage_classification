import os

def update_yolo_labels(folder_path, old_index=0, new_index=1):
    """
    更新 YOLO 格式标签文件，将所有标签的索引从 old_index 改为 new_index。
    
    :param folder_path: 标签文件所在的文件夹路径
    :param old_index: 需要替换的旧索引值，默认为 0
    :param new_index: 替换后的新索引值，默认为 1
    """
    # 确保输入的索引是字符串格式，便于后续比较和替换
    old_index = str(old_index)
    new_index = str(new_index)
    
    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 仅处理以 .txt 结尾的文件
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            # 打开文件读取所有行
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # 处理每一行，替换行首的索引
            updated_lines = []
            for line in lines:
                parts = line.split()
                if parts and parts[0] == old_index:
                    parts[0] = new_index
                updated_lines.append(' '.join(parts) + '\n')
            
            # 将修改后的内容写回文件
            with open(file_path, 'w') as file:
                file.writelines(updated_lines)
            print(f"已更新文件: {filename}")

# 使用示例
folder_path = r"C:\garbage_classification\valid\labels\can"  # 替换为实际的文件夹路径
update_yolo_labels(folder_path)
