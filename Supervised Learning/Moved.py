import os
import pandas as pd
import shutil

csv_file = 'COVID_Data_filter.csv'  # CSV文件路径
image_dir = 'images'  # 图像存放路径
output_dir_0 = '0'  # 标签为0的目标路径
output_dir_1 = '1'  # 标签为1的目标路径

# 创建目标目录（如果不存在）
os.makedirs(output_dir_0, exist_ok=True)
os.makedirs(output_dir_1, exist_ok=True)

# 读取CSV文件
df = pd.read_csv(csv_file)

# 遍历CSV文件中的每一行
for index, row in df.iterrows():
    image_name = row[0]  # 图像名称
    label = row[1]  # 标签

    # 定义图像的完整路径
    src_image_path = os.path.join(image_dir, image_name)

    # 根据标签定义目标路径
    if label == 0:
        dst_image_path = os.path.join(output_dir_0, image_name)
    else:
        dst_image_path = os.path.join(output_dir_1, image_name)

    # 移动图像到目标路径
    if os.path.exists(src_image_path):
        shutil.move(src_image_path, dst_image_path)
        print(f"Moved {image_name} to {dst_image_path}")
    else:
        print(f"Image {image_name} not found in {image_dir}")

print("All images have been moved based on their labels.")
