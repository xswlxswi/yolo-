import os
import random

# 1. 配置参数
dataset_path = 'E:\Desktop\DL\yolov5-master\datasets\data\obj'  # 数据集图片所在路径
valid_percentage = 15  # 验证集比例 (%)
train_txt_path = 'data/train.txt'
valid_txt_path = 'data/valid.txt'

# 2. 确保输出目录存在
if not os.path.exists('data'):
    os.makedirs('data')

# 3. 获取所有图像文件列表
# 检查目录是否存在
if not os.path.exists(dataset_path):
    print(f"错误：找不到目录 {dataset_path}，请检查路径是否正确。")
else:
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_paths = [os.path.abspath(os.path.join(dataset_path, f)) for f in image_files]

    # 4. 打乱文件列表
    random.shuffle(image_paths)

    # 5. 计算分割点
    num_images = len(image_paths)
    num_valid = int(num_images * valid_percentage / 100)
    num_train = num_images - num_valid

    # 6. 分割列表
    train_paths = image_paths[:num_train]
    valid_paths = image_paths[num_train:]

    # 7. 写入 train.txt
    with open(train_txt_path, 'w') as f_train:
        for path in train_paths:
            f_train.write(path + '\n')
    print(f"成功生成 {train_txt_path}，包含 {len(train_paths)} 张训练图。")

    # 8. 写入 valid.txt
    with open(valid_txt_path, 'w') as f_valid:
        for path in valid_paths:
            f_valid.write(path + '\n')
    print(f"成功生成 {valid_txt_path}，包含 {len(valid_paths)} 张验证图。")