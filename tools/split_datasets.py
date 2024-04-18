import os
import random
import shutil


dataset_dir = 'datasets'
train_dir = 'datasets/train'
test_dir = 'datasets/test'
class_name = ['cat', 'dog']
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 划分比例
train_ratio = 0.8

# 遍历数据集
for category in class_name:
    category_dir = os.path.join(dataset_dir, category)
    images = os.listdir(category_dir)
    random.shuffle(images)  # 随机打乱文件列表

    # 计算划分位置
    train_size = int(len(images) * train_ratio)

    # 划分训练集和测试集
    train_images = images[:train_size]
    test_images = images[train_size:]

    # 创建类别目录
    train_category_dir = os.path.join(train_dir, category)
    test_category_dir = os.path.join(test_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(test_category_dir, exist_ok=True)

    # 移动训练集图片
    for image in train_images:
        src_path = os.path.join(category_dir, image)
        dest_path = os.path.join(train_category_dir, image)
        shutil.move(src_path, dest_path)

    # 移动测试集图片
    for image in test_images:
        src_path = os.path.join(category_dir, image)
        dest_path = os.path.join(test_category_dir, image)
        shutil.move(src_path, dest_path)

print("数据集划分完成！")