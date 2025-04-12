import os
import shutil
import random

# 设置随机种子确保结果可复现（每次划分一致）
random.seed(42)

# ========================== 路径配置 ==========================
# 原始数据集目录，格式应为：每个类别一个子文件夹
data_dir = "./Dataset"

# 划分后的数据存放目录，会被创建成：./data/train、./data/val、./data/test
output_dir = "./data"

# ========================== 数据划分比例 ==========================
train_ratio = 0.8  # 训练集占比
val_ratio = 0.1  # 验证集占比
test_ratio = 0.1  # 测试集占比

# ========================== 创建输出文件夹结构 ==========================
# 遍历每个划分（train/val/test）和每个类别，创建对应的子文件夹
for split in ["train", "val", "test"]:
    for class_name in os.listdir(data_dir):
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# ========================== 数据划分逻辑 ==========================
# 遍历原始数据集中每个类别文件夹
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)

    # 如果不是文件夹则跳过（防止误读到非类别目录）
    if not os.path.isdir(class_dir):
        continue

    # 获取该类别下的所有图片文件名
    files = os.listdir(class_dir)
    random.shuffle(files)  # 打乱顺序，保证随机划分

    # 计算各个数据集的结束索引
    train_end = int(len(files) * train_ratio)
    val_end = train_end + int(len(files) * val_ratio)

    # 分别划分为训练、验证、测试集
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # =================== 拷贝文件到目标文件夹 ===================
    for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        for file_name in split_files:
            src_path = os.path.join(class_dir, file_name)  # 原始文件路径
            dst_path = os.path.join(output_dir, split, class_name, file_name)  # 目标路径
            shutil.copy(src_path, dst_path)  # 复制文件

print("数据集划分完成！")
