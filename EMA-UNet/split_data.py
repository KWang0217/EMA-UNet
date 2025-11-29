import os
import shutil
from sklearn.model_selection import train_test_split

# 数据集路径
dataset_dir = "isic2018/"  # 替换为你的数据集路径
images_dir = os.path.join(dataset_dir, "images")  # 图片目录
masks_dir = os.path.join(dataset_dir, "masks")  # 掩码目录

# 输出路径
output_dir = "data/" + dataset_dir  # 替换为你的输出路径
train_images_dir = os.path.join(output_dir, "train", "images")
train_masks_dir = os.path.join(output_dir, "train", "masks")
val_images_dir = os.path.join(output_dir, "val", "images")
val_masks_dir = os.path.join(output_dir, "val", "masks")

# 创建输出目录
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)

# 1. 获取所有图片路径和对应的掩码路径
image_paths = []  # 存储图片路径
mask_paths = []  # 存储掩码路径

# 遍历 images 目录
for image_name in os.listdir(images_dir):
    if image_name.endswith(".jpg"):  # 确保是图片文件
        image_path = os.path.join(images_dir, image_name)
        mask_name = image_name.replace(".jpg", "_segmentation.png")  # 根据图片名生成掩码文件名
        mask_path = os.path.join(masks_dir, mask_name)

        # 检查掩码文件是否存在
        if os.path.exists(mask_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)
        else:
            print(f"警告: 掩码文件 {mask_path} 不存在，跳过该图片。")

# 2. 检查数据
print("总图片数量:", len(image_paths))
print("总掩码数量:", len(mask_paths))

# 3. 划分训练集和验证集
train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, train_size=1886, test_size=808, random_state=42
)


# 4. 拷贝文件到目标目录
def copy_files(file_pairs, target_image_dir, target_mask_dir):
    for image_path, mask_path in file_pairs:
        # 拷贝图片
        shutil.copy(image_path, target_image_dir)
        # 拷贝掩码
        shutil.copy(mask_path, target_mask_dir)


# 拷贝训练集
copy_files(zip(train_images, train_masks), train_images_dir, train_masks_dir)
# 拷贝验证集
copy_files(zip(val_images, val_masks), val_images_dir, val_masks_dir)

print("\n文件拷贝完成！")
print(f"训练集图片数量: {len(train_images)}")
print(f"验证集图片数量: {len(val_images)}")
