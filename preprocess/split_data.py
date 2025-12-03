import os
import shutil
from sklearn.model_selection import train_test_split

# Dataset paths
dataset_dir = "isic2018/"  # Replace with your dataset path
images_dir = os.path.join(dataset_dir, "images")  # Directory for images
masks_dir = os.path.join(dataset_dir, "masks")  # Directory for masks

# Output paths
output_dir = "data/" + dataset_dir  # Replace with your desired output path
train_images_dir = os.path.join(output_dir, "train", "images")
train_masks_dir = os.path.join(output_dir, "train", "masks")
val_images_dir = os.path.join(output_dir, "val", "images")
val_masks_dir = os.path.join(output_dir, "val", "masks")

# Create output directories
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)

# 1. Retrieve all image paths and corresponding mask paths
image_paths = []  # List to store image paths
mask_paths = []  # List to store mask paths

# Iterate through the images directory
for image_name in os.listdir(images_dir):
    if image_name.endswith(".jpg"):  # Ensure it is an image file
        image_path = os.path.join(images_dir, image_name)
        # Generate mask filename based on image filename
        mask_name = image_name.replace(".jpg", "_segmentation.png")
        mask_path = os.path.join(masks_dir, mask_name)

        # Check if the corresponding mask file exists
        if os.path.exists(mask_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)
        else:
            print(f"Warning: Mask file {mask_path} not found, skipping this image.")

# 2. Check data statistics
print("Total images:", len(image_paths))
print("Total masks:", len(mask_paths))

# 3. Split into training and validation sets
# Note: Hardcoded sizes 1886/808 based on ISIC2018 partition
train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, train_size=1886, test_size=808, random_state=42
)


# 4. Copy files to target directories
def copy_files(file_pairs, target_image_dir, target_mask_dir):
    for image_path, mask_path in file_pairs:
        # Copy image
        shutil.copy(image_path, target_image_dir)
        # Copy mask
        shutil.copy(mask_path, target_mask_dir)


# Copy training set
copy_files(zip(train_images, train_masks), train_images_dir, train_masks_dir)
# Copy validation set
copy_files(zip(val_images, val_masks), val_images_dir, val_masks_dir)

print("\nFile copy completed!")
print(f"Training set size: {len(train_images)}")
print(f"Validation set size: {len(val_images)}")
