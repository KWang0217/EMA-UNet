import numpy as np
import cv2
import os

# root = './data/isic2017/'
# data_f = ['train/images/', 'val/images/']
# mask_f = ['train/masks/', 'val/masks/']
# set_size = [1500, 650]
# save_name = ['train', 'val']

root = './data/isic2018/'
data_f = ['train/images/', 'val/images/']
mask_f = ['train/masks/', 'val/masks/']
set_size = [1886, 808]
save_name = ['train', 'val']

height = 256
width = 256

for j in range(2):
    print('processing ' + data_f[j] + '......')
    count = 0
    length = set_size[j]
    imgs = np.uint8(np.zeros([length, height, width, 3]))
    masks = np.uint8(np.zeros([length, height, width, 1]))

    path = root + data_f[j]
    mask_p = root + mask_f[j]

    for i in os.listdir(path):
        if len(i.split('_')) == 2:
            img = cv2.imread(path + i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (width, height))

            m_path = mask_p + i.replace('.jpg', '_segmentation.png')
            mask = cv2.imread(m_path, 0)
            mask = cv2.resize(mask, (width, height))
            mask = np.expand_dims(mask, axis=-1)

            imgs[count] = img
            masks[count] = mask

            count += 1
            print(count)

    save_path = os.path.join("../", root)
    os.makedirs(save_path, exist_ok=True)

    np.save('{}/data_{}.npy'.format(save_path, save_name[j]), imgs)
    np.save('{}/mask_{}.npy'.format(save_path, save_name[j]), masks)
