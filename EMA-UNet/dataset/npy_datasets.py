import numpy as np
import albumentations as A
import torch.utils.data as data
import torchvision.transforms as transforms


class SkinDataset(data.Dataset):
    def __init__(self, image_root, gt_root, train):

        self.train = train
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if train:
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index]

        if self.train:
            transformed = self.transform(image=image, mask=gt)
            image = transformed['image']
            gt = transformed['mask']

        return self.img_transform(image), self.gt_transform(gt)

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batch_size, train=None, shuffle=None, num_workers=None, pin_memory=None):
    dataset = SkinDataset(image_root, gt_root, train)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory, )
    return data_loader
