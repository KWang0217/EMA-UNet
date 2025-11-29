from torch.utils.data import DataLoader
from models.emaunet import EMAUNet
from dataset.npy_datasets import get_loader
from engine import *
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config


def main(datasets, checkpoint_path):
    config = setting_config
    model_cfg = config.model_config
    model = EMAUNet(num_classes=model_cfg['num_classes'],
                     input_channels=model_cfg['input_channels'],
                     c_list=model_cfg['c_list'],
                     bridge=model_cfg['bridge'],
                     deep_supervision=model_cfg['deep_supervision'])

    # ========== 加载模型权重 ==========
    print(f"=> loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    print("=> loaded checkpoint successfully!")
    model = model.cuda().eval()

    # 加载数据
    if datasets == 'isic18':
        data_path = '../data/isic2018/'
    elif datasets == 'isic17':
        data_path = '../data/isic2017/'
    else:
        raise Exception('datasets in not right!')

    val_image_root = '{}/data_val.npy'.format(data_path)
    val_gt_root = '{}/mask_val.npy'.format(data_path)
    val_loader = get_loader(val_image_root,
                            val_gt_root,
                            train=False,
                            shuffle=False,
                            pin_memory=True,
                            batch_size=1,
                            num_workers=0)

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            img, _ = data
            img = img.cuda(non_blocking=True).float()
            _, output = model(img)

            output = (output >= 0.5).float()
            output_np = output.squeeze().cpu().numpy()
            output_np = (output_np * 255).astype(np.uint8)

            out_dir = './predict_masks/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(out_dir + f"LEMAUNet_{i:03d}.png".format(i), output_np)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    datasets = 'isic18'
    checkpoint_path = './results/...'
    main(datasets, checkpoint_path)
