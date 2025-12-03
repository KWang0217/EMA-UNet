# NLFER: EMA-UNet: Efficient Multi-Attention UNet for Skin Lesion Segmentation

This repository contains the official implementation of EMA-UNet, a multi-attention network for skin lesion segmentation. The overall architecture is illustrated in the figure below.
<div align="center">
  <img width="90%" alt="Fig 3" src="https://github.com/user-attachments/assets/110ead52-30a9-406c-8e86-0b1ba3c81c13" />
</div>

## Installation

Clone this repo and install dependencies.
```bash
git clone https://github.com/KWang0217/EMA-UNet
pip install -r requirements.txt
```

## Preparation

1.  **Environment Setup:**
    Create the conda environment using the provided file:
    ```bash
    pip install -r requirements.txt
    ```
    
2.  **Data Preprocessing:**
    Download the **ISIC2017 and ISIC2018** dataset and ensure the directory structure is organized as follows:
    ```text
    preprocess/
    ├── isic2017/
    │   ├── images/
    │   │   ├── ISIC_0000000.jpg
    │   │   ├── ISIC_0000001.jpg
    │   │   └── ...
    │   └── masks/
    │       ├── ISIC_0000000_segmentation.png
    │       ├── ISIC_0000001_segmentation.png
    │       └── ...
    ├── isic2018/
    │   ├── images/
    │   │   ├── ISIC_0000000.jpg
    │   │   ├── ISIC_0000001.jpg
    │   │   └── ...
    │   └── masks/
    │       ├── ISIC_0000000_segmentation.png
    │       ├── ISIC_0000001_segmentation.png
    │       └── ...
    ├── process.py
    └── split_data.py
    ```

Run the following Python scripts sequentially to convert the dataset into NumPy format:
```bash
python split_data.py
python process.py
```


## Run
```bash
python train.py  
```

## Results

<div align="center">
  <img width="90%" alt="Fig 7" src="https://github.com/user-attachments/assets/894551d1-ad80-49c0-8762-fc3a7fc0e9c4" />
</div>


