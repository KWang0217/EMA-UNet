EMA-UNet: Efficient Multi-Attention UNet for Skin Lesion Segmentation

<img width="517" height="327" alt="Fig 3" src="https://github.com/user-attachments/assets/110ead52-30a9-406c-8e86-0b1ba3c81c13" />

Fig The overview of EMA-UNet

Skin cancer, particularly malignant melanoma, poses a significant threat to global public health, and accurate skin lesion segmentation is critical for its early diagnosis and treatment. 
Recently, with the increasing demand for automated skin lesion diagnosis, deep learning has achieved remarkable progress in segmentation tasks. However, existing methods generally suffer 
from large parameter counts and high computational complexity, which severely restricts their deployment in resource-constrained scenarios such as mobile health. To address this challenge,
this paper proposes EMA-UNet, a lightweight and efficient model based on the U-Net architecture. The model integrates three core modules: Group Multi-Axis Hadamard Product Attention (GHPA),
the proposed Spatial-Channel Enhancement (SCE), and Dynamic Feature Fusion (DFF). EMA-UNet achieves high-performance segmentation with minimal parameters (0.068 M) and computational complexity
(0.066 GFLOPs), demonstrating superior mIoU values of 84.20% and 84.32% on the ISIC2017 and ISIC2018 datasets, respectively. This model provides an effective solution for resource-constrained 
environments, striking a balance between accuracy and low resource consumption. 
