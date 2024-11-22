# [NoisyNN: Exploring the Impact of Information Entropy Change in Learning Systems](https://arxiv.org/pdf/2309.10625)

### updates (11/20/2024)
-The learning theory proposed in this work primarily enhances model performance in single-modality classification tasks, including image classification, domain adaptation/generalization, semi-supervised classification, and text classification.

-Applications of NoisyNN in semi-supervised learning and domain adaptation have been accepted at ICML 2024 and WACV 2025. 

-NoisyNN shows significant potential for other learning tasks, which I will explore further.

<p align="left"> 
<img width="800" src="https://github.com/Shawey94/NoisyNN/blob/main/NoisyNNMethod.png">
</p>

### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install tqdm==4.50.2
pip install tensorboard==2.8.0
# apex 0.1
conda install -c conda-forge nvidia-apex
pip install scipy==1.5.2
pip install ml-collections==0.1.0
pip install scikit-learn==0.23.2
pip install timm==0.6.13
pip install torchvision==0.16.2
pip install albumentations==1.3.0
pip install accelerate==0.18.0
```

### Pretrained ViT
-NoisyViT with ViT-B_16 (pre-trained on ImageNet-21K) achieved a top 1 accuracy of over 95% and a top 5 accuracy of 100% on ImageNet-1K:
<p align="left"> 
<img width="500" src="https://github.com/Shawey94/NoisyNN/blob/main/ResImageNet.png">
</p>

### Datasets:

- Download the ImageNet-1K(https://www.image-net.org/download.php) dataset.

### Training:

Commands can be found in `script.txt`. An example:
```
python Main.py --lr 0.000001 --epochs 50 --batch_size 16 --layer 11 --gpu_id 0 --res 384 --patch_size 16 --scale base --noise_type linear --datasets ImageNet --num_classes 1000 --tra 0 --inf 1 --OptimalQ 1
```

### Citation:
```
@article{Yu2023NoisyNN,
  title={NoisyNN: Exploring the Impact of Information Entropy Change in Learning Systems},
  author={Yu, Xiaowei and Huang, Zhe and Xue, Yao and Zhang, Lu and Wang, Li and Liu, Tianming and Dajiang Zhu},
  journal={arXiv preprint arXiv:2309.10625},
  year={2023}
}

@article{Huang2024InterLUDE,
  title={InterLUDE: Interactions between Labeled and Unlabeled Data to Enhance Semi-Supervised Learning},
  author={Huang, Zhe and Yu, Xiaowei and Zhu, Dajiang and Michael C. Hughes},
  journal={International Conference on Machine Learning},
  year={2024}
}

@article{Yu2025FFTAT,
  title={Feature Fusion Transferability Aware Transformer for Unsupervised Domain Adaptation},
  author={Yu, Xiaowei and Huang, Zhe and Zhang, Zao},
  journal={IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2025}
}
```
Our code is largely borrowed from [Timm](https://github.com/huggingface/pytorch-image-models/tree/main/timm)

[InterLUDE: Interactions between Labeled and Unlabeled Data to Enhance Semi-Supervised Learning](https://github.com/tufts-ml/InterLUDE)
[InterLUDE: Interactions between Labeled and Unlabeled Data to Enhance Semi-Supervised Learning](https://github.com/Shawey94/WACV2025-FFTAT)
