# [NoisyNN: Exploring the Impact of Information Entropy Change in Learning Systems](https://arxiv.org/pdf/2309.10625)

### updates (11/20/2024)

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
Download the pretrained models and put them in `checkpoint/`
- ViT-B_16 [(ImageNet-21K)](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.49067683.-40935391.1637977007)

TVT with ViT-B_16 (ImageNet-21K) performs a little bit better than TVT with ViT-B_16 (ImageNet):
<p align="left"> 
<img width="500" src="https://github.com/uta-smile/TVT/blob/main/ImageNet_vs_ImageNet21K.png">
</p>

### Datasets:

- Download [data](https://drive.google.com/file/d/1rnU49vEEdtc3EYVo7QydWzxcSuYqZbUB/view?usp=sharing) and replace the current `data/`

- Download images from [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and put them under `data/`. For example, images of `Office-31` should be located at `data/office/domain_adaptation_images/`

### Training:

Commands can be found in `script.txt`. An example:
```
python3 main.py --train_batch_size 64 --dataset office --name wa \
--source_list data/office/webcam_list.txt --target_list data/office/amazon_list.txt \
--test_list data/office/amazon_list.txt --num_classes 31 --model_type ViT-B_16 \
--pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 \
--beta 0.1 --gamma 0.01 --use_im --theta 0.1
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
Our code is largely borrowed from [Timm]([https://github.com/jeonsworld/ViT-pytorch](https://github.com/huggingface/pytorch-image-models/tree/main/timm))
