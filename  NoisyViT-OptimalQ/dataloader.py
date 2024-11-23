from __future__ import print_function
from re import L
import re
from parameters import *
import torch.utils.data as data
import random
import os
import numpy as np
import torch
import albumentations as A
import cv2
import pdb
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import RandAugment

def img_resize(img, img_resize):
    min_size = min(img.shape[0:2])
    retio = float(img_resize / min_size)
    width = int(img.shape[1] * retio)
    height = int(img.shape[0] * retio)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img




def get_imagenet(root, target_transform = None):
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
            transforms.RandomRotation(10),        # Randomly rotate the image by up to 10 degrees
            transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
            transforms.ToTensor(),                # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandAugment() 
        ])

        transform_val = transforms.Compose([
            transforms.Resize((opt.res, opt.res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tra_root = os.path.join(root,'train')
        trainset = datasets.ImageFolder(root=tra_root,
                                transform=transform_train,
                                target_transform=target_transform)
        val_root = os.path.join(root,'val')
        valset = datasets.ImageFolder(root=val_root,
                                transform=transform_val,
                                target_transform=target_transform)
        return trainset,valset


def get_loader(root):
    trainset, testset = get_imagenet(root)

    train_sampler = DistributedSampler(trainset) 
    test_sampler = DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              pin_memory = True,
                              #shuffle = True,  #ValueError: sampler option is mutually exclusive with shuffle
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=opt.te_batch_size,
                             num_workers=4,
                             pin_memory = True,
                             #shuffle = True
                             ) if testset is not None else None
    return train_loader, test_loader


if __name__ == '__main__':

    root_path = opt.imagenet_path
    train_loader, test_loader = get_loader(root_path)
    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( tra_transformed_normalized_img[0], (1,2,0)))
        plt.show()



