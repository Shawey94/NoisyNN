from __future__ import print_function
from re import L
import re
import torch.utils.data as data
import random
import os
import numpy as np
import torch
import albumentations as A
import cv2
import pdb
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from torchvision.datasets import ImageFolder
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg
from parameters import *


class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_integrity():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images



def img_resize(img, img_resize):
    min_size = min(img.shape[0:2])
    retio = float(img_resize / min_size)
    width = int(img.shape[1] * retio)
    height = int(img.shape[0] * retio)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img


class CP_Net_Dataset(data.Dataset):
    def __init__(self, data_training, data_testing, train=True, test=False, img_resize=224):
        self.train = train  # training set or val set
        self.test = test
        self.img_resize = img_resize

        # pdb.set_trace()
        if self.train:
            self.data = data_training
            self.transform = A.Compose([A.GaussNoise(p=0.2), 
                                        A.Resize(height = 224, width = 224, p =1.0), 
                                        A.HorizontalFlip(p = 0.5), 
                                        A.RandomBrightnessContrast(p=0.2), 
                                        A.Flip(p=0.2), 
                                        A.Normalize(mean=(0.5,0.5,0.5),std=(0.3,0.3,0.3),p=1.0),
                                        ])
        if self.test:
            self.data = data_testing
            self.transform = A.Compose([A.GaussNoise(p=0.2), 
                                        A.Resize(height = 224, width = 224, p =1.0), 
                                        A.HorizontalFlip(p = 0.5), 
                                        A.RandomBrightnessContrast(p=0.2), 
                                        A.Flip(p=0.2), 
                                        A.Normalize(mean=(0.5,0.5,0.5),std=(0.3,0.3,0.3),p=1.0),
                                        ])
        random.shuffle(self.data)


    def __getitem__(self, index):
        img_path, label_ID = self.data[index]
        img = cv2.imread(img_path)
        #resized_img = img_resize(img, self.img_resize)
        #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img)
        transformed_img = transformed["image"]
        #transformed_normalized_img = (transformed_img/255.)*2 - 1

        return transformed_img, label_ID

    def debug_getitem__(self, index=0):
        img_path, label_ID = self.data[index]
        img = cv2.imread(img_path)
        # if not self.train and not self.test:
        #     print(img_path)
        #     print(img.shape)
        # cv2.imwrite('./image_orig_resize_transform/' + str(index) + '_orig_(' + str(img.shape[0]) + '_' + str(img.shape[1]) + ')' + '.jpg', img)
        #resized_img = img_resize(img, self.img_resize)
        # cv2.imwrite('./image_orig_resize_transform/' + str(index) + '_resize_(' + str(resized_img.shape[0]) + '_' + str(resized_img.shape[1]) + ')' + '.jpg', resized_img)
        #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img)
        transformed_img = transformed["image"]
        # cv2.imwrite('./image_orig_resize_transform/' + str(index) + '_transformed_(' + str(transformed_img.shape[0]) + '_' + str(transformed_img.shape[1]) + ')' + '.jpg', cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
        #transformed_normalized_img = (transformed_img/255.)*2 - 1

        print(transformed_img.shape)
        #pdb.set_trace()

        return img_path, transformed_img, label_ID

    def __len__(self):
        return len(self.data)


def pil_loader(img):
    return img.convert('RGB')


def get_imagenet(root, target_transform = None):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
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

    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=4,
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.batch_size,
                             num_workers=4,
                             ) if testset is not None else None
    return train_loader, test_loader

#05/01/2023
if __name__ == '__main__':


    #train_dataset = TinyImageNet('/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/CP_ViT_TinyImageNet', split='train', download=True)
    #test_dataset = TinyImageNet('/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/CP_ViT_TinyImageNet', split='val', download=True)

    root_path = opt.imagenet_path
    train_loader, test_loader = get_loader(root_path)
    for i, (te_transformed_normalized_img, tra_labels) in enumerate(test_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( te_transformed_normalized_img[0], (1,2,0)))
        plt.show()



