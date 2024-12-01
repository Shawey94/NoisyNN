# -*- coding: utf-8 -*-
"""
@date: 2023/3/11 4:00PM
@file: CNNs.py
@author: shoreway
@description: 
"""
import torch
import torch.nn as nn
import CNNModels as models
from CNNModels import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, ResNet34_Weights, ResNet18_Weights
from parameters import * 

def Gaussian(x):
    #credit: http://blog.moephoto.tech/pytorch%e7%94%9f%e6%88%90%e5%8a%a0%e6%80%a7%e9%ab%98%e6%96%af%e7%99%bd%e5%99%aa%e5%a3%b0awgn/
    s1,s2,s3,s4 = x.shape
    means= args.gau_mean * torch.ones(s1,s2,s3,s4)
    stds = args.gau_var * torch.ones(s1,s2,s3,s4)
    gaussian_noise = (torch.normal(means, stds)).to(x.device)
    return args.noise_str * gaussian_noise + x

def Impulse(x,prob): #salt_and_pepper noise. strength has no meaning here, use the probality [0,1] to control
    #credit: https://blog.csdn.net/jzwong/article/details/109159682
    noise_tensor=torch.rand(x.size())
    salt=(torch.max(x.clone())).detach()
    pepper=(torch.min(x.clone())).detach()
    x_clone = x.clone()
    #x[noise_tensor<prob/2]=salt    #cause in-place graident computation error
    #x[noise_tensor>prob + prob/2]=pepper  #cause in-place graident computation error
    x_clone[noise_tensor<prob/2]=salt
    x_clone[noise_tensor>1-prob/2]=pepper
    return x_clone


class ResNet152(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet152, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet152(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer1[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer2[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer3[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer4[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x

class ResNet101(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet101, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet101(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer1[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer2[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer3[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer4[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x
 
class ResNet50(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet50, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet50(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer1[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer2[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer3[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer4[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x
    
class ResNet34(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet34, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet34(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer1[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer2[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer3[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer4[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x
    
class ResNet18(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet18, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet18(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer1[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer2[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x.detach())
                        x = self.model.layer3[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.clone()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        noise = (x_copy - x) 
                        x = self.model.layer4[temp_sub](x+ self.noise_str * noise )
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x
