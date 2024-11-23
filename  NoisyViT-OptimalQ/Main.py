
from ViT import *
import numpy as np
import itertools
import time
from parameters import *
import torch
from Accdataloader import *
from utils import *
import timm
from timm.loss import LabelSmoothingCrossEntropy
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import logging
from torch.utils.tensorboard import SummaryWriter
if(opt.OptimalQ):
    writer = SummaryWriter('./output/optimal')
else:
    writer = SummaryWriter('./output/linear_'+str(int(10*opt.strength)))

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def train():

    if(opt.scale == 'tiny'):
        Layers = 12
        HiddenSize = 192
        Heads = 3
        MLPSize = 768   #MLP ratio 4
    elif(opt.scale == 'small'):
        Layers = 12
        HiddenSize = 384
        Heads = 12
        MLPSize = 1536   #MLP ratio 4
    elif(opt.scale == 'base'):
        Layers = 12
        HiddenSize = 768
        Heads = 12
        MLPSize = 3072
    elif(opt.scale == 'large'):
        Layers = 24
        HiddenSize = 1024
        Heads = 16
        MLPSize = 4096
    elif(opt.scale == 'huge'):
        Layers = 32
        HiddenSize = 1280
        Heads = 16
        MLPSize = 5120

    noise_vit = NoiseViT(patch_size = opt.patch_size, num_classes=opt.num_classes, embed_dim=HiddenSize, depth=Layers, num_heads=Heads)
    noise_vit.to(device)
    
    if(1):
        print('pretrained mode: '+'vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res))
        if(opt.scale == 'huge'):
            vit_pretrain = timm.create_model('vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res)+'_in21k', pretrained=True, num_classes = opt.num_classes)
        else:
            vit_pretrain = timm.create_model('vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res), pretrained=True, num_classes = opt.num_classes)  #vit_small_patch16_224, vit_base_patch16_224
        vit_pretrain_weight = vit_pretrain.state_dict()
        noise_vit.load_state_dict(vit_pretrain_weight)   
        #if local_rank == 0:
        #    dist.barrier()
        last_th = 0


    print('model parameters:', sum(param.numel() for param in noise_vit.parameters()) /1e6)

    if(opt.datasets == 'ImageNet'):
        tra_dataloader, te_dataloader = get_loader(opt.imagenet_path)
    elif(opt.datasets == 'TinyImageNet'):
        tra_dataloader, te_dataloader = get_loader(opt.tinyImagenet_path)
    elif(opt.datasets == 'Cifar100'):
        train_data = datasets.CIFAR100(root='./data', train=True,transform=transform_train,download=True)
        test_data =datasets.CIFAR100(root='./data',train=False,transform=transform_test,download=True)
        tra_dataloader = DataLoader(dataset=train_data,batch_size=opt.batch_size,shuffle=True,)
        te_dataloader = DataLoader(dataset=test_data,batch_size=opt.batch_size,shuffle=False,)
    elif(opt.datasets == 'Cifar10'):
        train_data = datasets.CIFAR10(root='./data', train=True,transform=transform_train,download=True)
        test_data =datasets.CIFAR10(root='./data',train=False,transform=transform_test,download=True)
        tra_dataloader = DataLoader(dataset=train_data,batch_size=opt.batch_size,shuffle=True,)
        te_dataloader = DataLoader(dataset=test_data,batch_size=opt.batch_size,shuffle=False,)
    
    start_t = time.time()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, noise_vit.parameters()), 
                                        lr=opt.lr, weight_decay=opt.weight_decay,)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, noise_vit.parameters()), 
    #                                    lr=opt.lr, weight_decay=opt.weight_decay, momentum = 0.9)
    #optimizer = create_optimizer(opt, noise_vit)
    #scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs/2, eta_min=1e-6 )
    scheduler = CosineLRScheduler(optimizer, t_initial=opt.epochs, lr_min=1e-7, warmup_t=opt.warm_up)
    loss_fn = LabelSmoothingCrossEntropy(0.1)

    for epoch in range(opt.epochs):
        noise_vit.train()

        num_steps_per_epoch = len(tra_dataloader)
        num_updates = epoch * num_steps_per_epoch

        batch_time = AverageMeter()  # forward prop. + back prop. time
        accs = AverageMeter()
        epoch_tra_loss = AverageMeter()  
        
        total = 0
        correct = 0
        #tra_dataloader.sampler.set_epoch(epoch)  # randomize the training data
        pbar = tqdm(enumerate(tra_dataloader),total=len(tra_dataloader),)
        #for i, (tra_transformed_normalized_img, tra_labels) in enumerate(tra_dataloader):
        for i, (tra_transformed_normalized_img, tra_labels) in pbar:
            batchSize = tra_transformed_normalized_img.shape[0]
            #print('current lr: ', (optimizer.state_dict()['param_groups'][0]['lr']))

            #------------------------------------------------------
            tra_transformed_normalized_img = tra_transformed_normalized_img.float().to(device)
            
            outputs = noise_vit(tra_transformed_normalized_img, noise_layer_index = opt.layer, strength = opt.strength, \
                noise_type = opt.noise_type, train=opt.tra) 
            cls_loss = loss_fn(outputs, tra_labels.cuda())

            #acc---------------------------------------------------------------------
            _, predictions = torch.max(outputs, 1)

            total += tra_labels.size(0)
            correct += (predictions == tra_labels.to(device)).sum().item()
            #------------------------------------------------------------------------

            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step() 
            scheduler.step_update(num_updates=num_updates)      

            epoch_tra_loss.update(cls_loss.detach())
            accs.update(correct/total)
            batch_time.update(time.time() - start_t)
            
            #pbar.set_description(f"epoch {epoch + 1} iter {i}: train loss {cls_loss.item():.3f}. lr {scheduler.get_last_lr()[0]:e}")
            # Print log info     
            
            if i % 200 == 5:
                # print('======================== print results \t' + time.asctime(time.localtime(time.time())) + '=============================')
                print('Train Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(epoch, i, len(tra_dataloader),
                                                                            batch_time=batch_time,
                                                                            loss=epoch_tra_loss,
                                                                            top1=accs,))
        #logger.info(f'Total local training loss: {epoch_tra_loss.avg}, acc: {accs.avg}', main_process_only= True)
          

        scheduler.step(epoch+1)
        writer.add_scalar('clssification_train_loss', epoch_tra_loss.avg, epoch)
        writer.add_scalar('train_acc', accs.avg, epoch)


        #############################################################################

        accs = AverageMeter()
        epoch_te_loss = AverageMeter()
        accs_top5 = AverageMeter()  

        total = 0
        correct = 0
        correct_top5 = 0
        noise_vit.eval()
        for i,(te_transformed_normalized_img, te_labels) in enumerate(te_dataloader):
            te_transformed_normalized_img = te_transformed_normalized_img.float().cuda()

            with torch.no_grad():
                outputs = noise_vit(te_transformed_normalized_img, noise_layer_index = opt.layer, strength = opt.strength, \
                    noise_type = opt.noise_type, train=opt.inf) 
                cls_loss = loss_fn( (outputs), (te_labels.to(device)) )
       
                #acc---------------------------------------------------------------------
                _, predictions = torch.max(outputs, 1)

                total += (te_labels).size(0)
                correct += ( (predictions) == (te_labels.to(device)) ).sum().item()  #.cpu()
                #------------------------------------------------------------------------
                epoch_te_loss.update( cls_loss.detach() )
                accs.update(correct/total)  #acc = correct/total

                # Get top 5 predictions
                _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)

                # Check if te_labels are in the top 5 predictions
                correct_top5 += (top5_pred == te_labels.view(-1, 1).to(device)).sum().item()
                accs_top5.update(correct_top5/total)  #acc = correct/total
        
            # Print log info
            #print('te batch size', len(te_labels))
            if i % 5 == 0:  #opt.log_step
                # print('======================== print results \t' + time.asctime(time.localtime(time.time())) + '=============================')
                print('Test Epoch: [{0}][{1}/{2}]\t'                 
                    'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(te_dataloader),
                                                                                loss=epoch_te_loss,
                                                                                top1=accs,
                                                                                top5=accs_top5))
        writer.add_scalar('clssification_test_loss', epoch_te_loss.avg, epoch)
        writer.add_scalar('test_acc', accs.avg, epoch)  #add_scalar
        writer.add_scalar('test_top5 acc', accs_top5.avg, epoch)  #add_scalar
        #logger.info(f'Total local eval loss: {epoch_te_loss.avg}, acc: {accs.avg}', main_process_only= True)              
        if( accs.avg > last_th):
            last_th = (accs.avg) #.cpu().item()
            if(opt.noise_type == 'gaussian'):
                torch.save(noise_vit.state_dict(),  opt.model_saved_path + '/'+'acc_'+str(last_th)+ '_lr_'+str(opt.lr)+'_strth_'+str(opt.strength)+'_layer_'+str(opt.layer)+'_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+opt.noise_type+\
                '_'+ opt.datasets +'_'+str(opt.gau_mean)+'_'+str(opt.gau_var)+'_NV.pkl') 
            else:
                if(opt.OptimalQ == 1):
                    torch.save(noise_vit.state_dict(),  opt.model_saved_path + '/'+'acc_'+str(last_th)+ '_lr_'+str(opt.lr)+'_bs_'+str(opt.batch_size)+'_layer_'+str(opt.layer)+'_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+opt.noise_type+\
                    '_'+ opt.datasets +'_NoisyViT.pkl') 
                else:
                    torch.save(noise_vit.state_dict(),  opt.model_saved_path + '/'+'acc_'+str(last_th)+ '_lr_'+str(opt.lr)+'_bs_'+str(opt.batch_size)+'_layer_'+str(opt.layer)+'_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+opt.noise_type+\
                    '_'+ opt.datasets +'_OrdinaryViT.pkl') 
            
        #print('%d epoch done' % epoch)

    writer.close()

if __name__ == '__main__':

    set_seed(42)

    train()

    

