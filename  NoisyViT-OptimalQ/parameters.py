import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--gau_mean', default= 0.0, type= float, help='gaussian mean,[-1,1,0.5]')
parser.add_argument('--gau_var', default= 1.0, type= float, help='gaussian variance [0,2,0.5]')
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--model_saved_path', type=str, default='./')
parser.add_argument('--imagenet_path', type=str, default='./ImageNet1K/')
parser.add_argument('--tinyImagenet_path', type=str, default='./')
parser.add_argument('--log_step', type=int, default=2000, help='log_step')
parser.add_argument('--batch_size', type=int, default=10)  
parser.add_argument('--te_batch_size', type=int, default=10)  
parser.add_argument('--warm_up', default=10, type=int)
parser.add_argument('--noise_type', default='liner', type=str, help='gaussian, linear, impulse')
parser.add_argument('--layer', default=11, type=int)
parser.add_argument('--strength', default=0.1, type=float)
parser.add_argument('--gpu_id', default='1', type = str, help="gpu id")
parser.add_argument('--res', default=384, type = int,
                        help="image resolution")
parser.add_argument('--patch_size', default=16, type = int,
                        help="patch size")
parser.add_argument('--scale', default='tiny', type = str,
                        help="model scale")
parser.add_argument('--datasets', default='ImageNet', type = str,
                        help="choose dataset to experiment, ImageNet, TinyImageNet, Cifar100, Cifar10")
parser.add_argument('--num_classes', default=1000, type = int,
                        help="number of class categories")
parser.add_argument('--tra', default=1, type = int,
                        help="")
parser.add_argument('--inf', default=1, type = int,
                        help="")
parser.add_argument('--OptimalQ', default=0, type = int,
                        help="use OptimalQ.")

#parser.add_argument("--local_rank", type= int, default = -1)
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

#if torch.cuda.device_count() > 1:
#          print("Let's use", torch.cuda.device_count(), "GPUs!")
