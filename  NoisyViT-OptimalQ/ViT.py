
import torch
from torch import nn
from torch._C import set_flush_denormal
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os
from torch.autograd import Variable
from parameters import *
import seaborn as sns
import matplotlib.pyplot as plt
#from dataloader import *
from timm.models.vision_transformer import partial
from timm.models.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import LayerScale
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import Mlp
from utils import *
import sys

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

def OptimalQ(x):
    s1,s2,s3 = x.shape
    hidden_states_copy = x.clone()
    noise = 0 
    for i in range(1,s1):  
        sub_noise = torch.cat( (hidden_states_copy[i:,1:,:], hidden_states_copy[0:i,1:,:]),dim=0 )
        noise = noise + sub_noise 
    x[:,1:,:] = x[:,1:,:] - x[:,1:,:].clone() * (s1-1) /(s1+1) + noise / (s1+1)   
    return x

def Gaussian(x):
    #credit: http://blog.moephoto.tech/pytorch%e7%94%9f%e6%88%90%e5%8a%a0%e6%80%a7%e9%ab%98%e6%96%af%e7%99%bd%e5%99%aa%e5%a3%b0awgn/
    s1,s2,s3 = x.shape
    means= opt.gau_mean * torch.ones(s1,s2-1,s3) 
    stds = opt.gau_var * torch.ones(s1,s2-1,s3)
    gaussian_noise = (torch.normal(means, stds)).to(x.device)
    #print('guassian noise size:', gaussian_noise.size())
    #print('x[:,1:,:] size:', x[:,1:,:].size())
    x[:,1:,:] = opt.strength * gaussian_noise + x[:,1:,:]
    return x

def Impulse(x,prob): #salt_and_pepper noise. strength has no meaning here, use the probality [0,1] to control
    #credit: https://blog.csdn.net/jzwong/article/details/109159682
    s1,s2,s3 = x.shape
    noise_tensor=torch.rand(s1,s2-1,s3) #s2-1: to keep class token unchanged
    x_clone = x.clone()
    salt=(torch.max(x_clone[:,1:,:])).detach()
    pepper=(torch.min(x_clone[:,1:,:])).detach()
    #x[noise_tensor<prob/2]=salt    #cause in-place graident computation error
    #x[noise_tensor>prob + prob/2]=pepper  #cause in-place graident computation error
    x_clone[:,1:,:][noise_tensor<prob/2]=salt
    x_clone[:,1:,:][noise_tensor>1-prob/2]=pepper
    return x_clone


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, flag, strength = 0.0, noise_type = 'linear', train=False):
        
        if(flag and train):
            if(noise_type == 'linear'):
                if(opt.OptimalQ):
                    x = OptimalQ(x)
            elif(noise_type == 'gaussian'):
                x = Gaussian(x)
            elif(noise_type == 'impulse'):
                x = Impulse(x, strength)     
            else:
                print('no noise usage!')

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, flag = False, strength = 0.0, noise_type='linear', train=False):  # default: not add noise
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), flag, strength, noise_type, train)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=opt.res, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) # Batch * 3 * 224 *224 -> Batch * 768 * 14 * 14
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, noise_layer_index, strength):
        x = self.patch_embed(x)

        if(noise_layer_index == -1):  #-1 means after embedding before attn
            hidden_states_copy = x.detach()
            hidden_states_copy = torch.cat( (hidden_states_copy[1:,:,:], hidden_states_copy[0,:,:].unsqueeze(0)),dim=0 )
            x = x + strength * (hidden_states_copy - x.detach())

        x = self._pos_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            for i in range(len(self.blocks)):
                if( i == noise_layer_index ):
                    x = self.blocks[i](x, True, strength) # add noise to the layer
                else:
                    x = self.blocks[i](x, False) # not add noise to the layer
            #x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    '''
    def forward(self, x, noise_layer_index, strength):
        x = self.forward_features(x, noise_layer_index, strength)
        x = self.forward_head(x)
        return x
    '''

class NoiseViT(VisionTransformer):
    def __init__(self, patch_size, num_classes, embed_dim, depth, num_heads, ):
        super().__init__(patch_size=patch_size, num_classes=num_classes, embed_dim = embed_dim, depth = depth, num_heads = num_heads, img_size = opt.res)
        self.layer = depth  
    
    def forward(self, x, strength = 0.0, noise_layer_index = 11, noise_type = 'linear', train=True):   #default to the last layer
        x = self.patch_embed(x)
        B, num_patches = x.shape[0], x.shape[1]

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim = 1)
        x = self.pos_drop(x + self.pos_embed)
        for i in range(self.layer):
            if(i == noise_layer_index):
                x = self.blocks[i](x, True, strength, noise_type, train)
            else:
                x = self.blocks[i](x, False, strength, noise_type, train)
        x = self.norm(x)
        x = self.fc_norm(x)
        return self.head(x[:,0])








if __name__ == '__main__':

    print('debug purpose.')

    
