import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
# from models.swin import SwinTransformer
from torch import nn
from einops import rearrange
from models.resnet import ResBlockGroup
from models.newDyD import DynamicDWConv
import torch.nn.functional as F

from models.simimatrix import SimilarityModule

class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)
   
    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x

class TEABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, query_features, image_features):
        _x = image_features # torch.Size([20, 3072, 784])
        B, C, N = image_features.shape
        q = self.c_q(query_features) # 纹理作为query torch.Size([20, 768, 784])
        k = self.c_k(image_features)  #torch.Size([20, 3072, 784])
        v = self.c_v(image_features)

        attn = (q @ k.transpose(-2, -1)) * self.norm_fact
        attn = self.softmax(attn)
        x = attn @ v.transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x # 保留自连接
        return x

class TA(nn.Module): # texture attention
    def __init__(self, channels):
        super(TA, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((28, 28))
        self.texture_attention = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    def forward(self, texture_image):
        x_ta = self.pool(texture_image) # torch.Size([28, 3, 28, 28])
        x_ta = self.texture_attention(x_ta)
        return x_ta

class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.conv1 = nn.Conv2d(embed_dim * 4, 384, 1, 1, 0)
        self.fc_score = nn.Linear(384 * self.input_size * self.input_size, num_outputs)
    
    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x
    
    def forward(self, x, x_texture, x_slic_pix):
        # vit features
        ######################################################################
        _x = self.vit(x)
        x = self.extract_feature(self.save_output) # torch.Size([28, 784, 3072])
        self.save_output.outputs.clear()

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x.view(x.size(0), -1)  # 展平特征
        ######################################################################
        
        # fc
        score = self.fc_score(x)
        return score