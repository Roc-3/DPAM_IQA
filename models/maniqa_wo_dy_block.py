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

        # slic
        ######################################################################
        self.feature_channel = 256 
        self.simi_matrix = SimilarityModule(self.feature_channel)
        self.slic_conv = nn.Conv2d(1, 3072, 1, 1, 0)
        
        # texture
        ######################################################################
        self.ta = TA(3072)
        self.t_conv = nn.Conv2d(3072, 3072, 1, 1, 0)

        # fusion attention block
        ######################################################################
        self.teablock = nn.ModuleList()
        for i in range(num_tab):
            tab = TEABlock(self.input_size ** 2)
            self.teablock.append(tab)
      
        self.fusionconv = nn.Conv2d(embed_dim * 4 * 2, embed_dim * 4, 1, 1, 0)

        # stage 1
        ######################################################################
        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        # vitblock
        self.block1 = Block(
            dim=embed_dim,
            num_heads=num_heads[0],
            mlp_ratio=dim_mlp / embed_dim,
            drop=drop,
        )
        # res res dckg
        self.resblock1 = ResBlockGroup(embed_dim, num_blocks=2)
        self.dyd1= DynamicDWConv(embed_dim , 3, 1, embed_dim)
        # reduce dim
        self.catconv1 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0) # stage1
        
        # tablock
        ######################################################################
        self.tablock = TABlock(self.input_size ** 2)

        # stage2
        ######################################################################
        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        # vitblock
        self.block2 = Block(
            dim=embed_dim // 2,
            num_heads=num_heads[1],
            mlp_ratio=dim_mlp / (embed_dim // 2),
            drop=drop,
        )     
        # res res dckg
        self.resblock2 = ResBlockGroup(embed_dim // 2, num_blocks=2)
        self.dyd2= DynamicDWConv(embed_dim // 2, 3, 1, embed_dim // 2)
        # reduce dim
        self.catconv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)

        # fc
        ######################################################################
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )
    
    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x
    
    def forward(self, x, x_texture, x_slic_pix):
        # slic query
        ######################################################################
        x_slic = self.simi_matrix(x_slic_pix) # torch.Size([20, 1, image_n_nodes, image_n_nodes])
        x_slic = self.slic_conv(x_slic) # torch.Size([20, 3072, 300, 300])
        x_slic = F.interpolate(x_slic, 
                               size=(self.input_size, self.input_size), 
                               mode='bilinear', align_corners=False) # torch.Size([20, 3072, 28, 28])

        # texture query
        ######################################################################
        x_ta = self.ta(x_texture)
        x_texture = self.t_conv(x_ta) # torch.Size([1, 3072, 28, 28])

        # vit features
        ######################################################################
        _x = self.vit(x)
        x = self.extract_feature(self.save_output) # torch.Size([28, 784, 3072])
        self.save_output.outputs.clear()

        ######################################################################
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        x_texture = rearrange(x_texture, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        x_slic = rearrange(x_slic, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        
        # fusion attention block
        ######################################################################
        x_fusion1 = self.teablock[0](x_texture, x) # torch.Size([20, 3072, 784])
        """相似度矩阵下采样和vit特征进行融合"""
        x_fusion2 = self.teablock[1](x_slic, x) # torch.Size([20, 3072, 784])
        x = torch.cat((x_fusion1, x_fusion2), dim=1)

        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.fusionconv(x)
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)

        ######################################################################
        
        # stage 1
        ######################################################################
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)

        x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x_vit = self.block1(x_vit)
        x_vit1 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)

        x_res1 = self.resblock1(x)
        # x_res1 = self.dyd1(x_res1)

        x = torch.cat((x_vit1, x_res1), dim=1) 
        x = self.catconv1(x) # torch.Size([12, 768, 28, 28])
        
        ######################################################################
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        x = self.tablock(x)
        # for tab in self.tablock:
        #     x = tab(x)        
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        ######################################################################
  
        # stage2
        ######################################################################
        x = self.conv2(x)

        x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x_vit = self.block2(x_vit)
        x_vit2 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)

        x_res2 = self.resblock2(x)
        # x_res2 = self.dyd2(x_res2)

        x = torch.cat((x_vit2, x_res2), dim=1)
        x = self.catconv2(x) # torch.Size([2, 384, 28, 28])

        # fc
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score