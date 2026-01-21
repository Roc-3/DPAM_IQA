import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import torch.optim as optim

from timm.models.vision_transformer import Block
from torch import nn
from einops import rearrange
from models.resnet import ResBlockGroup
from models.newDyD import DynamicDWConv
from models.simimatrix import SimilarityModule

# ==========================================
# CNNRIM from DSN-IQA/spix_rim.py
# ==========================================

def conv_in_relu(in_c, out_c):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_c, out_c, 3, bias=False),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.ReLU()
    )

class CNNRIM(nn.Module):
    """
    code for
    T.Suzuki, ICASSP2020
    Superpixel Segmentation via Convolutional Neural Networks with Regularized Information Maximization
    https://arxiv.org/abs/2002.06765
    """

    def __init__(self, in_c=5, n_spix=100, n_filters=32, n_layers=5, use_recons=True, use_last_inorm=True):
        super().__init__()
        self.n_spix = n_spix
        self.use_last_inorm = use_last_inorm
        self.use_recons = use_recons
        self.num_class = 39
        self._criterion = torch.nn.MSELoss()
        out_c = n_spix
        if use_recons:
            out_c += 3

        layers = []
        for i in range(n_layers - 1):
            layers.append(conv_in_relu(in_c, n_filters << i))
            in_c = n_filters << i
        layers.append(nn.Conv2d(in_c, out_c, 1))
        self.layers = nn.Sequential(*layers)
        if use_last_inorm:
            self.norm = nn.InstanceNorm2d(n_spix, affine=True)

        self.classifier = nn.Linear(256, self.num_class)
        self.initThePara()

    def initThePara(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                m.bias.data.zero_()

    def forward(self, x):
        spix = self.layers(x)
        if self.use_recons:
            recons, spix = spix[:, :3], spix[:, 3:]
        else:
            recons = None
        if self.use_last_inorm:
            spix = self.norm(spix)
        return spix, recons

    def mutual_information(self, logits, coeff):
        prob = logits.softmax(1)
        pixel_wise_ent = - (prob * F.log_softmax(logits, 1)).sum(1).mean()
        marginal_prob = prob.mean((2, 3))
        marginal_ent = - (marginal_prob * torch.log(marginal_prob + 1e-16)).sum(1).mean()
        return pixel_wise_ent - coeff * marginal_ent

    def smoothness(self, logits, image):
        prob = logits.softmax(1)
        dp_dx = prob[..., :-1] - prob[..., 1:]
        dp_dy = prob[..., :-1, :] - prob[..., 1:, :]
        di_dx = image[..., :-1] - image[..., 1:]
        di_dy = image[..., :-1, :] - image[..., 1:, :]

        return (dp_dx.abs().sum(1) * (-di_dx.pow(2).sum(1) / 8).exp()).mean() + \
               (dp_dy.abs().sum(1) * (-di_dy.pow(2).sum(1) / 8).exp()).mean()

    def reconstruction(self, recons, image):
        return F.mse_loss(recons, image)

    def __preprocess(self, image, device="cuda"):
        # image is already a tensor (C, H, W) on device
        image = image.unsqueeze(0)  # 1 3 H W
        h, w = image.shape[-2:]
        coord = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))).float()[None]
        input = torch.cat([image, coord], 1)
        # Add normalization to balance RGB and XY channels
        input = (input - input.mean((2, 3), keepdim=True)) / input.std((2, 3), keepdim=True)
        return input

    def optimize(self, image, n_iter=500, lr=1e-2, lam=2, alpha=2, beta=10, device="cuda"):
        # image: Tensor (3, H, W)
        input = self.__preprocess(image, device)
        self.initThePara()
        optimizer = optim.Adam(self.parameters(), lr)

        for i in range(n_iter):
            spix, recons = self.forward(input)

            loss_mi = self.mutual_information(spix, lam)
            loss_smooth = self.smoothness(spix, input)
            loss = loss_mi + alpha * loss_smooth
            if recons is not None:
                loss = loss + beta * self.reconstruction(recons, input[:, :3])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        spix, recons = self.forward(input)
        return spix.detach()

# ==========================================
# MANIQA from models/maniqa.py
# ==========================================

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
        _x = image_features 
        B, C, N = image_features.shape
        q = self.c_q(query_features) 
        k = self.c_k(image_features)  
        v = self.c_v(image_features)

        attn = (q @ k.transpose(-2, -1)) * self.norm_fact
        attn = self.softmax(attn)
        x = attn @ v.transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x 
        return x

class TA(nn.Module): 
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
        x_ta = self.pool(texture_image) 
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
        x_slic = self.simi_matrix(x_slic_pix) 
        x_slic = self.slic_conv(x_slic) 
        x_slic = F.interpolate(x_slic, 
                               size=(self.input_size, self.input_size), 
                               mode='bilinear', align_corners=False) 

        # texture query
        ######################################################################
        x_ta = self.ta(x_texture)
        x_texture = self.t_conv(x_ta) 

        # vit features
        ######################################################################
        _x = self.vit(x)
        x = self.extract_feature(self.save_output) 
        self.save_output.outputs.clear()

        ######################################################################
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        x_texture = rearrange(x_texture, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        x_slic = rearrange(x_slic, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        
        # fusion attention block
        ######################################################################
        x_fusion1 = self.teablock[0](x_texture, x) 
        x_fusion2 = self.teablock[1](x_slic, x) 
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
        x_res1 = self.dyd1(x_res1)

        x = torch.cat((x_vit1, x_res1), dim=1) 
        x = self.catconv1(x) 
        
        ######################################################################
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        x = self.tablock(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        ######################################################################
  
        # stage2
        ######################################################################
        x = self.conv2(x)

        x_vit = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x_vit = self.block2(x_vit)
        x_vit2 = rearrange(x_vit, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)

        x_res2 = self.resblock2(x)
        x_res2 = self.dyd2(x_res2)

        x = torch.cat((x_vit2, x_res2), dim=1)
        x = self.catconv2(x) 

        # fc
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([], device=x.device)
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score
