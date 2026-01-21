import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import sys
import os

# Add workspace root to sys.path to allow importing AINET
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AINET.models.Spixel_single_layer import SpixelNet1l_bn
from AINET.train_util import shift9pos
from models.maniqa import TEABlock, TABlock, ResBlockGroup, DynamicDWConv, Block

def safe_update_spixl_map(spixl_map_idx_in, assig_map_in):
    # spixl_map_idx_in: [B, 9, h_s, w_s]
    # assig_map_in: [B, 9, h, w]
    
    b, c, h, w = assig_map_in.shape
    _, _, id_h, id_w = spixl_map_idx_in.shape

    if (id_h == h) and (id_w == w):
        spixl_map_idx = spixl_map_idx_in
    else:
        spixl_map_idx = F.interpolate(spixl_map_idx_in, size=(h,w), mode='nearest')
        
    # Use argmax to find the channel with max probability
    # This guarantees a single index per pixel
    max_indices = torch.argmax(assig_map_in, dim=1, keepdim=True) # [B, 1, h, w]
    
    # Gather the superpixel index from the winning channel
    # spixl_map_idx is [B, 9, h, w]
    # max_indices is [B, 1, h, w]
    new_spixl_map = torch.gather(spixl_map_idx, 1, max_indices) # [B, 1, h, w]
    
    return new_spixl_map.long() # Return as long for indexing


class SimilarityModuleAINET(nn.Module):
    def __init__(self, feature_channels):
        super(SimilarityModuleAINET, self).__init__()
        self.pixel_conv = nn.Conv2d(3, feature_channels, kernel_size=1)

        # Reduce Conv layer to combine avg and std features
        self.reduce_conv = nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=1)

        # Similarity weight block
        self.similarity_weight_block = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.BatchNorm2d(1),  # Normalize along channels
            nn.ReLU()
        )

    def forward(self, x, labels, num_spixels):
        """
        x: [B, 3, H, W] - Original image
        labels: [B, 1, H, W] - Superpixel labels (0 to num_spixels-1)
        num_spixels: int - Number of superpixels
        """
        B, C, H, W = x.shape
        
        # 1x1 pixel-level conv on the whole image
        features = self.pixel_conv(x) # [B, 256, H, W]
        
        # Flatten features and labels
        features = features.view(B, -1, H * W) # [B, 256, N]
        labels = labels.view(B, H * W).long() # [B, N]
        
        # Initialize outputs
        # We need to aggregate features by label
        # Since labels vary per batch, we loop over batch (or use advanced scatter)
        
        avg_pool_features = torch.zeros(B, features.size(1), num_spixels, device=x.device)
        std_pool_features = torch.zeros(B, features.size(1), num_spixels, device=x.device)
        
        for b in range(B):
            curr_feat = features[b] # [256, N]
            curr_lab = labels[b] # [N]
            
            # One-hot encoding is too big (N x num_spixels).
            # Use index_add_
            
            # Count per label
            ones = torch.ones_like(curr_lab, dtype=torch.float)
            count = torch.zeros(num_spixels, device=x.device)
            count.index_add_(0, curr_lab, ones)
            count = count.clamp(min=1.0)
            
            # Sum features
            sum_feat = torch.zeros(features.size(1), num_spixels, device=x.device)
            sum_feat.index_add_(1, curr_lab, curr_feat)
            
            # Mean
            mean = sum_feat / count.unsqueeze(0)
            avg_pool_features[b] = mean
            
            # Sum squares for std
            sum_sq_feat = torch.zeros(features.size(1), num_spixels, device=x.device)
            sum_sq_feat.index_add_(1, curr_lab, curr_feat ** 2)
            
            # Variance = E[x^2] - (E[x])^2
            mean_sq = sum_sq_feat / count.unsqueeze(0)
            var = mean_sq - mean ** 2
            std = torch.sqrt(var.clamp(min=1e-6))
            std_pool_features[b] = std

        # avg_pool_features: [B, 256, num_spixels]
        # std_pool_features: [B, 256, num_spixels]
        
        # Reshape to match original SimilarityModule expectations: [B, 256, 1, num_spixels]
        avg_pool_features = avg_pool_features.unsqueeze(2)
        std_pool_features = std_pool_features.unsqueeze(2)
        
        # Combine avg and std features
        combined_features = torch.cat([avg_pool_features, std_pool_features], dim=1)  # Shape: [batch_size, 512, 1, num_spixels]

        # Reduce features to original channel size
        reduced_features = self.reduce_conv(combined_features)  # Shape: [batch_size, 256, 1, num_spixels]

        # Normalize reduced features
        reduced_features_normalized = F.normalize(reduced_features, p=2, dim=1)  # Shape: [batch_size, 256, 1, num_spixels]
        reduced_features_normalized = reduced_features_normalized.squeeze(2)  # Shape: [batch_size, 256, num_spixels]

        # Compute similarity matrix
        similarity_matrix_normalized = torch.bmm(
            reduced_features_normalized.permute(0, 2, 1), reduced_features_normalized
        )  # Shape: [batch_size, num_spixels, num_spixels]
        
        similarity_matrix_reshaped = similarity_matrix_normalized.unsqueeze(1)  # Shape: [batch_size, 1, num_spixels, num_spixels]

        # Compute image weight features
        image_weight_features = self.similarity_weight_block(similarity_matrix_reshaped)  # Shape: [batch_size, 1, num_spixels, num_spixels]
        
        return image_weight_features

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

class MANIQA_AINET(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        # AINET Initialization
        self.ainet = SpixelNet1l_bn(dataset='BDS500', Train=False)
        # Load AINET weights
        ainet_checkpoint = torch.load('AINET/model_best', map_location='cpu')
        # Check if state_dict is inside 'state_dict' key or direct
        if 'state_dict' in ainet_checkpoint:
            self.ainet.load_state_dict(ainet_checkpoint['state_dict'])
        else:
            self.ainet.load_state_dict(ainet_checkpoint)
        
        # Freeze AINET
        for param in self.ainet.parameters():
            param.requires_grad = False
        self.ainet.eval()
        
        # AINET grid params
        self.downsize = 16
        self.n_spixl_h = int(np.floor(img_size / self.downsize))
        self.n_spixl_w = int(np.floor(img_size / self.downsize))
        self.num_spixels = self.n_spixl_h * self.n_spixl_w
        
        # Precompute spixel grid indices
        spix_values = np.int32(np.arange(0, self.num_spixels).reshape((self.n_spixl_h, self.n_spixl_w)))
        spix_idx_tensor = shift9pos(spix_values)
        self.register_buffer('spix_idx_tensor', torch.from_numpy(spix_idx_tensor).float())

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
        self.simi_matrix = SimilarityModuleAINET(self.feature_channel)
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
    
    def forward(self, x, x_texture):
        # AINET Forward Pass
        with torch.no_grad():
            prob0, _ = self.ainet(x)
            # Prepare spixel grid for batch
            B, _, H, W = x.shape
            spix_idx = self.spix_idx_tensor.repeat(B, 1, 1, 1)
            
            labels = safe_update_spixl_map(spix_idx, prob0) # [B, 1, H, W]

        # slic query
        ######################################################################
        # Pass image and labels to SimilarityModuleAINET
        x_slic = self.simi_matrix(x, labels, self.num_spixels) # [B, 1, num_spixels, num_spixels]
        
        x_slic = self.slic_conv(x_slic) # torch.Size([B, 3072, num_spixels, num_spixels])
        x_slic = F.interpolate(x_slic, 
                               size=(self.input_size, self.input_size), 
                               mode='bilinear', align_corners=False) # torch.Size([B, 3072, 28, 28])

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
        x_res1 = self.dyd1(x_res1)

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
        x_res2 = self.dyd2(x_res2)

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

from einops import rearrange
