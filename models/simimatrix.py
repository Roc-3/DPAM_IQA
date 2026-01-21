
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityModule(nn.Module):
    def __init__(self, feature_channels):
        super(SimilarityModule, self).__init__()
        self.pixel_conv = nn.Conv2d(3, feature_channels, kernel_size=1)

        # Reduce Conv layer to combine avg and std features
        self.reduce_conv = nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=1)

        # Similarity weight block
        self.similarity_weight_block = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.BatchNorm2d(1),  # Normalize along channels
            nn.ReLU()
        )

    def forward(self, x_slic):
        # Convert to tensor and permute to NCHW format
        slic_res = x_slic.permute(0, 3, 2, 1)  # Shape: [batch_size, 3, patch_n_nodes, image_n_nodes]

        # 1x1 pixel-level conv
        slic_features = self.pixel_conv(slic_res)

        # Channel-wise avg and std pooling
        avg_pool_features = torch.mean(slic_features, dim=2, keepdim=True)
        std_pool_features = torch.std(slic_features, dim=2, keepdim=True) # Shape: [batch_size, 256, 1, image_n_nodes]

        # Combine avg and std features
        combined_features = torch.cat([avg_pool_features, std_pool_features], dim=1)  # Shape: [batch_size, 2*num_superpixels, 1, 300]

        # Reduce features to original channel size
        reduced_features = self.reduce_conv(combined_features)  # Shape: [batch_size, 256, 1, image_n_nodes]

        # Normalize reduced features
        reduced_features_normalized = F.normalize(reduced_features, p=2, dim=1)  # Shape: [batch_size, num_superpixels, 1, 300]
        reduced_features_normalized = reduced_features_normalized.squeeze(2)  # Shape: [batch_size, num_superpixels, 300]

        # Compute similarity matrix
        similarity_matrix_normalized = torch.bmm(
            reduced_features_normalized.permute(0, 2, 1), reduced_features_normalized
        )  # Shape: [batch_size, 300, 300]
        similarity_matrix_reshaped = similarity_matrix_normalized.unsqueeze(1)  # Shape: [batch_size, 1, image_n_nodes, image_n_nodes]

        # Compute image weight features
        image_weight_features = self.similarity_weight_block(similarity_matrix_reshaped)  # Shape: [batch_size, 1, 300, 300]

        # Multiply similarity matrix and weights, then reduce
        combined_features = torch.mul(similarity_matrix_reshaped, image_weight_features) # Shape: [batch_size, 1, 300, 300]

        return combined_features