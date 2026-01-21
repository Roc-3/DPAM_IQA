import os
import torch
import numpy as np
import cv2
from utils.slic.slic_func import SLIC
from utils.process import RandCrop

class Kadid10k(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, normalize, keep_ratio):
        super(Kadid10k, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.normalize = normalize

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]
                if dis[1:3] in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        
        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

        # sythetic data setting
        self.crop_size = 224
        self.rand_crop = RandCrop(self.crop_size)

        # slic setting
        self.slic_args = {
            'image_n_nodes': 99,
            'patch_n_nodes': 600,
            'region_size': 45,
            'ruler': 10.0,
            'iterate': 10
        }

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
    
        # slic img
        d_img_slic = np.array(d_img).astype('uint8') # hwc
        # slic superpixel
        ############################################
        save_dir = 'slic_kadid10k'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{os.path.splitext(d_img_name)[0]}_seg.npy')

        slic_class = SLIC(img=d_img_slic, args=self.slic_args)
        d_img_slic = slic_class.slic_function(save_path=save_path, visualize_path='visual_path') 
        d_img_slic = d_img_slic.astype('float32') / 255 # (image_n_nodes, patch_n_nodes, 3)
        ############################################

        if self.transform:  # random flip
            d_img, flipped = self.transform(d_img)
        
        # for sythetic dataset
        if self.rand_crop:
            d_img = np.transpose(d_img, (2, 0, 1))
            d_img = self.rand_crop(d_img)
            d_img = np.transpose(d_img, (1, 2, 0))

        # vit img 
        d_img_vit = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img_vit = cv2.cvtColor(d_img_vit, cv2.COLOR_BGR2RGB)
        d_img_vit = np.array(d_img_vit).astype('float32') / 255
        d_img_vit = np.transpose(d_img_vit, (2, 0, 1))
        if self.normalize: # vit normalization
            d_img_vit = self.normalize(d_img_vit) # (3, 224, 224)

        # texture img 
        d_img_gray = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
        _, d_img_texture = self.structure_texture_decomposition(d_img_gray, sigma=2.0)
        d_img_texture = d_img_texture.astype('float32') / 255
        d_img_texture = np.expand_dims(d_img_texture, axis=0)
        d_img_texture = np.repeat(d_img_texture, 3, axis=0)# (3, 500, 500)
        d_img_texture = np.transpose(d_img_texture, (1, 2, 0))# (500, 500, 3)
        d_img_texture = cv2.resize(d_img_texture, (224, 224), interpolation=cv2.INTER_CUBIC)# (224, 224, 3)        
        d_img_texture = torch.tensor(d_img_texture, dtype=torch.float32)
        d_img_texture = np.transpose(d_img_texture, (2, 0, 1)) # (3, 500, 500)
        # visualize_and_save(d_img_vit, d_img_texture, d_img_name)        
        
        score = self.data_dict['score_list'][idx]
        score = torch.from_numpy(np.array(score)).type(torch.FloatTensor)
    
        sample = {
            'd_img_org': d_img_vit,
            'd_img_texture': d_img_texture,
            'd_img_slic': d_img_slic,
            'score': score
        }
        # print('sample d_img_org:', sample['d_img_org'].shape)  # (3, 224, 224)
        # print('sample d_img_texture:', sample['d_img_texture'].shape)  # (3, 224, 224)
    
        return sample
    
    def gaussian_filter(self, image, sigma):
        """
        对图像应用高斯滤波器
        """
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    def structure_texture_decomposition(self, image, sigma):
        """
        使用高斯滤波器进行结构-纹理分解
        :param image: 输入图像
        :param sigma: 高斯滤波器的标准差
        :return: 结构图像和纹理图像
        """
        # 计算结构图像（高斯模糊）
        structure_image = self.gaussian_filter(image, sigma)
        
        # 计算纹理图像（原图减去结构图像）
        texture_image = image - structure_image
        
        return structure_image, texture_image
    
def visualize_and_save(d_img_org, d_img_texture, img_name):
    """
    Visualize and save the original image and texture image side by side.

    Parameters:
    - d_img_org: The original image (normalized and transposed to (3, H, W))
    - d_img_texture: The texture image (normalized and transposed to (3, H, W))
    - img_name: Name of the image (used for saving the visualized output)
    """
    # Convert the tensors back to image format for visualization
    d_img_org_vis = (torch.tensor(d_img_org).permute(1, 2, 0) * 255).numpy().astype(np.uint8)  # Convert to (H, W, 3)
    d_img_texture_vis = (d_img_texture.permute(1, 2, 0) * 255).numpy().astype(np.uint8)  # Convert to (H, W, 3)

    # Determine the maximum height and total width for the combined image
    max_height = max(d_img_org_vis.shape[0], d_img_texture_vis.shape[0])
    total_width = d_img_org_vis.shape[1] + d_img_texture_vis.shape[1]

    # Create a blank canvas with the determined size
    combined_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Place the original image on the left
    combined_img[:d_img_org_vis.shape[0], :d_img_org_vis.shape[1], :] = d_img_org_vis

    # Place the texture image on the right
    combined_img[:d_img_texture_vis.shape[0], d_img_org_vis.shape[1]:, :] = d_img_texture_vis

    # Save the combined image
    save_path = os.path.join('./', f"visualized_{img_name}.png")
    cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))

    print(f"Visualized image saved at: {save_path}")