import os
import torch
import numpy as np
import cv2
import argparse
import glob
import sys
from torchvision import transforms

# Add workspace root to sys.path
sys.path.append(os.getcwd())

from models.maniqa import MANIQA as MANIQA_SLIC
from models.maniqa_spim import MANIQA as MANIQA_SPIM, CNNRIM
from models.maniqa_ainet import MANIQA_AINET
from config import Config
from utils.process import Normalize, five_point_crop
from utils.slic.slic_func import SLIC

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def gaussian_filter(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

def structure_texture_decomposition(image, sigma=2.0):
    structure_image = gaussian_filter(image, sigma)
    texture_image = image - structure_image
    return structure_image, texture_image

def get_slic_patches_global(img_path, slic_dir, slic_args):
    # Read image
    d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if d_img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Check if .npy exists
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    npy_path = os.path.join(slic_dir, f'{base_name}_seg.npy')
    
    if os.path.exists(npy_path):
        # Load from npy
        slic_res = np.load(npy_path, allow_pickle=True).item()
        superpixel_data = slic_res['superpixel_data']
        
        # Convert to tensor (logic from slic_func.py)
        image_n_nodes = slic_args['image_n_nodes']
        patch_n_nodes = slic_args['patch_n_nodes']
        
        superpixel_tensor = np.stack([superpixel_data[sp_id] for sp_id in range(image_n_nodes) if sp_id in superpixel_data], axis=0)
        if superpixel_tensor.shape != (image_n_nodes, patch_n_nodes, 3):
             superpixel_tensor = np.resize(superpixel_tensor, (image_n_nodes, patch_n_nodes, 3))
        
        d_img_slic = superpixel_tensor.astype('float32') / 255
    else:
        # Compute if missing
        d_img_slic_input = cv2.resize(d_img, (500, 500), interpolation=cv2.INTER_CUBIC)
        d_img_slic_input_uint8 = np.array(d_img_slic_input).astype('uint8')
        slic_class = SLIC(img=d_img_slic_input_uint8, args=slic_args)
        tensor_from_compute = slic_class.slic_function(save_path='', visualize_path='')
        d_img_slic = tensor_from_compute.astype('float32') / 255
        
    return d_img, d_img_slic

def get_spim_patches(img, spix_logits, n_spix=100, n_pixels=600):
    # img: (B, 3, H, W)
    # spix_logits: (B, n_spix, H, W)
    B, C, H, W = img.shape
    spix_labels = spix_logits.argmax(dim=1) # (B, H, W)
    
    patches = torch.zeros(B, n_spix, n_pixels, C, device=img.device)
    
    for b in range(B):
        flat_img = img[b].view(C, -1).permute(1, 0) # (H*W, C)
        flat_labels = spix_labels[b].view(-1) # (H*W)
        
        for k in range(n_spix):
            mask = (flat_labels == k)
            pixels = flat_img[mask]
            num = pixels.shape[0]
            if num > 0:
                if num >= n_pixels:
                    # Sample
                    idx = torch.randperm(num, device=img.device)[:n_pixels]
                    patches[b, k, :, :] = pixels[idx]
                else:
                    # Pad (repeat)
                    repeat_times = n_pixels // num + 1
                    extended = pixels.repeat(repeat_times, 1)
                    patches[b, k, :, :] = extended[:n_pixels]
            # else: remains zeros
            
    return patches

def preprocess_common(d_img, normalize):
    # 1. ViT Input
    d_img_vit = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
    d_img_vit = cv2.cvtColor(d_img_vit, cv2.COLOR_BGR2RGB)
    d_img_vit = np.array(d_img_vit).astype('float32') / 255
    d_img_vit = np.transpose(d_img_vit, (2, 0, 1)) # (3, 224, 224)
    if normalize:
        d_img_vit = normalize(d_img_vit)
    d_img_vit = torch.from_numpy(d_img_vit).unsqueeze(0).float()

    # 2. Texture Input
    d_img_gray = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
    _, d_img_texture = structure_texture_decomposition(d_img_gray, sigma=2.0)
    d_img_texture = d_img_texture.astype('float32') / 255
    d_img_texture = np.expand_dims(d_img_texture, axis=0)
    d_img_texture = np.repeat(d_img_texture, 3, axis=0) # (3, H, W)
    d_img_texture = np.transpose(d_img_texture, (1, 2, 0)) # (H, W, 3)
    d_img_texture = cv2.resize(d_img_texture, (224, 224), interpolation=cv2.INTER_CUBIC)
    d_img_texture = np.transpose(d_img_texture, (2, 0, 1)) # (3, 224, 224)
    d_img_texture = torch.from_numpy(d_img_texture).unsqueeze(0).float()
    
    return d_img_vit, d_img_texture

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input image folder")
    parser.add_argument("--ckpt_slic", type=str, required=True, help="Path to SLIC model checkpoint")
    parser.add_argument("--ckpt_spim", type=str, required=True, help="Path to SPIM model checkpoint")
    parser.add_argument("--ckpt_ainet", type=str, required=True, help="Path to AINET model checkpoint")
    parser.add_argument("--label_path", type=str, default="data/livec/livec_label.txt", help="Path to label file")
    parser.add_argument("--slic_dir", type=str, default="slic_livec", help="Path to slic npy folder")
    parser.add_argument("--gpu", type=int, default=2, help="GPU ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(20)

    # Load labels
    min_score = 0
    max_score = 100
    gt_scores = {}
    if os.path.exists(args.label_path):
        print(f"Loading labels from {args.label_path}")
        scores = []
        with open(args.label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    scores.append(float(parts[1]))
                    gt_scores[parts[0]] = float(parts[1])
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            print(f"Score range: [{min_score:.4f}, {max_score:.4f}]")
    else:
        print(f"Warning: Label file {args.label_path} not found. Using default range [0, 100].")

    # Config
    config = Config({
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,
        "crop_size": 224,
        "num_avg_val": 5
    })
    
    # SLIC Args
    slic_args = {
        'image_n_nodes': 140,
        'patch_n_nodes': 600,
        'region_size': 40,
        'ruler': 10.0,
        'iterate': 10
    }

    # --- Load Models ---
    print("Loading models...")
    
    # 1. SLIC Model
    net_slic = MANIQA_SLIC(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)
    ckpt_slic = torch.load(args.ckpt_slic, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in ckpt_slic.items()}
    net_slic.load_state_dict(new_state_dict)
    net_slic = net_slic.to(device).eval()
    
    # 2. SPIM Model
    net_spim = MANIQA_SPIM(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)
    ckpt_spim = torch.load(args.ckpt_spim, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in ckpt_spim.items()}
    net_spim.load_state_dict(new_state_dict)
    net_spim = net_spim.to(device).eval()
    
    model_sps = CNNRIM(n_spix=100).to(device) # SPIM Generator

    # 3. AINET Model
    net_ainet = MANIQA_AINET(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)
    ckpt_ainet = torch.load(args.ckpt_ainet, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in ckpt_ainet.items()}
    net_ainet.load_state_dict(new_state_dict)
    net_ainet = net_ainet.to(device).eval()

    normalize = Normalize(0.5, 0.5)

    # Get images
    image_paths = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    image_paths = sorted(list(set(image_paths)))
    
    print(f"Found {len(image_paths)} images.")
    
    # Header
    print(f"{'Image':<40} | {'GT':<8} | {'SLIC':<8} | {'SPIM':<8} | {'AINET':<8} | {'D_SLIC':<8} | {'D_SPIM':<8} | {'D_AINET':<8}")
    print("-" * 120)

    for img_path in image_paths:
        try:
            img_name = os.path.basename(img_path)
            
            # --- Preprocessing ---
            # Common
            d_img, d_img_slic_patches_np = get_slic_patches_global(img_path, args.slic_dir, slic_args)
            x_d, x_t = preprocess_common(d_img, normalize)
            x_d, x_t = x_d.to(device), x_t.to(device)
            
            # SLIC Patches
            x_s_slic = torch.from_numpy(d_img_slic_patches_np).unsqueeze(0).float().to(device)

            # --- Inference ---
            
            # 1. SLIC
            score_slic = 0
            with torch.no_grad():
                for i in range(config.num_avg_val):
                    x_d_crop = five_point_crop(i, d_img=x_d, config=config)
                    s = net_slic(x_d_crop, x_t, x_s_slic)
                    score_slic += s.item()
            score_slic /= config.num_avg_val
            score_slic_orig = score_slic * (max_score - min_score) + min_score

            # 2. SPIM
            score_spim = 0
            with torch.no_grad():
                # Optimize SPIM on full image (x_d is 224x224)
                with torch.enable_grad():
                    spix_logits = model_sps.optimize(x_d[0], n_iter=5, lr=1e-2, lam=2, alpha=2, beta=2, device=device)
                
                for i in range(config.num_avg_val):
                    x_d_crop = five_point_crop(i, d_img=x_d, config=config)
                    spix_logits_crop = five_point_crop(i, d_img=spix_logits, config=config)
                    x_s_spim = get_spim_patches(x_d_crop, spix_logits_crop, n_spix=100, n_pixels=600)
                    x_s_spim = x_s_spim.to(device)
                    
                    s = net_spim(x_d_crop, x_t, x_s_spim)
                    score_spim += s.item()
            score_spim /= config.num_avg_val
            score_spim_orig = score_spim * (max_score - min_score) + min_score

            # 3. AINET
            score_ainet = 0
            with torch.no_grad():
                for i in range(config.num_avg_val):
                    x_d_crop = five_point_crop(i, d_img=x_d, config=config)
                    s = net_ainet(x_d_crop, x_t)
                    score_ainet += s.item()
            score_ainet /= config.num_avg_val
            score_ainet_orig = score_ainet * (max_score - min_score) + min_score

            # --- Results ---
            gt_val = gt_scores.get(img_name, None)
            
            if gt_val is not None:
                gt_str = f"{gt_val:.4f}"
                diff_slic = score_slic_orig - gt_val
                diff_spim = score_spim_orig - gt_val
                diff_ainet = score_ainet_orig - gt_val
                
                d_slic_str = f"{diff_slic:.4f}"
                d_spim_str = f"{diff_spim:.4f}"
                d_ainet_str = f"{diff_ainet:.4f}"
            else:
                gt_str = "N/A"
                d_slic_str = "N/A"
                d_spim_str = "N/A"
                d_ainet_str = "N/A"

            print(f"{img_name:<40} | {gt_str:<8} | {score_slic_orig:<8.4f} | {score_spim_orig:<8.4f} | {score_ainet_orig:<8.4f} | {d_slic_str:<8} | {d_spim_str:<8} | {d_ainet_str:<8}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main()

# python test_score.py \
# --image_dir "DSN-IQA/vis_pics_spim/source"\
#  --ckpt_slic "all_save_dataset/output_livec/models/livec/epoch37.pt" \
# --ckpt_spim "all_save_seg/output_livec_spim/models/livec/epoch31.pt" \
# --ckpt_ainet "all_save_seg/output_livec_ainet/models/livec/epoch17.pt" \
# --label_path "data/livec/livec_label.txt" --slic_dir "slic_livec"