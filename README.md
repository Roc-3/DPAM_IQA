# Dual Perception-Aware Blind Image Quality Assessment with Semantic-Distortion Integration and Dynamic Global-Local Refinement

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)


## 📋 Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Cross-Dataset Evaluation](#cross-dataset-evaluation)
  - [Testing on Single Images](#testing-on-single-images)
- [Model Checkpoints](#model-checkpoints)
- [Superpixel Segmentation Methods](#superpixel-segmentation-methods)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [Updates](#updates)

## 🎯 Introduction

This repository provides the **official PyTorch implementation** of the paper:

> **"Dual Perception-Aware Blind Image Quality Assessment with Semantic-Distortion Integration and Dynamic Global-Local Refinement"**


### Architecture Overview

![Architecture of the proposed network](architecture.png)

*Fig. The framework of the proposed method.*


## 🛠️ Installation

### Prerequisites

- *NVIDIA GPU with CUDA support (recommended: RTX 3090 with 24GB VRAM)*
- *Python 3.8+*
- *CUDA 11.8 (recommended)*
- *Anaconda or Miniconda*

### Step 1: Clone the Repository

```bash
git clone https://github.com/Roc-3/DPAM_IQA.git
cd DPAM_IQA
```

### Step 2: Create Conda Environment

```bash
conda create -n DPAM python=3.8 -y
conda activate DPAM
```

### Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Core dependencies include:**
- PyTorch 2.4.1 + CUDA 11.8
- torchvision
- numpy, scipy
- timm
- tensorboard
- einops

## 🧩 Dataset Preparation

### Supported Datasets

This project supports the following IQA datasets:

| Dataset | Images | Type | Download Link |
|---------|--------|------|---------------|
| **LIVE** | 29 reference + 779 distorted | Synthetic | [Link](https://live.ece.utexas.edu/research/quality/subjective.htm) |
| **CSIQ** | 30 reference + 866 distorted | Synthetic | [Link](http://vision.okstate.edu/csiq) |
| **TID2013** | 25 reference + 3000 distorted | Synthetic | [Link](http://www.ponomarenko.info/tid2013.htm) |
| **KADID-10k** | 81 reference + 10,125 distorted | Synthetic | [Link](http://database.mmsp-kn.de/kadid-10k-database.html) |
| **LIVE-C** | 1,162 images | Authentic | [Link](https://live.ece.utexas.edu/research/ChallengeDB/index.html) |
| **KonIQ-10k** | 10,073 images | Authentic | [Link](http://database.mmsp-kn.de/koniq-10k-database.html) |
| **SPAQ** | 11,125 images | Authentic | [Link](https://github.com/h4nwei/SPAQ) |

### Directory Structure

The project uses a two-level directory structure:

#### 1. Dataset Images (stored in `../all_dataset/`)

Download and organize the actual dataset images in the parent directory:

```
../all_dataset/
├── LIVEC/
│   └── Images/                    # 1,162 authentically distorted images
├── tid2013/
│   ├── distorted_images/          # 3,000 distorted images
├── CSIQ/
│   ├── dist_imgs/                 # Distorted images
├── KADID-10K/
│   └── images/                    # 10,125 images
├── KonIQ-10K/
│   └── 1024x768/                  # 10,073 images
├── SPAQ/
│   └── TestImage/                 # 11,125 images
└── ...                            # Other datasets
```

#### 2. Data Processing Scripts (in `data/`)

The `data/` folder in the project contains dataset-specific processing scripts and label files:

```
data/
├── livec/
│   ├── livec.py              # LIVEC dataset loader with SLIC processing
│   └── livec_label.txt       # Image paths and MOS scores
├── tid2013/
│   ├── tid2013.py            # TID2013 dataset loader with SLIC processing
│   ├── tid2013_label.txt     # Image paths and MOS scores
└── ...                       # Other dataset folders
```

 The data loaders (e.g., `livec.py`, `tid2013.py`) automatically read images from `../all_dataset/<dataset_name>/` and labels from `data/<dataset_name>/<dataset_name>_label.txt`.


**Note:** Each label file should be a text file with the following format:

```
<image_name> <quality_score>
```

**Example:**

```
158.bmp 79.416667
159.bmp 73.834437
```

## 🚀 Usage

### Training

#### Single Dataset Training

To train the model on a specific dataset (e.g., LIVEC):

```bash
python train_maniqa.py
```

**Key training features:**
- Automatically creates output directories (e.g., `all_save_dataset/output_livec/`)
- Generates superpixel segmentation files in `slic_<dataset>/` directories
- Saves checkpoints and logs during training
- Supports TensorBoard for training visualization

#### Training Configuration

Edit the training script or create a configuration file to customize like:
- Dataset selection
- Batch size
- etc.


#### Monitor Training Progress

Use TensorBoard to visualize training:

```bash
tensorboard --logdir=all_save_dataset/output_<dataset>/tensorboard
```

### Cross-Dataset Evaluation

Evaluate the model's generalization ability by training on one dataset and testing on another:

```bash
python cross_dataset.py
```

This script supports all cross-dataset combinations, such as:
- LIVE → CSIQ
- CSIQ → TID2013
- LIVE-C → TID2013
- etc.

Results will be saved in `all_save_cross_dataset/output_<source>2<target>/`

### Testing on Single Images

#### Predict Quality Scores

To predict quality scores for your own images:

```bash
python test_score.py \
  --image_dir /path/to/your/images \
  --ckpt_slic pretrained/checkpoint_slic.pth \
  --label_path /path/to/label_file.txt \
  --slic_dir ./slic_output
```

**Arguments:**
- `--image_dir`: Directory containing images to evaluate
- `--ckpt_slic`: Path to SLIC-based model checkpoint
- `--label_path`: (Optional) Path to ground truth labels for correlation computation
- `--slic_dir`: Directory to save/load superpixel segmentation results



## 💾 Model Checkpoints

Pre-trained model checkpoints will be available soon. They will include:

- ✅ Single-dataset models
- ✅ Cross-dataset models
- ✅ Single-distortion-type models
- 🙌 coming soon

**Download links:** Coming soon

## 🧩 Superpixel Segmentation Methods

We integrate multiple superpixel segmentation methods for comprehensive semantic analysis:

### 1. SLIC (Simple Linear Iterative Clustering)

**Default method** used in our framework.

**Reference:**
```
R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua and S. Süsstrunk, "SLIC Superpixels Compared to State-of-the-Art Superpixel Methods," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 11, pp. 2274-2282, Nov. 2012, doi: 10.1109/TPAMI.2012.120.
```

### 2. RIM (Regularized Information Maximization)

**Compared** deep learning-based superpixel segmentation.

**Code:** [https://github.com/DensoITLab/ss-with-RIM](https://github.com/DensoITLab/ss-with-RIM)

**Reference:**
```
Suzuki, Teppei. "Superpixel segmentation via convolutional neural networks with regularized information maximization." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.
```

### 3. AINET (Association Implantation Network)

**Compared** superpixel segmentation.

**Code:** [https://github.com/YanFangCS/AINET](https://github.com/YanFangCS/AINET)

**Reference:**
```
Y. Wang, Y. Wei, X. Qian, L. Zhu and Y. Yang, "AINet: Association Implantation for Superpixel Segmentation," 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, 2021, pp. 7058-7067, doi: 10.1109/ICCV48922.2021.00699. keywords: {Convolutional codes;Image segmentation;Computer vision;Convolution;Implants;Jitter;Benchmark testing;Segmentation;grouping and shape;Vision applications and systems},
```

### Impact Comparison

![Impact of segmentation methods](segmentation_comparison.png)

*Fig. Qualitative comparison of different superpixel segmentation methods.*

## 📊 Results

### Performance on Individual Datasets

![Result on individual datasets](result.png)

## 📖 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{
}
```

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Contact: [yuting_xiao@yeah.net]

## 🔄 Updates

- **2026-01-22**: Initial release

---

**Star ⭐ this repository if you find it helpful!**
