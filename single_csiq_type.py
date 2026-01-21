import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.process import RandCrop, ToTensor, Normalize, five_point_crop
from utils.process import split_dataset_kadid10k, split_dataset_koniq10k, split_dataset_livec, split_dataset_tid2013, split_dataset_spaq, split_dataset_csiq
from utils.process import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm

from models.maniqa import MANIQA
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        x_d = data['d_img_org'].cuda()
        x_t = data['d_img_texture'].cuda()
        x_s = data['d_img_slic'].cuda()
        labels = data['score']

        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()  
        pred_d = net(x_d, x_t, x_s)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                x_t = data['d_img_texture'].cuda()
                x_s = data['d_img_slic'].cuda()
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                x_d = five_point_crop(i, d_img=x_d, config=config)
                pred += net(x_d, x_t, x_s)

            pred /= config.num_avg_val
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        return np.mean(losses), rho_s, rho_p


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "dataset_name": "csiq", # kadid10k, koniq10k, livec, tid2013, spaq, csiq

        # csiq
        "csiq_path": "../all_dataset/CSIQ/dist_imgs/",
        "csiq_label": "data/CSIQ/type/type_1.txt",

        # optimization
        "batch_size":12, 
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 200,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 1, # if training koniq10k, num_avg_val is set to 1
        "num_workers": 8,
        
        # data
        "split_seed": 20,
        "train_keep_ratio": 1.0,
        "val_keep_ratio": 1.0,
        "crop_size": 224,
        "prob_aug": 0.7,

        # model
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
        
        # load & save checkpoint
        "ckpt_path": "./output_csiq_type1/models/",               # directory for saving checkpoint
        "log_path": "./output_csiq_type1/log/",
        "log_file": ".log",
        "tensorboard_path": "./output_csiq_type1/"
    })
    
    config.log_file = config.dataset_name + ".log"
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.dataset_name)

    config.ckpt_path = os.path.join(config.ckpt_path, config.dataset_name)

    config.log_path = os.path.join(config.log_path, config.dataset_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    
    if not os.path.exists(config.tensorboard_path):
        os.makedirs(config.tensorboard_path)

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    if config.dataset_name == 'csiq':
        from data.CSIQ.csiq import CSIQ
        train_name, val_name = split_dataset_csiq(
            txt_file_name=config.csiq_label,
            split_seed=config.split_seed
        )
        dis_train_path = config.csiq_path
        dis_val_path = config.csiq_path
        label_train_path = config.csiq_label
        label_val_path = config.csiq_label
        Dataset = CSIQ
    else:
        pass
    
    # data load
    train_dataset = Dataset(
        dis_path=dis_train_path,
        txt_file_name=label_train_path,
        list_name=train_name,
        transform=transforms.Compose([
            # RandCrop(patch_size=config.crop_size),
            RandHorizontalFlip(prob_aug=config.prob_aug),
        ]),
        normalize=Normalize(0.5, 0.5),  # vit
        keep_ratio=config.train_keep_ratio
    )
    val_dataset = Dataset(
        dis_path=dis_val_path,
        txt_file_name=label_val_path,
        list_name=val_name,
        # transform=transforms.Compose([
        #     RandCrop(patch_size=config.crop_size), 
        # ]),
        transform = None,
        normalize=Normalize(0.5, 0.5),  # vit
        keep_ratio=config.val_keep_ratio
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    # load the data
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, drop_last=True, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, drop_last=True, shuffle=False)

    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    net = nn.DataParallel(net)
    net = net.cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = 0
    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalars("Metrics/Loss", {"Train": loss_val}, epoch)
        writer.add_scalars("Metrics/SRCC", {"Train": rho_s}, epoch)
        writer.add_scalars("Metrics/PLCC", {"Train": rho_p}, epoch)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p = eval_epoch(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')

            writer.add_scalars("Metrics/Loss", {"Test": loss}, epoch)
            writer.add_scalars("Metrics/SRCC", {"Test": rho_s}, epoch)
            writer.add_scalars("Metrics/PLCC", {"Test": rho_p}, epoch)

            if rho_s + rho_p > main_score:
                main_score = rho_s + rho_p
                best_srocc = rho_s
                best_plcc = rho_p

                logging.info('======================================================================================')
                logging.info('============================== best main score is {} ================================='.format(main_score))
                logging.info('======================================================================================')

                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.ckpt_path, model_name)
                torch.save(net.module.state_dict(), model_save_path)
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))