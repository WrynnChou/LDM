import argparse
import os

import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = 只显示错误信息
import torch.nn as nn
import torch.optim as optim

from latent_diffusion import LatentDiffusion
from model.HybridEncoder import HybridEncoder
from model.Unet import UNetModel
from model.clip_embedder import CLIPTextEmbedder
from sampler.ddpm import DDPMSampler
from utils import *

parser = argparse.ArgumentParser(description="PyTorch stable diffusion model Training")
parser.add_argument("--data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    help="model architecture: " + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128), this is the total "
         "batch size of all GPUs on the current node when "
         "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.)",
    dest="weight_decay",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--pretrained", default=None, type=str, help="path to autoencoder pretrained checkpoint"
)
parser.add_argument(
    "--num_classes", default=10, type=int, help= "Number of classes to be classification."
)
parser.add_argument(
    "--n_steps", default=1000, type=int, help= "T in diffusion model."
)
parser.add_argument(
    "--save_dir", default="log", type=str, help="Diffusion model checkpoint dir path"
)

if __name__ == '__main__':

    args = parser.parse_args()

    device =  'cuda:0' if torch.cuda.is_available else 'cpu'
    save_dir = args.save_dir
    n_steps = args.n_steps

    hybrid_encoder = HybridEncoder(args.num_classes, args.pretrained)
    hybrid_encoder.to(device)
    Clip_emb = CLIPTextEmbedder("openai/clip-vit-large-patch14/clip-vit-large-patch14", device=device)
    Unet = UNetModel(in_channels=4, out_channels=4, channels=320, attention_levels=[0, 1, 2], n_res_blocks=2,
                                channel_multipliers=[1, 2, 4, 4], n_heads=8, tf_layers=1, d_cond=768)
    beta= torch.linspace(0.0001, 0.02, n_steps).to(device)
    latent_dm = LatentDiffusion(Unet, hybrid_encoder, Clip_emb, 2, 1000, 0.0001,
                                0.2)
    latent_dm.to(device)
    ddpm = DDPMSampler(latent_dm)
    class_name = label2text('cifar')
    # 定义损失函数
    criterion = nn.MSELoss()

    for param in Clip_emb.parameters():
        param.requires_grad = False

    # 定义优化器
    optimizer = optim.Adam(
        list(hybrid_encoder.parameters()) + list(Unet.parameters()),
        lr=1e-4
    )
    train_loader, valid_loader = get_dataset('data', 'cifar', 8, False, 4, True)
    loss_record = []

    best_score, score, epochs, early_stop_time, early_stop_threshold= 1e10, 0, 200, 0, 40
    for epoch in range(epochs):
        loss_record= []
        for pic, labels in train_loader:
            pic= pic.to(device)
            labels = labels
            optimizer.zero_grad()
            cond = latent_dm.get_text_conditioning(get_text_labels(labels, class_name))
            cond = cond.to(device)
            loss= ddpm.loss(pic, cond)
            loss_record.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f'training epoch: {epoch}, mean loss: {torch.tensor(loss_record).mean()}')
        loss_record= []
        with torch.no_grad():
            for step, (pic, labels) in enumerate(valid_loader):
                pic= pic.to(device)
                cond = latent_dm.get_text_conditioning(get_text_labels(labels, class_name))
                cond = cond.to(device)
                loss= ddpm.loss(pic, cond)
                loss_record.append(loss.item())
        mean_loss= torch.tensor(loss_record).mean()
        # early stopping
        if mean_loss< best_score:
            early_stop_time= 0
            best_score= mean_loss
            torch.save(Unet, f'{save_dir}')
        else:
            early_stop_time= early_stop_time+ 1
        if early_stop_time> early_stop_threshold:
            break
        # output
        print(f'early_stop_time/early_stop_threshold: {early_stop_time}/{early_stop_threshold}, mean loss: {mean_loss}')

    torch.save(latent_dm.state_dict(), 'log/sdm_checkpoint.pth.tar')

