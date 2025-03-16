"""
This file use a virtual input help u to see the shape of each block.
"""

import torch.nn as nn

from model.Unet import UNetModel
from model.Clip_embedder import CLIPTextEmbedder
from sampler.ddpm import DDPMSampler
from model.HybridEncoder import HybridEncoder
from utils import *
from model.Latent_diffusion import LatentDiffusion

batch_size = 4  # 批量大小
channels = 3     # RGB 图像的通道数
height = 256   # 图像高度
width = 256     # 图像宽度
device = 'cpu'
save_dir = 'log'
n_steps = 1000
out_channels = 4

# 生成虚拟输入数据（随机值）
virtual_input = torch.randn(batch_size, channels, height, width)
time = torch.randint(0, 100, (batch_size,))
label = torch.randint(0, 10, (batch_size,))

class_name = label2text('cifar')
class_text = get_text_labels(label, class_name)

hybrid_encoder = HybridEncoder(10)
Clip_emb = CLIPTextEmbedder("openai/clip-vit-large-patch14/clip-vit-large-patch14", device='cpu')

latent = hybrid_encoder(virtual_input)
cond = Clip_emb(class_text)

print("The shape of latent space: " + str(latent.shape))
print("The shape of text embedding: " + str(cond.shape))

Unet = UNetModel(in_channels=4, out_channels=4, channels=160, attention_levels=[0, 1, 2], n_res_blocks=2,
                            channel_multipliers=[1, 2, 4, 4], n_heads=8, tf_layers=1, d_cond=768)
beta= torch.linspace(0.0001, 0.02, n_steps).to(device)
latent_dm = LatentDiffusion(Unet, hybrid_encoder, Clip_emb, 2, 1000, 0.0001, 0.2)
ddpm = DDPMSampler(latent_dm)

time_emb = Unet.time_step_embedding(time)
t_emb = Unet.time_embed(time_emb)
print("The shape of time embedding: " + str(time_emb.shape))
print("Time embedding after MLP: " + str(t_emb.shape))

# first stage
x_input_block = []
x = latent

for module in Unet.input_blocks:
    x = module(x, t_emb, cond)
    x_input_block.append(x)

print("After first stage: " + str(x.shape))

# Output half of the U-Net
x = Unet.middle_block(x, t_emb, cond)
print("After middle block: " + str(x.shape))

# Second stage
for module in Unet.output_blocks:
    x = torch.cat([x, x_input_block.pop()], dim=1)
    x = module(x, t_emb, cond)
print("After second stage: " + str(x.shape))

class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups..
    """
    return GroupNorm32(32, channels)

out = nn.Sequential(
            normalization(160),
            nn.SiLU(),
            nn.Conv2d(160, out_channels, 3, padding=1),
        )

res_1 = out(x)

res_2 = Unet.forward(latent, time, cond)

print(res_1.shape)
print(res_2.shape)
print(res_1 == res_2)
print("Good luck!")