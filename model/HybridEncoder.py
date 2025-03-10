import torchvision.models as models
import torch.nn as nn
import torch
from Unet import UNetModel
from clip_embedder import CLIPTextEmbedder
from utils import *
# 加载预训练的 ResNet50
resnet50 = models.resnet50(pretrained=True)

# 移除最后的全局平均池化层和全连接层
resnet50_features = nn.Sequential(*list(resnet50.children())[:-2])  # 保留卷积部分

class LightweightEncoder(nn.Module):
    def __init__(self, in_channels=2048, out_channels=4):
        super().__init__()
        self.adaptor = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),  # 降低通道数
            nn.ReLU(),
            # nn.Upsample(scale_factor=2),  # 上采样到16x16
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # nn.Upsample(scale_factor=2),  # 上采样到32x32
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(512, out_channels, kernel_size=1)  # 输出通道数4
        )

    def forward(self, x):
        return self.adaptor(x)

class HybridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50_features = resnet50_features  # ResNet50 特征提取器
        self.lightweight_encoder = LightweightEncoder()  # 轻量级编码器

    def forward(self, x):
        features = self.resnet50_features(x)  # ResNet50 提取特征
        latent = self.lightweight_encoder(features)  # 映射到潜在空间
        return latent

class StableDiffusionWithHybridEncoder(nn.Module):
    def __init__(self, hybrid_encoder, diffusion_model):
        super().__init__()
        self.hybrid_encoder = hybrid_encoder  # 混合编码器
        self.diffusion_model = diffusion_model  # 扩散模型

    def forward(self, x, t):
        latent = self.hybrid_encoder(x)  # 编码到潜在空间
        noise_pred = self.diffusion_model(latent, t)  # 扩散模型预测噪声
        return noise_pred

batch_size = 4  # 批量大小
channels = 3     # RGB 图像的通道数
height = 256     # 图像高度
width = 256      # 图像宽度

# 生成虚拟输入数据（随机值）
virtual_input = torch.randn(batch_size, channels, height, width)
virtual_input2 = torch.randn(batch_size, channels, 224, 224)
time = torch.randint(0, 100, (batch_size,))
label = torch.randint(0, 10, (batch_size,))
cond = torch.randn(batch_size, 77, 768) # batch size * n_cond * d_cond
class_name = label2text('cifar')
class_text = get_text_labels(label, class_name)

hybrid_encoder = HybridEncoder()
latent = hybrid_encoder(virtual_input)
Clip_emb = CLIPTextEmbedder(device='cpu')
cond = Clip_emb(class_text)

diffusion_model = UNetModel(in_channels=4,
                               out_channels=4,
                               channels=320,
                               attention_levels=[0, 1, 2],
                               n_res_blocks=2,
                               channel_multipliers=[1, 2, 4, 4],
                               n_heads=8,
                               tf_layers=1,
                               d_cond=768)


time_emb = diffusion_model.time_step_embedding(time)
t_emb = diffusion_model.time_embed(time_emb)
x_input_block = []
x = latent

for module in diffusion_model.input_blocks:
    x = module(x, t_emb, cond)
    x_input_block.append(x)
x = diffusion_model.middle_block(x, t_emb, cond)
# Output half of the U-Net
for module in diffusion_model.output_blocks:
    x = torch.cat([x, x_input_block.pop()], dim=1)
    x = module(x, t_emb, cond)

import torch.optim as optim

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.Adam(
    list(hybrid_encoder.parameters()) + list(diffusion_model.parameters()),
    lr=1e-4
)











