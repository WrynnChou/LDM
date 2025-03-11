import torch.nn as nn

from utils import *

# 加载预训练的 ResNet50
resnet50 = models.resnet50(pretrained=True)
for param in resnet50.parameters():
    param.requires_grad = False
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





