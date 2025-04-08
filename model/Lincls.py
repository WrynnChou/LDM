import torch.nn as nn

# class Discriminator(nn.Module):
#     def __init__(self, layer_4_2, avgpool, fc):
#         super().__init__()
#         self.layer_4_2 = layer_4_2     # 倒数第二层
#         self.fc = fc               # 分类头
#         self.avgpool = avgpool
#
#     def forward(self, x):
#         x = self.layer_4_2(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)    # [B, 512]
#         x = self.fc(x)             # [B, 1000]
#         return x

class Discriminator(nn.Module):
    def __init__(self, avgpool, fc):
        super().__init__()
        self.fc = fc               # 分类头
        self.avgpool = avgpool

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)    # [B, 512]
        x = self.fc(x)             # [B, 1000]
        return x
