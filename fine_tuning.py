from utils import *
import torch.nn as nn
import torch
model = get_model('resnet50', 10, 'log/checkpoint.pth.tar')

for name, param in model.named_parameters():
    if name not in ["fc.weight", "fc.bias"]:
        param.requires_grad = False

new_featrue = torch.load('log/sample.pth.tar')

print('Have a nice day!')

