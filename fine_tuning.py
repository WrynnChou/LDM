import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import *


class Projector(nn.Module):
    def __init__(self, num_classes, path):
        super().__init__()

        resnet50 = models.resnet50(num_classes=num_classes)

        if path is not None:
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["state_dict"]
            msg = resnet50.load_state_dict(state_dict, strict=False)
            print(msg)

        # 只剩最后的全局平均池化层和全连接层
        resnet50_projector = nn.Sequential(*list(resnet50.children())[-2:])  # 保留卷积部分
        self.resnet50_projector = resnet50_projector  # ResNet50 特征提取器

    def forward(self, x):
        return self.resnet50_projector(x)  # ResNet50 提取特征

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        初始化数据集。
        :param data: 数据，可以是列表、NumPy 数组或张量。
        :param labels: 标签，可以是列表、NumPy 数组或张量。
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """返回数据集的大小。"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本。
        :param idx: 样本的索引。
        :return: 数据和标签。
        """

        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

if __name__ == "__main__":

    lr = 0.0001
    epochs = 50
    model = get_model('resnet50', 10, False)
    weights = torch.load('log/checkpoint.pth.tar')['state_dict']
    msg = model.load_state_dict(weights)
    print(msg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    new_featrue = torch.load('log/sample.pth.tar').to('cpu')
    labels = [9] * 100
    data = CustomDataset(new_featrue, labels)

    projector = Projector(10, 'log/checkpoint.pth.tar')
    projector.to(device)
    # 创建 DataLoader
    train_loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=4)

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(projector.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for _ in range(epochs):
        correct, total, loss_ = 0., 0., 0.
        for ind, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = projector(inputs)
            _, indices = torch.max(outputs, 1)
            correct += (indices == labels).sum()
            total += labels.size()[0]
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ += loss.item()
        print('Train loss: ' + str(loss_ / total) + ', Train acc: ' + str(correct / total * 100))

    # train_loader_real, valid_loader = get_dataset('data', 'cifar', 32, False,4, True)
    # model.eval()
    # correct, total, loss_ = 0., 0., 0.
    #
    # for inputs, labels in valid_loader:
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     outputs = model(inputs)
    #     _, indices = torch.max(outputs, 1)
    #     correct += (indices == labels).sum()
    #     total += labels.size()[0]
    #     loss = loss_func(outputs, labels)
    #     loss_ += loss.item()
    #
    # print('Test loss: ' + str(loss_ / total) + ', Test acc: ' + str(correct / total * 100))
print('Have a nice day!')

