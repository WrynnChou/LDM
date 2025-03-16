import torch.nn as nn

from utils import *

net = get_model("resnet50", 10, False)
path1 = 'log/checkpoint.pth.tar'
checkpoint1 = torch.load(path1, weights_only=True)
net.load_state_dict(checkpoint1['state_dict'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader, valid_loader = get_dataset('data', 'cifar', 32, False,
                                         4, True)
criterion = nn.CrossEntropyLoss()
net.eval()
correct, total, loss_ = 0., 0., 0.

for inputs, labels in valid_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    _, indices = torch.max(outputs, 1)
    correct += (indices == labels).sum()
    total += labels.size()[0]
    loss = criterion(outputs, labels)
    loss_ += loss.item()

print('Test loss: ' + str(loss_ / total) + ', Test acc: ' + str(correct / total * 100))

correct, total, loss_ = 0., 0., 0.

for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    _, indices = torch.max(outputs, 1)
    correct += (indices == labels).sum()
    total += labels.size()[0]
    loss = criterion(outputs, labels)
    loss_ += loss.item()
print('Train loss: ' + str(loss_ / total) + ', Train acc: ' + str(correct / total * 100))

print('Good luck!')


































