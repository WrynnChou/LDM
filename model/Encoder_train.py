
import argparse
import datetime
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils import tensorboard as tb

from utils import get_model, get_dataset, save_checkpoint

parser = argparse.ArgumentParser(description="PyTorch Encoder Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
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
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
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
parser.add_argument(
    "--schedule",
    default=[30, 60, 80],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by a ratio)",
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
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument(
    "--pretrained", default="", type=str, help="path to moco pretrained checkpoint"
)
parser.add_argument(
    "--feature_dir", default="", type=str,
    help="out feature dir path"
)
parser.add_argument(
    "--num_classes", default=10, type=int, help= "Number of classes to be classification."
)
class Log():
    @classmethod
    def Log(cls, name, path):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s-8s:%(message)s')

        file_handler = logging.FileHandler(os.path.join(path, 'main.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        scream_handler = logging.StreamHandler()
        scream_handler.setFormatter(formatter)
        scream_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(scream_handler)

        return logger


class Train_model():
    def __init__(self, log, train_loader, valid_loader, net, device, criterion, max_epoch,
                 train_interval, valid_interval):
        self.logger = log
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.device = device
        self.net = net
        self.criterion = criterion
        self.max_epoch = max_epoch
        self.train_interval = train_interval
        self.valid_interval = valid_interval
        # self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    def train(self, epoch):
        self.net.train()
        if epoch == 0: lr = 0.01
        if epoch > 0: lr = 0.1
        if epoch > 60: lr = 0.01
        if epoch > 120: lr = 0.001
        if epoch > 150: lr = 0.0008
        if epoch > 170: lr = 0.0004
        if epoch > 190: lr = 0.0002
        optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        correct, total, loss_ = 0., 0., 0.
        for ind, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            _, indices = torch.max(outputs, 1)
            correct += (indices == labels).sum()
            total += labels.size()[0]
            loss = self.criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ += loss.item()
            writer.add_scalar('train_loss', loss_, epoch * len(train_loader) + ind + 1)
            writer.add_scalar('train_acc', correct / total, epoch * len(train_loader) + ind + 1)
            if (ind + 1) % self.train_interval == 0 or ind == len(self.train_loader) - 1:
                self.logger.info(
                    f'**Train**Epoch--{epoch + 1}--Iter--{ind + 1}--Loss_mean={loss_ / (ind + 1):<05.2f},Acc={correct / total * 100:<05.2f}%')

    def valid(self, epoch):
        self.net.eval()
        correct, total, loss_ = 0., 0., 0.
        with torch.no_grad():
            for ind, data in enumerate(self.valid_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, indices = torch.max(outputs, 1)
                correct += (indices == labels).sum()
                total += labels.size()[0]
                loss = self.criterion(outputs, labels)
                loss_ += loss.item()
                writer.add_scalar('val_loss', loss_, epoch * len(valid_loader) + ind + 1)
                writer.add_scalar('val_acc', correct / total, epoch * len(valid_loader) + ind + 1)

            self.logger.info(
                f'**Validation**Epoch--{epoch + 1}--Loss_mean={loss_ / (ind + 1):<05.2f},Acc={correct / total * 100:<05.2f}%')

    def process(self):
        start = time.time()
        for epoch in range(self.max_epoch):
            self.train(epoch)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": self.net.state_dict(),
                },
                is_best=False,
                filename="checkpoint.pth.tar",
            )
            if (epoch + 1) % self.valid_interval == 0 or epoch == self.max_epoch - 1:
                self.valid(epoch)
        end = time.time()
        writer.close()
        self.logger.info(f'**Total time: {(end - start) / 60} mins.')


if __name__ == '__main__':
    args = parser.parse_args()

    tensorboard_path = './tensorboard/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.mkdir(tensorboard_path)
    logger = Log.Log('resnet50', tensorboard_path)
    writer = tb.SummaryWriter(tensorboard_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = get_model(args.arch)
    net = net.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    train_loader, valid_loader = get_dataset(args.data, 'cifar', args.batch_size, False,
                                             args.workers, True)
    model = Train_model(logger, train_loader, valid_loader, net, device, criterion, args.epochs, 10,
                        1)
    model.process()

    checkpoint = torch.load(args.pretrained, map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint["state_dict"]
    msg = model.load_state_dict(state_dict, strict=False)