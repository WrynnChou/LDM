import shutil

import torch
import torch.utils.data as tdata
from torchvision import models
from torchvision import transforms, datasets


def get_dataset(dir, dataset_name, batch_size, shuffle=False, number_workers=4, pin_memery=True):

    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor()])
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform)
        eval_dataset = datasets.MNIST(dir, train=False, transform=transform)

    elif dataset_name == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                         transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    elif dataset_name == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.SVHN(dir, split='extra', download=True, transform=transform_train)
        eval_dataset = datasets.SVHN(dir, split='test', download=True, transform=transform_test)
    else:
        print('This datasets not support yet!')

    train_loader = tdata.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=number_workers, pin_memory=pin_memery)
    valid_loader = tdata.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=number_workers, pin_memory=pin_memery)
    return train_loader, valid_loader

def get_model(name="vgg16", pretrained=True):
    if name == "resnet18":
        if pretrained:
            model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        else:
            model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    elif name == "resnet50":
        model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    elif name == "densenet121":
        model = models.densenet121(weights=pretrained)
    elif name == "alexnet":
        model = models.alexnet(weights=pretrained)
    elif name == "vgg16":
        model = models.vgg16(weights=pretrained)
    elif name == "vgg19":
        model = models.vgg19(weights=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(weights=pretrained)
    elif name == "googlenet":
        model = models.googlenet(weights=pretrained)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

def label2text(name = 'cifar'):
    if name == 'cifar':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return class_names

def get_text_labels(labels, class_names):
    return [class_names[label] for label in labels]

if __name__ == "__main__":
    print('Good luck!')
    get_dataset('data', 'cifar', 32, False, 32, True)