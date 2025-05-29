import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class FineTuning(nn.Module):
    def __init__(self, num_classes):
        super(FineTuning, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


class LinearProbing(nn.Module):
    def __init__(self, num_classes):
        super(LinearProbing, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False

        # Replace the classifier head
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


def load_cifar10_dataset():
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    return loader


def main():
    # 加载数据集
    dataset = load_cifar10_dataset()
    model = LinearProbing(num_classes=10)
    # model = FineTuning(num_classes=10)

    for images, labels in dataset:
        logits = model(images)

        # classification loss
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        # 输出损失
        print(loss)

if __name__ == "__main__":
    main()