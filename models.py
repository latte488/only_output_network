import torch
from torch import nn
from torchvision.models import resnet

class OnlyOutputConvNet(nn.Module):

    def __init__(self, num_classes):
        super(OnlyOutputConvNet, self).__init__()
        self.bases = [resnet.resnet152(num_classes=1) for _ in range(num_classes)]
        self.bases = nn.ModuleList(self.bases)

    def forward(self, x):
        x = torch.stack([base(x) for base in self.bases])
        x = x.permute(2, 1, 0)
        x = torch.sigmoid(x[0])
        return x

def net(num_classes):
    return OnlyOutputConvNet(num_classes)

def loss(num_classes):
    def f(inputs, labels):
        onehot_labels = [torch.eye(num_classes)[label] for label in labels]
        onehot_labels = torch.stack(onehot_labels).to(labels.device)
        return nn.functional.mse_loss(inputs, onehot_labels)
    return f
