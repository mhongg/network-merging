from torch import nn
from torchvision.models import resnet18, resnet34


def ResNet18(input_channel, output_size):
    model = resnet18(num_classes=output_size)
    model.conv1 = nn.Conv2d(
        input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    return model

def ResNet34(input_channel, output_size):
    model = resnet34(num_classes=output_size)
    model.conv1 = nn.Conv2d(
        input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    return model