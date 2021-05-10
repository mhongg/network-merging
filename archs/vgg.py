from torch import nn
from torchvision.models import vgg11, vgg19


def VGG11(input_channel, output_size):
    model = vgg11(num_classes=output_size)
    model.features[0] = nn.Conv2d(
        input_channel, 64, kernel_size=3, stride=1, padding=1
    )
    return model

def VGG19(input_channel, output_size):
    model = vgg19(num_classes=output_size)
    model.features[0] = nn.Conv2d(
        input_channel, 64, kernel_size=3, stride=1, padding=1
    )
    return model