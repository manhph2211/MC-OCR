import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        ]

        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        modules = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPPooling, self).__init__(*modules)

    def forward(self, input_tensor):
        size = input_tensor.shape[-2:]
        for module in self:
            input_tensor = module(input_tensor)

        output_tensor = F.interpolate(input_tensor, size=size, mode='bilinear', align_corners=True)
        return output_tensor


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = list()

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        ))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels=in_channels, out_channels=out_channels, dilation=rate))

        modules.append(ASPPPooling(in_channels=in_channels, out_channels=out_channels))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels=len(self.convs) * out_channels, out_channels=out_channels,
                      kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, input_tensor):
        output_tensor = list()
        for module in self.convs:
            output_tensor.append(module(input_tensor))

        output_tensor = torch.cat(output_tensor, dim=1)
        output_tensor = self.project(output_tensor)

        return output_tensor


class DeeplabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeeplabHead, self).__init__()

        self.head = nn.Sequential(
            ASPP(in_channels=in_channels, atrous_rates=[12, 24, 36]),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=(1, 1))
        )

    def forward(self, input_tensor):

        return self.head(input_tensor)


if __name__ == '__main__':
    head = DeeplabHead(in_channels=32, num_classes=1)

    print(summary(head, (32, 128, 128)))
    pass
