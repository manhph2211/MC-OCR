import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchsummary import summary


def conv3x3(in_channels, out_channels, stride=(1, 1), groups=1, dilation=(1, 1)):

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=dilation,
                     stride=stride, groups=groups, dilation=dilation, bias=False)


def conv1x1(in_channels, out_channels, stride=(1, 1)):

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                     stride=stride, bias=False)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=(1, 1),
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = conv1x1(in_channels=in_channels, out_channels=width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(in_channels=width, out_channels=width, stride=stride, groups=groups,
                             dilation=(dilation, dilation))
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(in_channels=width, out_channels=self.expansion * out_channels)
        self.bn3 = norm_layer(self.expansion * out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, input_tensor):
        identity = input_tensor

        output_tensor = self.relu(self.bn1(self.conv1(input_tensor)))
        output_tensor = self.relu(self.bn2(self.conv2(output_tensor)))
        output_tensor = self.bn3(self.conv3(output_tensor))

        if self.downsample is not None:
            identity = self.downsample(identity)

        output_tensor += identity
        output_tensor = self.relu(output_tensor)

        return output_tensor


class Resnet(nn.Module):
    def __init__(self,
                 layers,
                 num_classes=1000,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None
                 ):
        super(Resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group
        self.block = BottleNeck

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_planes, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = self.norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.layer1 = self.__make_layer(64, layers[0])
        self.layer2 = self.__make_layer(128, layers[1], stride=(2, 2), dilation=replace_stride_with_dilation[0])
        self.layer3 = self.__make_layer(256, layers[2], stride=(2, 2), dilation=replace_stride_with_dilation[1])
        self.layer4 = self.__make_layer(512, layers[3], stride=(2, 2), dilation=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        output_tensor = self.relu(self.bn1(self.conv1(input_tensor)))
        output_tensor = self.max_pool1(output_tensor)
        output_tensor = self.layer1(output_tensor)
        output_tensor = self.layer2(output_tensor)
        output_tensor = self.layer3(output_tensor)
        output_tensor = self.layer4(output_tensor)

        output_tensor = self.avgpool(output_tensor)
        output_tensor = torch.flatten(output_tensor, start_dim=1)
        output_tensor = self.fc(output_tensor)

        return output_tensor

    def __make_layer(self, planes, blocks, stride=(1, 1), dilation=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        block = BottleNeck

        if dilation:
            self.dilation *= stride[0]
            stride = (1, 1)

        if stride != 1 or self.in_planes != block.expansion * planes:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, block.expansion * planes, stride),
                norm_layer(block.expansion * planes)
            )

        layers = list()
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)


def resnet101(layers=(3, 4, 23, 3), pretrained=False, **kwargs):
    model = Resnet(layers=layers, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet101-63fe2227.pth')

        for key in list(model.state_dict().keys()):
            if key.endswith('num_batches_tracked'):
                continue
            model.state_dict()[key].data.copy_(state_dict[key].data)

        print('Load pretrained from: https://download.pytorch.org/models/resnet101-63fe2227.pth')

    return model


if __name__ == '__main__':
    resnet = resnet101(layers=[3, 4, 6, 3], pretrained=True, replace_stride_with_dilation=[False, True, True])
    pass
