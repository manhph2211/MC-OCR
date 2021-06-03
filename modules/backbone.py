import torch
import torch.nn as nn
import torchvision


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BottleNeck, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackBone(nn.Module):
    """
    Adapted from torchvison.model.resnet
    """

    def __init__(self, layers=(3, 4, 6, 3), block=BottleNeck, norm_layer=nn.BatchNorm2d):
        super(ResNetBackBone, self).__init__()

        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.inchannels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inchannels != channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inchannels, channels * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       self.norm_layer(channels * block.expansion))
        layers = [block(self.inchannels, channels, stride, downsample, self.norm_layer)]
        self.inchannels = channels * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inchannels, channels, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)
        self.channels.append(channels * block.expansion)
        self.layers.append(layer)

    def forward(self, x):
        """
        Returns a list of Convouts of each layer.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i > 0:
                outs.append(x)
        return tuple(outs)

    def init_backbone(self):
        """
        Initialize the backbone weights for training.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net_50 = torchvision.models.resnet50(pretrained=True, progress=True)
        net_50 = net_50.to(device)
        state_dict = net_50.state_dict()
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")

        keys = list(state_dict)
        for key in keys:
            if key.startswith("layer"):
                idx = int(key[5])
                new_key = "layer." + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # Note: use the strict= False is berry scary. Tripple check it
        self.load_state_dict(state_dict, strict=False)
