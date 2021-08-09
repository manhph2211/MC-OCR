import torch.nn as nn
from backbone import resnet101
from head import DeeplabHead
from torchsummary import summary


class DeeplabV3(nn.Module):
    def __init__(self, pretrained_backbone=True, num_classes=1):
        super(DeeplabV3, self).__init__()
        self.backbone = self.get_backbone(pretrained=pretrained_backbone)
        self.head = DeeplabHead(in_channels=2048, num_classes=num_classes)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, input_tensor):
        output_tensor = self.backbone(input_tensor)
        output_tensor = self.head(output_tensor)
        output_tensor = self.upsample(output_tensor)

        return output_tensor

    @staticmethod
    def get_backbone(pretrained=False):
        backbone = resnet101(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
        backbone = nn.Sequential(*(list(backbone.children())[: -2]))

        return backbone


if __name__ == '__main__':
    model = DeeplabV3()

    summary(model, (3, 416, 416))
    pass
