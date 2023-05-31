import torch
from torch import nn
from torch.nn import functional as F

from .backbone import ResNet
from .aspp import ASPP
from .decoder import Decoder

# DeepLab v3+
class DeepLab(nn.Module):
    def __init__(
        self, 
        num_classes, 
        in_channels=3, 
        backbone='resnet101', 
        pretrain=True,
        output_stride=16):
                
        super(DeepLab, self).__init__()
        assert 'resnet' in backbone
        self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrain=pretrain)
        low_level_channels = 256
 
        self.ASPP = ASPP(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)
 
    def forward(self, x):
        H, W = x.size(2), x.size(3)
        
        endpoints = self.backbone(x)
        x = endpoints["output"]
        low_level_features = endpoints["low_level_features"]
        
        x = self.ASPP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) # N, numClass, H, W
        return x