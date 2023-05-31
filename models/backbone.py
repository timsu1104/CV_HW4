"""
Reference: https://github.com/yassouali/pytorch-segmentation/blob/master/models/deeplabv3_plus.py
"""

import torch
from torch import nn
from torch.nn import functional as F

from torchvision import models

class ResNet(nn.Module):
    def __init__(self, in_channels: int=3, output_stride: int=16, backbone: str='resnet101', pretrain: bool=True) -> None:
        super().__init__()
        assert in_channels == 3 and pretrain
        model = getattr(models, backbone)(weights=models.ResNet101_Weights.DEFAULT) # Pretrained ResNet on ImageNet
        
        self.layer0 = nn.Sequential(*list(model.children())[:4]) # Keep the first 4 layers
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        if output_stride == 16: stride_3, stride_4, dilation_3, dilation_4 = (2, 1, 1, 2)
        else: stride_3, stride_4, dilation_3, dilation_4 = (1, 1, 2, 4)
        
        if output_stride == 8: 
            for name, module in self.layer3.named_modules():
                if 'conv1' in name and (backbone == 'resnet34' or backbone == 'resnet18'):
                    module.dilation, module.padding, module.stride = (dilation_3,dilation_3), (dilation_3,dilation_3), (stride_3,stride_3)
                elif 'conv2' in name:
                    module.dilation, module.padding, module.stride = (dilation_3,dilation_3), (dilation_3,dilation_3), (stride_3,stride_3)
                elif 'downsample.0' in name:
                    module.stride = (stride_3, stride_3)

        for name, module in self.layer4.named_modules():
            if 'conv1' in name and (backbone == 'resnet34' or backbone == 'resnet18'):
                module.dilation, module.padding, module.stride = (dilation_4,dilation_4), (dilation_4,dilation_4), (stride_4,stride_4)
            elif 'conv2' in name:
                module.dilation, module.padding, module.stride = (dilation_4,dilation_4), (dilation_4,dilation_4), (stride_4,stride_4)
            elif 'downsample.0' in name:
                module.stride = (stride_4, stride_4)
                
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        low_level_feats = x1
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        endpoints = dict(
            output=x4,
            low_level_features=low_level_feats
        )
        
        return endpoints