import torch
from torch import nn
from torch.nn import functional as F

def aspp_branch(in_channels, out_channels, kernel_size, dilation) -> nn.Sequential:
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

class ASPP(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(ASPP, self).__init__()

        assert output_stride in [8, 16]
        dilations = [1, 12, 24, 36] if output_stride == 8 else [1, 6, 12, 18]
        kernel_sizes = [1, 3, 3, 3]
        
        self.aspps = nn.ModuleList([aspp_branch(in_channels, 256, kernel_size, dilation=dilation) for kernel_size, dilation in zip(kernel_sizes, dilations)])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        
        self.postprocess = nn.Sequential(
            nn.Conv2d(256*5, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # initialize weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        out_aspp = [aspp(x) for aspp in self.aspps]
        out_global = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        x = self.postprocess(torch.cat([*out_aspp, out_global], dim=1))
        
        return x