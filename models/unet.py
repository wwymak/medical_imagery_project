import torch
from torch import nn


def conv_block(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(output_channels),
        nn.ReLU(inplace=True)
    )





class Unet(nn.Module):
    def __init__(self, s=32):
        super().__init__()
        self.start = unet_block(1, s, s)
        self.down1 = unet_block(s, s * 2, s * 2)
        self.down2 = unet_block(s * 2, s * 4, s * 4)
        self.bridge = unet_block(s * 4, s * 8, s * 4)
        self.up2 = unet_block(s * 8, s * 4, s * 2)
        self.up1 = unet_block(s * 4, s * 2, s)
        self.final = nn.Sequential(*conv(s * 2, s), nn.Conv3d(s, 1, 1))

    def unet_block(self, input_channels, middle_channels, output_channels):
        return nn.Sequential(conv_block(input_channels, middle_channels), conv_block(middle_channels, output_channels))

    def forward(self, x):
        r = [self.start(x)]
        r.append(self.down1(F.max_pool3d(r[-1], 2)))
        r.append(self.down2(F.max_pool3d(r[-1], 2)))
        x = F.interpolate(self.bridge(F.max_pool3d(r[-1], 2)), size=r[-1].shape[2:])
        x = F.interpolate(self.up2(torch.cat((x, r[-1]), dim=1)), size=r[-2].shape[2:])
        x = F.interpolate(self.up1(torch.cat((x, r[-2]), dim=1)), size=r[-3].shape[2:])
        x = self.final(torch.cat((x, r[-3]), dim=1))
        return x