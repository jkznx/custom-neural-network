"""Custom small residual U-shaped network; no pretrained layers."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(3, outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(3, outgoing))
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1, bias=False)

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class LaneNet(nn.Module):
    def __init__(self, base=6):
        super().__init__()
        if base < 3 or base % 3:
            raise ValueError('base must be positive and divisible by three')
        self.enc1, self.enc2, self.enc3 = Block(3, base), Block(base, 2*base), Block(2*base, 4*base)
        self.bridge = Block(4*base, 8*base)
        self.dec3, self.dec2, self.dec1 = Block(12*base, 4*base), Block(6*base, 2*base), Block(3*base, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.apply(self.initialize)

    @staticmethod
    def initialize(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        if min(x.shape[-2:]) < 8:
            raise ValueError('height and width must be at least 8')
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        x = self.bridge(F.max_pool2d(e3, 2))
        for skip, decoder in [(e3, self.dec3), (e2, self.dec2), (e1, self.dec1)]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([x, skip], 1))
        return self.head(x)


if __name__ == '__main__':
    net = LaneNet().eval()
    with torch.inference_mode():
        print('Output:', tuple(net(torch.zeros(1, 3, 96, 160)).shape))
    print('Parameters:', sum(p.numel() for p in net.parameters()))
