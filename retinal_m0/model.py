import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256, 512), groups=8):
        super().__init__()
        self.widths = tuple(widths)
        self.encoders, self.ups, self.decoders = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        previous = 3
        for width in widths:
            self.encoders.append(self.block(previous, width, groups))
            previous = width
        for width in reversed(widths[:-1]):
            self.ups.append(nn.ConvTranspose2d(previous, width, 2, stride=2))
            self.decoders.append(self.block(width * 2, width, groups))
            previous = width
        self.head = nn.Conv2d(widths[0], 1, 1)

    @staticmethod
    def block(inputs, outputs, groups):
        return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                             nn.GroupNorm(groups, outputs), nn.ReLU(),
                             nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                             nn.GroupNorm(groups, outputs), nn.ReLU())

    def forward(self, image):
        factor = 2 ** (len(self.encoders) - 1)
        if image.ndim != 4 or image.shape[1] != 3 or any(s % factor for s in image.shape[-2:]):
            raise ValueError("Expected NCHW RGB input with spatial dimensions divisible by encoder stride")
        skips = []
        for index, encoder in enumerate(self.encoders):
            image = encoder(image)
            if index < len(self.encoders) - 1:
                skips.append(image)
                image = nn.functional.max_pool2d(image, 2)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            image = decoder(torch.cat((up(image), skip), dim=1))
        return self.head(image)
