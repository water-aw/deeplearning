import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic conv block: conv -> BN -> ReLU -> conv -> BN -> ReLU
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)  # concat channel = out_ch (from up) + out_ch (skip)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed to handle odd input sizes
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_c=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_c) 
        self.down1 = Down(base_c, base_c * 2) 
        self.down2 = Down(base_c * 2, base_c * 4) 
        self.down3 = Down(base_c * 4, base_c * 8) 
        self.down4 = Down(base_c * 8, base_c * 16) 
        self.up1 = Up(base_c * 16, base_c * 8) 
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)
        self.outc = nn.Conv2d(base_c, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# quick sanity check
if __name__ == "__main__":
    net = UNet(n_channels=3, n_classes=2)
    inp = torch.randn(1, 3, 256, 256)
    out = net(inp)
    print("out shape:", out.shape)  # expect [1, 2, 256, 256]
