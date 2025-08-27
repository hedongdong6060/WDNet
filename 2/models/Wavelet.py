import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


def down_conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2, padding=0
    )


def same_conv(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )


def transition_conv(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out


class DoubleBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, transition=None):
        super(DoubleBasicBlock, self).__init__()
        self.block1 = BasicBlock(in_channels, out_channels)
        self.block2 = BasicBlock(out_channels, out_channels)
        self.transition = transition

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        if self.transition:
            out = self.transition(out)
        return out


class DiffusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_diffusion_steps=4):
        super(DiffusionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_diffusion_steps)
            ]
        )
        self.channel_project = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        for layer in self.noise_layers:
            noise = torch.randn_like(x) * 0.1
            x = x + noise
            x = layer(x)
        x = self.channel_project(x)
        return x


class WaveletDecoder(nn.Module):
    def __init__(self, num_classes, num_diffusion_steps=4):
        super(WaveletDecoder, self).__init__()
        self.diff5 = DiffusionBlock(
            in_channels=2048, out_channels=1024, num_diffusion_steps=num_diffusion_steps
        )
        self.up4 = up_conv(1024, 512)

        self.diff4 = DiffusionBlock(
            in_channels=1536, out_channels=512, num_diffusion_steps=num_diffusion_steps
        )
        self.up3 = up_conv(512, 256)

        self.diff3 = DiffusionBlock(
            in_channels=768, out_channels=256, num_diffusion_steps=num_diffusion_steps
        )
        self.up2 = up_conv(256, 128)

        self.diff2 = DiffusionBlock(
            in_channels=384, out_channels=128, num_diffusion_steps=num_diffusion_steps
        )
        self.up1 = up_conv(128, 64)
        self.diff1 = DiffusionBlock(
            in_channels=192, out_channels=64, num_diffusion_steps=num_diffusion_steps
        )

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, c5, c4, c3, c2, c1):
        d5 = self.diff5(c5)
        d4_up = self.up4(d5)
        d4_in = torch.cat([d4_up, c4], dim=1)
        d4 = self.diff4(d4_in)

        d3_up = self.up3(d4)
        d3_in = torch.cat([d3_up, c3], dim=1)
        d3 = self.diff3(d3_in)

        d2_up = self.up2(d3)
        d2_in = torch.cat([d2_up, c2], dim=1)
        d2 = self.diff2(d2_in)

        d1_up = self.up1(d2)
        d1_in = torch.cat([d1_up, c1], dim=1)
        d1 = self.diff1(d1_in)

        out = self.out_conv(d1)
        return out


class WaveletNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_diffusion_steps=4):
        super(WaveletNet, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c), conv3x3(l1c, l1c), BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(
            l1c + l1c,
            l1c,
            nn.Sequential(
                conv1x1(l1c + l1c, l1c),
                nn.BatchNorm2d(l1c, momentum=BN_MOMENTUM),
            ),
        )
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)

        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(
            l2c + l2c,
            l2c,
            nn.Sequential(
                conv1x1(l2c + l2c, l2c),
                nn.BatchNorm2d(l2c, momentum=BN_MOMENTUM),
            ),
        )
        self.b1_2_4_up = up_conv(l2c, l1c)

        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_3 = DoubleBasicBlock(
            l3c + l3c,
            l3c,
            nn.Sequential(
                conv1x1(l3c + l3c, l3c),
                nn.BatchNorm2d(l3c, momentum=BN_MOMENTUM),
            ),
        )
        self.b1_3_4_up = up_conv(l3c, l2c)

        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_3_down = down_conv(l4c, l4c)
        self.b1_4_3_same = same_conv(l4c, l4c)
        self.b1_4_4_transition = transition_conv(l4c + l5c + l4c, l4c)
        self.b1_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_6 = DoubleBasicBlock(
            l4c + l4c,
            l4c,
            nn.Sequential(
                conv1x1(l4c + l4c, l4c),
                nn.BatchNorm2d(l4c, momentum=BN_MOMENTUM),
            ),
        )
        self.b1_4_7_up = up_conv(l4c, l3c)

        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_2_up = up_conv(l5c, l5c)
        self.b1_5_2_same = same_conv(l5c, l5c)
        self.b1_5_3_transition = transition_conv(l5c + l5c + l4c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # branch2
        self.b2_1_1 = nn.Sequential(
            conv3x3(1, l1c), conv3x3(l1c, l1c), BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(
            l1c + l1c,
            l1c,
            nn.Sequential(
                conv1x1(l1c + l1c, l1c),
                nn.BatchNorm2d(l1c, momentum=BN_MOMENTUM),
            ),
        )
        self.b2_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)

        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(
            l2c + l2c,
            l2c,
            nn.Sequential(
                conv1x1(l2c + l2c, l2c),
                nn.BatchNorm2d(l2c, momentum=BN_MOMENTUM),
            ),
        )
        self.b2_2_4_up = up_conv(l2c, l1c)

        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_3 = DoubleBasicBlock(
            l3c + l3c,
            l3c,
            nn.Sequential(
                conv1x1(l3c + l3c, l3c),
                nn.BatchNorm2d(l3c, momentum=BN_MOMENTUM),
            ),
        )
        self.b2_3_4_up = up_conv(l3c, l2c)

        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_2_down = down_conv(l4c, l5c)
        self.b2_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_3_down = down_conv(l4c, l4c)
        self.b2_4_3_same = same_conv(l4c, l4c)
        self.b2_4_4_transition = transition_conv(l4c + l5c + l4c, l4c)
        self.b2_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_6 = DoubleBasicBlock(
            l4c + l4c,
            l4c,
            nn.Sequential(
                conv1x1(l4c + l4c, l4c),
                nn.BatchNorm2d(l4c, momentum=BN_MOMENTUM),
            ),
        )
        self.b2_4_7_up = up_conv(l4c, l3c)

        self.b2_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_2_up = up_conv(l5c, l5c)
        self.b2_5_2_same = same_conv(l5c, l5c)
        self.b2_5_3_transition = transition_conv(l5c + l5c + l4c, l5c)
        self.b2_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_5_up = up_conv(l5c, l4c)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.decoder = WaveletDecoder(num_classes, num_diffusion_steps)

    def forward(self, input1, input2):
        x1_1 = self.b1_1_1(input1)
        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)
        x1_3 = self.b1_2_2_down(x1_2)
        x1_3 = self.b1_3_1(x1_3)
        x1_4_1 = self.b1_3_2_down(x1_3)
        x1_4_1 = self.b1_4_1(x1_4_1)
        x1_4_2 = self.b1_4_2_down(x1_4_1)
        x1_4_2 = self.b1_5_1(x1_4_2)

        x2_1 = self.b2_1_1(input2)
        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)
        x2_3 = self.b2_2_2_down(x2_2)
        x2_3 = self.b2_3_1(x2_3)
        x2_4_1 = self.b2_3_2_down(x2_3)
        x2_4_1 = self.b2_4_1(x2_4_1)
        x2_4_2 = self.b2_4_2_down(x2_4_1)
        x2_4_2 = self.b2_5_1(x2_4_2)

        c5 = torch.cat([x1_4_2, x2_4_2], dim=1)

        c4 = torch.cat([x1_4_1, x2_4_1], dim=1)

        c3 = torch.cat([x1_3, x2_3], dim=1)

        c2 = torch.cat([x1_2, x2_2], dim=1)

        c1 = torch.cat([x1_1, x2_1], dim=1)

        out = self.decoder(c5, c4, c3, c2, c1)
        return out
