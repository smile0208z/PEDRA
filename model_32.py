import torch
import torch.nn as nn
import torch.nn.functional as F

class PEBlock(nn.Module):
    def __init__(self, c_in):
        super(PEBlock, self).__init__()
        layers = [
            nn.Conv2d(c_in, 32, kernel_size=32, stride=32, bias=False),
            nn.Dropout(p=0.1),
            nn.MaxPool1d(kernel_size=8, stride=8)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        x = x.flatten(2)
        x = x.unsqueeze(1)
        return x

class SEBBlock(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(SEBBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(c_in, c_in // reduction, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_in // reduction, c_in, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.pool(x)
        y = self.fc(y)
        return x * y

class CABlock(nn.Module):
    def __init__(self, c_out):
        super(CABlock, self).__init__()
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, c_out, kernel_size=1),
            nn.Sigmoid(),
            nn.Dropout(p=0.1)
        )
        self.spa_gate = nn.Sequential(
            nn.Conv2d(c_out, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
            nn.Dropout(p=0.1)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c_out, c_out, kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c_out, c_out, kernel_size=1),
            nn.Sigmoid(),
            nn.Dropout(p=0.1)
        )

    def forward(self, features, weight):
        ch_att = self.ch_gate(features)
        spa_att = self.spa_gate(features) * weight
        comb_att = ch_att * spa_att
        gate_out = self.gate(features)
        final_att = comb_att * gate_out
        return features + final_att * features

class AMBlock(nn.Module):
    def __init__(self, c_out):
        super(AMBlock, self).__init__()
        self.att = CABlock(c_out)
        self.conv = nn.Sequential(
            nn.Conv2d(c_out + 1, 320, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(320, c_out, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )

    def forward(self, orig_features, proc_features):
        proc_features = F.interpolate(proc_features, size=orig_features.shape[2:], mode='bilinear', align_corners=False)
        weight = torch.sigmoid(proc_features)
        att_input = torch.cat([orig_features, proc_features], dim=1)
        processed_input = self.conv(att_input)
        return self.att(processed_input, weight)

class MBConvBlk(nn.Module):
    def __init__(self, c_in, c_out, exp_factor, k_size, stride, pad, reduction=4):
        super(MBConvBlk, self).__init__()
        self.use_res = stride == 1 and c_in == c_out
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * exp_factor, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_in * exp_factor),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_in * exp_factor, c_in * exp_factor, kernel_size=k_size, stride=stride, padding=pad, groups=c_in * exp_factor, bias=False),
            nn.BatchNorm2d(c_in * exp_factor),
            nn.SiLU(inplace=True),
            SEBBlock(c_in * exp_factor, reduction),
            nn.Conv2d(c_in * exp_factor, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.use_res:
            out += identity
        return out

class EffNetB0(nn.Module):
    def __init__(self):
        super(EffNetB0, self).__init__()
        self.pe_block = PEBlock(c_in=1)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        mb_params = [
            (32, 16, 1, 3, 1), (16, 24, 6, 3, 1), (24, 24, 6, 3, 1),
            (24, 40, 6, 5, 2), (40, 40, 6, 5, 2), (40, 80, 6, 3, 1),
            (80, 80, 6, 3, 1), (80, 80, 6, 3, 1), (80, 112, 6, 5, 2),
            (112, 112, 6, 5, 2), (112, 112, 6, 5, 2), (112, 192, 6, 5, 2),
            (192, 192, 6, 5, 2), (192, 192, 6, 5, 2), (192, 192, 6, 5, 2),
            (192, 320, 6, 3, 1)
        ]
        self.mb_blocks = nn.ModuleList([MBConvBlk(*params) for params in mb_params])
        self.att_blocks = nn.ModuleList([
            AMBlock(24), AMBlock(40), AMBlock(80), AMBlock(192), AMBlock(320)
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True)
        )
        self.feature = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, 1)
        )

    def forward(self, x):
        x_resized = self.pe_block(x)
        x = self.init_conv(x)
        att_indices = {1: 0, 3: 1, 5: 2, 11: 3, 15: 4}
        for i, mb in enumerate(self.mb_blocks):
            x = mb(x)
            if i in att_indices:
                x = x + self.att_blocks[att_indices[i]](x, x_resized)
        x = self.conv(x)
        feature = self.feature(x)
        classification = self.classifier(x)
        return feature, classification