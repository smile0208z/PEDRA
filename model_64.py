import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Patch Embedding Module
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels):
        super(PatchEmbedding, self).__init__()
        self.patch_embed = nn.Conv2d(in_channels, 64, kernel_size=64, stride=64, bias=False)
        self.drop = nn.Dropout(p=0.1)
        self.max_pooling = nn.MaxPool1d(kernel_size=8, stride=8)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = self.drop(x)
        x = x.unsqueeze(1)
        return x

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(input_channels // reduction_ratio, input_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

# Combined Attention Module
class CombinedAttention(nn.Module):
    def __init__(self, out_channels):
        super(CombinedAttention, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
            nn.Dropout(p=0.1)
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
            nn.Dropout(p=0.1)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
            nn.Dropout(p=0.1)
        )

    def forward(self, features, weight):
        original = features
        channel_att = self.channel_gate(features)
        spatial_att = self.spatial_gate(features) * weight
        combined_att = channel_att * spatial_att
        gate_output = self.gate(features)
        final_att = combined_att * gate_output
        return original + final_att * features

# Attention Module
class AttentionModule(nn.Module):
    def __init__(self, out_channels):
        super(AttentionModule, self).__init__()
        self.attention = CombinedAttention(out_channels)
        self.conv_att = nn.Sequential(
            nn.Conv2d(out_channels + 1, 320, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(320, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )

    def forward(self, original_features, processed_features):
        processed_features = F.interpolate(processed_features, size=original_features.size()[2:], mode='bilinear', align_corners=False)
        weights = torch.sigmoid(processed_features)
        att_input = torch.cat([original_features, processed_features], dim=1)
        processed_input = self.conv_att(att_input)
        return self.attention(processed_input, weights)

# Mobile Inverted Bottleneck Convolution Block
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, stride, padding, reduction_ratio=4):
        super(MBConvBlock, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.SiLU(inplace=True),
            SEBlock(in_channels * expansion_factor, reduction_ratio),
            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out += x
        return out

# EfficientNet B0 Model
class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels=1)
        self.initial_conv = nn.Sequential(
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
        self.mb_conv_blocks = nn.ModuleList([MBConvBlock(*params) for params in mb_params])
        self.attention_modules = nn.ModuleList([
            AttentionModule(24), AttentionModule(40), AttentionModule(80), AttentionModule(192), AttentionModule(320)
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
        x_resized = self.patch_embedding(x)
        x = self.initial_conv(x)
        att_indices = {1: 0, 3: 1, 5: 2, 11: 3, 15: 4}
        for i, mb_conv in enumerate(self.mb_conv_blocks):
            x = mb_conv(x)
            if i in att_indices:
                x = x + self.attention_modules[att_indices[i]](x, x_resized)
        x = self.conv(x)
        feature = self.feature(x)
        classification = self.classifier(x)
        return feature, classification
