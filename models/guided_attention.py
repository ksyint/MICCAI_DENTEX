"""
Guided Attention Module for Tooth Classification
This module applies spatial attention with guidance towards the bottom half of images,
while maintaining adaptiveness to avoid over-dependence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedSpatialAttention(nn.Module):
    """
    Spatial attention with guidance for bottom-half regions.
    Balances between data-driven attention and bottom-region bias.
    """
    def __init__(self, kernel_size=7, guidance_weight=0.3):
        super(GuidedSpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.guidance_weight = guidance_weight
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(concat)
        
        B, _, H, W = x.shape
        guidance_mask = self._create_bottom_guidance_mask(B, H, W, x.device)
        
        guided_attention = attention_map + self.guidance_weight * guidance_mask
        
        return self.sigmoid(guided_attention)
    
    def _create_bottom_guidance_mask(self, batch_size, height, width, device):
        """
        Creates a smooth gradient mask that gives higher weights to bottom regions.
        Uses smooth transition to avoid hard boundaries.
        """
        y_coords = torch.linspace(0, 1, height, device=device)
        gradient = y_coords.view(height, 1).expand(height, width)
        
        gradient = torch.pow(gradient, 2)
        
        mask = gradient.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)
        
        return mask


class AdaptiveGuidedAttention(nn.Module):
    """
    Adaptive version that learns to modulate the guidance weight.
    This prevents over-dependence on bottom regions.
    """
    def __init__(self, in_channels, kernel_size=7, base_guidance=0.3):
        super(AdaptiveGuidedAttention, self).__init__()
        self.kernel_size = kernel_size
        self.base_guidance = base_guidance
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
        self.guidance_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, 1, 1),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(concat)
        
        guidance_weight = self.guidance_predictor(x) * self.base_guidance
        
        B, _, H, W = x.shape
        guidance_mask = self._create_bottom_guidance_mask(B, H, W, x.device)
        
        guided_attention = attention_map + guidance_weight * guidance_mask
        
        return self.sigmoid(guided_attention)
    
    def _create_bottom_guidance_mask(self, batch_size, height, width, device):
        """Creates smooth bottom-half guidance mask"""
        y_coords = torch.linspace(0, 1, height, device=device)
        gradient = y_coords.view(height, 1).expand(height, width)
        gradient = torch.pow(gradient, 2)
        mask = gradient.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)
        return mask


class ChannelAttention(nn.Module):
    """Channel Attention from CBAM"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class CBAM_Guided(nn.Module):
    """
    CBAM with Guided Spatial Attention for tooth classification.
    Combines channel attention with bottom-region guided spatial attention.
    """
    def __init__(self, in_channels, ratio=16, kernel_size=7, 
                 guidance_weight=0.3, adaptive=True):
        super(CBAM_Guided, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, ratio)
        
        if adaptive:
            self.spatial_attention = AdaptiveGuidedAttention(
                in_channels, kernel_size, guidance_weight
            )
        else:
            self.spatial_attention = GuidedSpatialAttention(
                kernel_size, guidance_weight
            )
    
    def forward(self, x):
        x = self.channel_attention(x) * x
        
        x = self.spatial_attention(x) * x
        
        return x
