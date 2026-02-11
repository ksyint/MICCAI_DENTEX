import torch
import torch.nn as nn
import torch.nn.functional as F
from .guided_attention import CBAM_Guided
import math

__all__ = ['EfficientNet_CBAM_Guided', 'efficientnet_b0_cbam_guided',
           'efficientnet_b1_cbam_guided', 'efficientnet_b2_cbam_guided',
           'efficientnet_b3_cbam_guided', 'efficientnet_b4_cbam_guided']

class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)

class MBConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio, stride,
                 kernel_size, reduction_ratio=4, use_cbam=True,
                 guidance_weight=0.3, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()

        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.use_cbam = use_cbam

        hidden_dim = in_channels * expand_ratio
        self.use_expansion = expand_ratio != 1
        if self.use_expansion:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)

        self.depthwise_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size,
            stride=stride, padding=kernel_size // 2,
            groups=hidden_dim, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)

        if use_cbam:
            self.cbam = CBAM_Guided(hidden_dim, guidance_weight=guidance_weight)

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.swish = Swish()

        self.use_skip = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x

        if self.use_expansion:
            out = self.expand_conv(x)
            out = self.expand_bn(out)
            out = self.swish(out)
        else:
            out = x

        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.swish(out)

        if self.use_cbam:
            out = self.cbam(out)

        out = self.project_conv(out)
        out = self.project_bn(out)

        if self.use_skip:
            if self.training and self.drop_connect_rate > 0:
                out = self.drop_connect(out, self.drop_connect_rate)
            out = out + identity

        return out

    def drop_connect(self, x, drop_ratio):

        keep_prob = 1 - drop_ratio
        batch_size = x.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        output = x / keep_prob * binary_tensor
        return output

class EfficientNet_CBAM_Guided(nn.Module):

    def __init__(self, width_coefficient, depth_coefficient, dropout_rate,
                 num_classes=1000, use_cbam=True, guidance_weight=0.3):
        super(EfficientNet_CBAM_Guided, self).__init__()

        blocks_args = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]

        def round_filters(filters, multiplier):

            if not multiplier:
                return filters
            filters *= multiplier
            new_filters = max(8, int(filters + 4) // 8 * 8)
            if new_filters < 0.9 * filters:
                new_filters += 8
            return int(new_filters)

        def round_repeats(repeats, multiplier):

            if not multiplier:
                return repeats
            return int(math.ceil(multiplier * repeats))

        out_channels = round_filters(32, width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )

        self.blocks = nn.ModuleList([])
        in_channels = out_channels

        for expand_ratio, channels, num_blocks, stride, kernel_size in blocks_args:
            out_channels = round_filters(channels, width_coefficient)
            num_blocks = round_repeats(num_blocks, depth_coefficient)

            for i in range(num_blocks):
                self.blocks.append(
                    MBConvBlock(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        expand_ratio,
                        stride if i == 0 else 1,
                        kernel_size,
                        use_cbam=use_cbam,
                        guidance_weight=guidance_weight
                    )
                )

        final_channels = round_filters(1280, width_coefficient)
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(final_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def efficientnet_b0_cbam_guided(num_classes=1000, **kwargs):

    return EfficientNet_CBAM_Guided(1.0, 1.0, 0.2, num_classes, **kwargs)

def efficientnet_b1_cbam_guided(num_classes=1000, **kwargs):

    return EfficientNet_CBAM_Guided(1.0, 1.1, 0.2, num_classes, **kwargs)

def efficientnet_b2_cbam_guided(num_classes=1000, **kwargs):

    return EfficientNet_CBAM_Guided(1.1, 1.2, 0.3, num_classes, **kwargs)

def efficientnet_b3_cbam_guided(num_classes=1000, **kwargs):

    return EfficientNet_CBAM_Guided(1.2, 1.4, 0.3, num_classes, **kwargs)

def efficientnet_b4_cbam_guided(num_classes=1000, **kwargs):

    return EfficientNet_CBAM_Guided(1.4, 1.8, 0.4, num_classes, **kwargs)
