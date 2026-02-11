"""
ResNet models with CBAM and Guided Attention
Supports ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .guided_attention import CBAM_Guided


__all__ = ['ResNet_CBAM_Guided', 'resnet18_cbam_guided', 'resnet34_cbam_guided',
           'resnet50_cbam_guided', 'resnet101_cbam_guided', 'resnet152_cbam_guided']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """Basic Block for ResNet18 and ResNet34"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 use_cbam=True, guidance_weight=0.3):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM_Guided(planes, guidance_weight=guidance_weight)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck Block for ResNet50, ResNet101, ResNet152"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 use_cbam=True, guidance_weight=0.3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM_Guided(planes * 4, guidance_weight=guidance_weight)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_CBAM_Guided(nn.Module):
    """
    ResNet with CBAM and Guided Attention
    
    Args:
        block: BasicBlock or Bottleneck
        layers: list of layer sizes
        num_classes: number of output classes
        use_cbam: whether to use CBAM modules
        guidance_weight: weight for bottom-region guidance (0-1)
    """
    def __init__(self, block, layers, num_classes=1000, use_cbam=True,
                 guidance_weight=0.3, dropout=0.0):
        self.inplanes = 64
        self.use_cbam = use_cbam
        self.guidance_weight = guidance_weight
        
        super(ResNet_CBAM_Guided, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                           use_cbam=self.use_cbam,
                           guidance_weight=self.guidance_weight))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                               use_cbam=self.use_cbam,
                               guidance_weight=self.guidance_weight))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def resnet18_cbam_guided(pretrained=False, num_classes=1000, **kwargs):
    """ResNet-18 with CBAM and Guided Attention"""
    model = ResNet_CBAM_Guided(BasicBlock, [2, 2, 2, 2], 
                               num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet34_cbam_guided(pretrained=False, num_classes=1000, **kwargs):
    """ResNet-34 with CBAM and Guided Attention"""
    model = ResNet_CBAM_Guided(BasicBlock, [3, 4, 6, 3],
                               num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50_cbam_guided(pretrained=False, num_classes=1000, **kwargs):
    """ResNet-50 with CBAM and Guided Attention"""
    model = ResNet_CBAM_Guided(Bottleneck, [3, 4, 6, 3],
                               num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101_cbam_guided(pretrained=False, num_classes=1000, **kwargs):
    """ResNet-101 with CBAM and Guided Attention"""
    model = ResNet_CBAM_Guided(Bottleneck, [3, 4, 23, 3],
                               num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152_cbam_guided(pretrained=False, num_classes=1000, **kwargs):
    """ResNet-152 with CBAM and Guided Attention"""
    model = ResNet_CBAM_Guided(Bottleneck, [3, 8, 36, 3],
                               num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
