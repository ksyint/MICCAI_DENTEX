import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .guided_attention import CBAM_Guided

__all__ = ['VGG_CBAM_Guided', 'vgg11_bn_cbam_guided', 'vgg13_bn_cbam_guided',
           'vgg16_bn_cbam_guided', 'vgg19_bn_cbam_guided']

model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_CBAM_Guided(nn.Module):

    def __init__(self, features, num_classes=1000, dropout=0.5, init_weights=True):
        super(VGG_CBAM_Guided, self).__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=True, use_cbam=True, guidance_weight=0.3):

    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            if use_cbam:
                layers += [CBAM_Guided(v, guidance_weight=guidance_weight)]

            in_channels = v

    return nn.Sequential(*layers)

def vgg11_bn_cbam_guided(pretrained=False, num_classes=1000, **kwargs):

    model = VGG_CBAM_Guided(make_layers(cfg['A'], **kwargs),
                            num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['vgg11_bn'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg13_bn_cbam_guided(pretrained=False, num_classes=1000, **kwargs):

    model = VGG_CBAM_Guided(make_layers(cfg['B'], **kwargs),
                            num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['vgg13_bn'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg16_bn_cbam_guided(pretrained=False, num_classes=1000, **kwargs):

    model = VGG_CBAM_Guided(make_layers(cfg['D'], **kwargs),
                            num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg19_bn_cbam_guided(pretrained=False, num_classes=1000, **kwargs):

    model = VGG_CBAM_Guided(make_layers(cfg['E'], **kwargs),
                            num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
