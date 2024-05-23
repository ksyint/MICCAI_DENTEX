from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D
from torch import nn
import torch


# resnet50 fpn
class Resnet50FPN(nn.Module):
    def __init__(self, num_classes, fpn_channels, in_channels=3):
        super().__init__()
        self.encoder = E.Resnet50Encoder()
        if in_channels != 3:
            self.encoder.change_input_channels(in_channels)
        self.decoder = D.FPNCatDecoder(self.encoder.channels, channels=fpn_channels)

        # global aberage pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # ooutput layer
        self.fc = nn.Linear(sum(self.decoder.channels_o), num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # gap and concat
        feat_map_list = []
        for feat_map in x:
            feat_map = self.avgpool(feat_map)
            feat_map = torch.flatten(feat_map, 1)
            feat_map_list.append(feat_map)
        else:
            x = torch.cat(feat_map_list, dim=1)
        out = self.fc(x)
        return out


if __name__ == "__main__":
    model = Resnet50FPN(5, fpn_channels=256, in_channels=1)
    sample_input = torch.randn(1, 1, 256, 256)
    sample_label = torch.randint(0, 4, size=(1,)).long()

    output = model(sample_input)
    print(output.shape)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, sample_label)
    print(loss)
