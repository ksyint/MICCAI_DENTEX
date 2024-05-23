# import pytorch_lightning as pl
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor

import torchmetrics

import torch
from torch import nn

from pydantic import BaseModel
import albumentations as A
from albumentations.pytorch import ToTensorV2


from dataset.datamodule import DataModule
from model_class.fpn_model import Resnet50FPN


# class ModelConfig(BaseModel):
#     class TimmConfig(BaseModel):
#         backbone_name: str = "resnet18"
#         in_chans: int = 1
#         num_classes: int = 5
#         pretrained = True


# class TrainConfig(BaseModel):
#     num_epochs: int = 10

#     learning_rate: float = 1e-3
#     optimizer: torch.optim.Optimizer = torch.optim.Adam(lr=learning_rate)
#     learning_reate_scheduler: torch.optim.lr_scheduler = (
#         torch.optim.lr_scheduler.CosineAnnealingLR(T_max=10)
#     )
#     loss_fn: nn.Module = nn.CrossEntropyLoss()


# class ResourceConfig(BaseModel):
#     devices: list = [0]


# class DatasetConfig(BaseModel):
#     base_dir: str = "/home/username/dataset"
#     read_gray: bool = True

#     # data preprocessing by albumentations
#     transform: A.Compose = A.Compose(
#         [A.Resize(256, 256), A.Normalize(mean=0, std=1), ToTensorV2()]
#     )

#     class DataloaderConfig(BaseModel):
#         batch_size: int = 32
#         num_workers: int = 4
#         shuffle: bool = True
#         pin_memory: bool = True


# pytorch lightning class
class LightningModel(pl.LightningModule):
    def __init__(
        self,
    ):
        super(LightningModel, self).__init__()

        self.backbone = Resnet50FPN(num_classes=5, fpn_channels=256, in_channels=1)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_metric = torchmetrics.F1Score(num_classes=5, task="multiclass")
        self.valid_metric = torchmetrics.F1Score(num_classes=5, task="multiclass")

    def forward(self, x):
        out = self.backbone(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)

        self.log("train_f1", self.train_metric(out, y))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)

        self.log("valid_f1", self.valid_metric(out, y))
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        return [optimizer], [scheduler]


def main():
    randomness_augmentation = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]

    train_trans = A.Compose(
        [
            A.Resize(512, 512),
            # Randomness Augmentation
            *randomness_augmentation,
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ]
    )
    valid_trans = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ]
    )
    datamodule = DataModule(train_trans=train_trans, valid_trans=valid_trans)
    model = LightningModel()
    # callback
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer_config_dict = {
        "accelerator": "mps",
        "devices": 1,
        "max_epochs": 10,
        "callbacks": [lr_monitor],
    }

    trainer = pl.Trainer(**trainer_config_dict)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
