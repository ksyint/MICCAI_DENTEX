from torch.utils.data import DataLoader, Dataset

# import pytorch_lightning as pl
import lightning.pytorch as pl

from glob import glob

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from pydantic import BaseModel


# pytorch Dataset class
class ClassificationDataset(Dataset):
    def __init__(
        self, image_list: list, label_list: list, img_scale: str = "gray", trans=None
    ) -> None:
        super().__init__()
        assert img_scale in ["gray", "rgb"]

        self.image_paths = image_list
        self.label_list = label_list
        self.img_scale = img_scale
        self.trans = trans

    def get_img(self, path: str) -> np.ndarray:
        if self.img_scale == "gray":
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index):
        img = self.get_img(self.image_paths[index])
        if self.trans is not None:
            img = self.trans(image=img)["image"]
        label = self.label_list[index]
        return img, label


class Dataloaderconfig(BaseModel):
    num_workers: int = 8
    batch_size: int = 16
    shuffle: bool = True


# pytorchl lightning datamoule class
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_fold: int = 5,
        c_fold: int = 0,
        image_scale: str = "gray",
        train_trans=None,
        valid_trans=None,
        train_dataloader_config: Dataloaderconfig = Dataloaderconfig(),
        valid_dataloader_config: Dataloaderconfig = Dataloaderconfig(),
    ) -> None:
        super().__init__()

        self.image_list = glob("../../data/classification/Teeth/*/Images/*png")
        self.label_cat_list = [
            "Caries",
            "DeepCaries",
            "Impacted",
            "Normal",
            "PeriapicalLesion",
        ]
        self.label_list = [
            self.label_cat_list.index(path.split("/")[-3]) for path in self.image_list
        ]

        self.train_dataloader_config = train_dataloader_config
        self.valid_dataloader_config = valid_dataloader_config
        # split train/val
        s_kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
        for i, (train_idx, val_idx) in enumerate(
            s_kfold.split(self.image_list, self.label_list)
        ):
            if i == c_fold:
                self.train_image_list = [self.image_list[idx] for idx in train_idx]
                self.train_label_list = [self.label_list[idx] for idx in train_idx]
                self.valid_image_list = [self.image_list[idx] for idx in val_idx]
                self.valid_label_list = [self.label_list[idx] for idx in val_idx]
                break

        self.train_dataset = ClassificationDataset(
            image_list=self.train_image_list,
            label_list=self.train_label_list,
            img_scale=image_scale,
            trans=train_trans,
        )
        self.valid_dataset = ClassificationDataset(
            image_list=self.valid_image_list,
            label_list=self.valid_label_list,
            img_scale=image_scale,
            trans=valid_trans,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_dataloader_config.dict())

    def val_dataloader(self):
        self.valid_dataloader_config.shuffle = False
        return DataLoader(self.valid_dataset, **self.valid_dataloader_config.dict())


if __name__ == "__main__":
    train_trans = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=0, std=1), ToTensorV2()]
    )
    valid_trans = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=0, std=1), ToTensorV2()]
    )

    dmodule = DataModule(train_trans=train_trans, valid_trans=valid_trans)
    train_loader = dmodule.train_dataloader()
    valid_loader = dmodule.val_dataloader()
    sample_img, sample_label = next(iter(train_loader))
    print(sample_img.shape, sample_label.shape)
