import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import random
import numpy as np

class RandAugment:

    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augment_list = [
            (AutoContrast, 0, 1),
            (Brightness, 0.05, 0.95),
            (Color, 0.05, 0.95),
            (Contrast, 0.05, 0.95),
            (Equalize, 0, 1),
            (Identity, 0, 1),
            (Posterize, 4, 8),
            (Rotate, -30, 30),
            (Sharpness, 0.05, 0.95),
            (ShearX, -0.3, 0.3),
            (ShearY, -0.3, 0.3),
            (Solarize, 0, 256),
            (TranslateX, -0.3, 0.3),
            (TranslateY, -0.3, 0.3),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        return img

def AutoContrast(img, _):
    return ImageOps.autocontrast(img)

def Brightness(img, v):
    return transforms.functional.adjust_brightness(img, v)

def Color(img, v):
    return transforms.functional.adjust_saturation(img, v)

def Contrast(img, v):
    return transforms.functional.adjust_contrast(img, v)

def Equalize(img, _):
    return ImageOps.equalize(img)

def Identity(img, _):
    return img

def Posterize(img, v):
    v = int(v)
    return ImageOps.posterize(img, v)

def Rotate(img, v):
    return img.rotate(v)

def Sharpness(img, v):
    return transforms.functional.adjust_sharpness(img, v)

def ShearX(img, v):
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

def Solarize(img, v):
    return ImageOps.solarize(img, int(v))

def TranslateX(img, v):
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

class GaussianBlur:

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class BottomHalfMask:

    def __init__(self, mask_prob=0.2, mask_value=0.5):
        self.mask_prob = mask_prob
        self.mask_value = mask_value

    def __call__(self, img):
        if random.random() < self.mask_prob:

            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
                was_pil = True
            else:
                was_pil = False

            _, h, w = img.shape
            img[:, :h//2, :] = self.mask_value

            if was_pil:
                img = transforms.ToPILImage()(img)

        return img

def get_transforms(args):

    img_size = getattr(args, 'img_size', 224)
    mean = getattr(args, 'mean', [0.485, 0.456, 0.406])
    std = getattr(args, 'std', [0.229, 0.224, 0.225])

    weak_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    strong_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        RandAugment(n=2, m=10),
        transforms.ToTensor(),

        transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return weak_transform, strong_transform, test_transform

def get_train_transform(img_size=224, mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]):

    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_val_transform(img_size=224, mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]):

    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
