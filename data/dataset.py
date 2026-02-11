"""
Dataset classes for Semi-Supervised Learning
Supports labeled, unlabeled, and test datasets
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ToothDataset(Dataset):
    """
    Base dataset for tooth classification
    
    Args:
        data_root: root directory of images
        data_list: list of (image_path, label) tuples
        transform: image transformations
    """
    def __init__(self, data_root, data_list, transform=None):
        self.data_root = data_root
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        
        img_full_path = os.path.join(self.data_root, img_path)
        image = Image.open(img_full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class LabeledDataset(ToothDataset):
    """Dataset for labeled data"""
    pass


class UnlabeledDataset(Dataset):
    """
    Dataset for unlabeled data with weak and strong augmentations
    Used in Meta Pseudo Label
    
    Args:
        data_root: root directory of images
        data_list: list of image paths (no labels)
        weak_transform: weak augmentation
        strong_transform: strong augmentation
    """
    def __init__(self, data_root, data_list, weak_transform=None, strong_transform=None):
        self.data_root = data_root
        self.data_list = data_list
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        
        # Load image
        img_full_path = os.path.join(self.data_root, img_path)
        image = Image.open(img_full_path).convert('RGB')
        
        # Apply weak augmentation (for teacher)
        if self.weak_transform:
            weak_aug = self.weak_transform(image)
        else:
            weak_aug = transforms.ToTensor()(image)
        
        # Apply strong augmentation (for student)
        if self.strong_transform:
            strong_aug = self.strong_transform(image)
        else:
            strong_aug = transforms.ToTensor()(image)
        
        return weak_aug, strong_aug


class SemiSupervisedDataset(Dataset):
    """
    Combined dataset for semi-supervised learning
    Returns both labeled and unlabeled samples
    
    Args:
        labeled_dataset: labeled dataset
        unlabeled_dataset: unlabeled dataset
        num_iters: number of iterations per epoch
    """
    def __init__(self, labeled_dataset, unlabeled_dataset, num_iters=None):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        
        if num_iters is None:
            num_iters = max(len(labeled_dataset), len(unlabeled_dataset))
        self.num_iters = num_iters
    
    def __len__(self):
        return self.num_iters
    
    def __getitem__(self, idx):
        # Sample from labeled data
        labeled_idx = idx % len(self.labeled_dataset)
        labeled_img, label = self.labeled_dataset[labeled_idx]
        
        # Sample from unlabeled data
        unlabeled_idx = idx % len(self.unlabeled_dataset)
        unlabeled_weak, unlabeled_strong = self.unlabeled_dataset[unlabeled_idx]
        
        return {
            'labeled_img': labeled_img,
            'label': label,
            'unlabeled_weak': unlabeled_weak,
            'unlabeled_strong': unlabeled_strong,
        }


def load_data_list(data_file):
    """
    Load data list from file
    
    Format for labeled data: image_path\tlabel
    Format for unlabeled data: image_path
    
    Args:
        data_file: path to data list file
    
    Returns:
        list of (image_path, label) tuples for labeled data
        or list of image_paths for unlabeled data
    """
    data_list = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            
            if len(parts) == 2:
                # Labeled data
                img_path, label = parts
                data_list.append((img_path, int(label)))
            elif len(parts) == 1:
                # Unlabeled data
                img_path = parts[0]
                data_list.append(img_path)
            else:
                raise ValueError(f"Invalid line format: {line}")
    
    return data_list


def split_labeled_unlabeled(data_list, labeled_ratio=0.1, seed=42):
    """
    Split dataset into labeled and unlabeled portions
    
    Args:
        data_list: list of (image_path, label) tuples
        labeled_ratio: ratio of labeled data
        seed: random seed
    
    Returns:
        labeled_list: list of (image_path, label) for labeled data
        unlabeled_list: list of image_path for unlabeled data
    """
    np.random.seed(seed)
    
    # Shuffle data
    indices = np.random.permutation(len(data_list))
    split_idx = int(len(data_list) * labeled_ratio)
    
    labeled_indices = indices[:split_idx]
    unlabeled_indices = indices[split_idx:]
    
    labeled_list = [data_list[i] for i in labeled_indices]
    unlabeled_list = [data_list[i][0] for i in unlabeled_indices]  # Only image paths
    
    return labeled_list, unlabeled_list


def create_datasets(args):
    """
    Create datasets for semi-supervised learning
    
    Args:
        args: arguments containing:
            - data_root: root directory of images
            - labeled_file: file with labeled data
            - unlabeled_file: file with unlabeled data
            - test_file: file with test data
            - weak_transform: weak augmentation
            - strong_transform: strong augmentation
            - test_transform: test transform
    
    Returns:
        labeled_dataset: labeled training dataset
        unlabeled_dataset: unlabeled dataset
        test_dataset: test dataset
    """
    from .transforms import get_transforms
    
    # Get transforms
    weak_transform, strong_transform, test_transform = get_transforms(args)
    
    # Load data lists
    labeled_list = load_data_list(args.labeled_file)
    unlabeled_list = load_data_list(args.unlabeled_file)
    test_list = load_data_list(args.test_file)
    
    # Create datasets
    labeled_dataset = LabeledDataset(
        args.data_root, labeled_list, transform=weak_transform
    )
    
    unlabeled_dataset = UnlabeledDataset(
        args.data_root, unlabeled_list,
        weak_transform=weak_transform,
        strong_transform=strong_transform
    )
    
    test_dataset = ToothDataset(
        args.data_root, test_list, transform=test_transform
    )
    
    return labeled_dataset, unlabeled_dataset, test_dataset
