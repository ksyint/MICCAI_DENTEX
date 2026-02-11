"""
Main training script for Semi-Supervised Tooth Classification
Combines Meta Pseudo Label with CBAM and Guided Attention
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from configs.default_config import get_default_config, update_config
from models import create_model
from data import create_datasets, LabeledDataset, UnlabeledDataset
from data.transforms import get_transforms
from trainers import MPLTrainer
from utils import Logger, TensorboardLogger


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(args):
    """Create data loaders for training and validation"""
    from data.dataset import load_data_list
    
    weak_transform, strong_transform, test_transform = get_transforms(args)
    
    labeled_list = load_data_list(args.labeled_file)
    unlabeled_list = load_data_list(args.unlabeled_file)
    test_list = load_data_list(args.test_file)
    
    labeled_dataset = LabeledDataset(
        args.data_root, labeled_list, transform=weak_transform
    )
    
    unlabeled_dataset = UnlabeledDataset(
        args.data_root, unlabeled_list,
        weak_transform=weak_transform,
        strong_transform=strong_transform
    )
    
    test_dataset = LabeledDataset(
        args.data_root, test_list, transform=test_transform
    )
    
    train_labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    train_unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.unlabeled_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    from itertools import cycle
    
    class CombinedLoader:
        """Combined loader for labeled and unlabeled data"""
        def __init__(self, labeled_loader, unlabeled_loader):
            self.labeled_loader = labeled_loader
            self.unlabeled_loader = unlabeled_loader
            self.labeled_iter = iter(cycle(labeled_loader))
            self.unlabeled_iter = iter(cycle(unlabeled_loader))
            # Use longer loader length
            self.length = max(len(labeled_loader), len(unlabeled_loader))
        
        def __iter__(self):
            self.labeled_iter = iter(cycle(self.labeled_loader))
            self.unlabeled_iter = iter(cycle(self.unlabeled_loader))
            return self
        
        def __next__(self):
            labeled_img, label = next(self.labeled_iter)
            unlabeled_weak, unlabeled_strong = next(self.unlabeled_iter)
            
            return {
                'labeled_img': labeled_img,
                'label': label,
                'unlabeled_weak': unlabeled_weak,
                'unlabeled_strong': unlabeled_strong,
            }
        
        def __len__(self):
            return self.length
    
    train_loader = CombinedLoader(train_labeled_loader, train_unlabeled_loader)
    
    return train_loader, test_loader


def create_optimizer_and_scheduler(model, args, is_teacher=False):
    """Create optimizer and learning rate scheduler"""
    lr = args.teacher_lr if is_teacher else args.lr
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )
    
    if args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=0
        )
    elif args.lr_schedule == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=args.milestones,
            gamma=0.1
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    
    return optimizer, scheduler


def main():
    """Main training function"""
    # Parse arguments
    parser = get_default_config()
    args = parser.parse_args()
    args = update_config(args)
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize logger
    logger = Logger(args.log_dir, log_name='train.log')
    logger.info('='*60)
    logger.info('Semi-Supervised Tooth Classification')
    logger.info('Meta Pseudo Label + CBAM + Guided Attention')
    logger.info('='*60)
    
    # Log configuration
    logger.info('Configuration:')
    for arg in vars(args):
        logger.info(f'  {arg:30s}: {getattr(args, arg)}')
    logger.info('='*60)
    
    # Initialize tensorboard
    if args.tensorboard:
        tb_logger = TensorboardLogger(log_dir=os.path.join(args.log_dir, 'tensorboard'))
        args.writer = tb_logger
    else:
        args.writer = None
    
    # Create data loaders
    logger.info('Creating data loaders...')
    train_loader, test_loader = create_dataloaders(args)
    logger.info(f'Training samples: {len(train_loader)}')
    logger.info(f'Test samples: {len(test_loader.dataset)}')
    
    # Create models
    logger.info('Creating models...')
    logger.info(f'Architecture: {args.model}')
    logger.info(f'CBAM: {args.use_cbam}, Guidance Weight: {args.guidance_weight}')
    
    teacher_model = create_model(
        args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        use_cbam=args.use_cbam,
        guidance_weight=args.guidance_weight,
        dropout=args.dropout
    )
    
    student_model = create_model(
        args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        use_cbam=args.use_cbam,
        guidance_weight=args.guidance_weight,
        dropout=args.dropout
    )
    
    # Move models to device
    teacher_model = teacher_model.to(args.device)
    student_model = student_model.to(args.device)
    
    # Multi-GPU
    if args.n_gpu > 1:
        logger.info(f'Using {args.n_gpu} GPUs')
        teacher_model = nn.DataParallel(teacher_model)
        student_model = nn.DataParallel(student_model)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizers and schedulers
    teacher_optimizer, teacher_scheduler = create_optimizer_and_scheduler(
        teacher_model, args, is_teacher=True
    )
    student_optimizer, student_scheduler = create_optimizer_and_scheduler(
        student_model, args, is_teacher=False
    )
    
    # Load checkpoint if resuming
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=args.device)
        student_model.load_state_dict(checkpoint['student_state_dict'])
        teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
        student_optimizer.load_state_dict(checkpoint['student_optimizer'])
        teacher_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resumed from epoch {start_epoch}')
    
    # Create trainer
    trainer = MPLTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        teacher_optimizer=teacher_optimizer,
        student_optimizer=student_optimizer,
        teacher_scheduler=teacher_scheduler,
        student_scheduler=student_scheduler,
        args=args,
        logger=logger
    )
    
    # Train or evaluate
    if args.eval_only:
        logger.info('Evaluation mode')
        val_loss, val_acc = trainer.validate(0)
        logger.info(f'Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%')
    else:
        # Start training
        trainer.fit()
    
    # Close loggers
    logger.info('Training finished!')
    if args.tensorboard:
        tb_logger.close()


if __name__ == '__main__':
    main()
