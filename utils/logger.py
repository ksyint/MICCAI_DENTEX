"""
Logger utilities for training
"""

import os
import sys
import logging
from datetime import datetime


class Logger:
    """
    Custom logger for training process
    Logs to both console and file
    """
    def __init__(self, log_dir='./logs', log_name=None, resume=False):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if log_name is None:
            log_name = f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        self.log_path = os.path.join(log_dir, log_name)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.logger.handlers = []
        
        mode = 'a' if resume else 'w'
        file_handler = logging.FileHandler(self.log_path, mode=mode)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        if not resume:
            self.logger.info(f"Logging to {self.log_path}")
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def append(self, message):
        """Append message (alias for info)"""
        self.info(str(message))
    
    def close(self):
        """Close logger handlers"""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, name='', fmt='.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricTracker:
    """
    Track multiple metrics over training
    """
    def __init__(self, *keys):
        self.metrics = {}
        for key in keys:
            self.metrics[key] = AverageMeter(name=key)
    
    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()
    
    def update(self, key, value, n=1):
        if key not in self.metrics:
            self.metrics[key] = AverageMeter(name=key)
        self.metrics[key].update(value, n)
    
    def avg(self, key):
        return self.metrics[key].avg
    
    def result(self):
        return {key: meter.avg for key, meter in self.metrics.items()}
    
    def __str__(self):
        result = []
        for key, meter in self.metrics.items():
            result.append(f"{key}: {meter.avg:.4f}")
        return " | ".join(result)


class TensorboardLogger:
    """
    Wrapper for TensorBoard logging
    """
    def __init__(self, log_dir='./runs', enabled=True):
        self.enabled = enabled
        self.log_dir = log_dir
        
        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("Warning: tensorboard not installed, logging disabled")
                self.enabled = False
                self.writer = None
        else:
            self.writer = None
    
    def add_scalar(self, tag, value, step):
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def add_scalars(self, main_tag, tag_scalar_dict, step):
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def add_image(self, tag, img_tensor, step):
        if self.enabled and self.writer is not None:
            self.writer.add_image(tag, img_tensor, step)
    
    def add_histogram(self, tag, values, step):
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        if self.enabled and self.writer is not None:
            self.writer.close()
