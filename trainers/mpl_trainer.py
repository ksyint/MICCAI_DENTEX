import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
import time
from tqdm import tqdm

from utils.logger import AverageMeter
from utils.metrics import accuracy
from utils.ema import TeacherEMA

class MPLTrainer:

    def __init__(self, teacher_model, student_model, train_loader, val_loader,
                 criterion, teacher_optimizer, student_optimizer,
                 teacher_scheduler, student_scheduler, args, logger):

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.teacher_optimizer = teacher_optimizer
        self.student_optimizer = student_optimizer
        self.teacher_scheduler = teacher_scheduler
        self.student_scheduler = student_scheduler
        self.args = args
        self.logger = logger

        if args.teacher_ema > 0:
            self.teacher_ema = TeacherEMA(
                teacher_model, decay=args.teacher_ema,
                device=args.device, warmup_steps=args.ema_warmup_steps
            )
        else:
            self.teacher_ema = None

        self.scaler = amp.GradScaler(enabled=args.amp)

        self.global_step = 0
        self.best_acc = 0.0
        self.start_epoch = 0

    def train_epoch(self, epoch):

        self.teacher_model.train()
        self.student_model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_l = AverageMeter()
        losses_u = AverageMeter()
        losses_mpl = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            data_time.update(time.time() - end)

            labeled_img = batch['labeled_img'].to(self.args.device)
            labels = batch['label'].to(self.args.device)
            unlabeled_weak = batch['unlabeled_weak'].to(self.args.device)
            unlabeled_strong = batch['unlabeled_strong'].to(self.args.device)

            batch_size = labeled_img.size(0)

            self.student_optimizer.zero_grad()

            with amp.autocast(enabled=self.args.amp):

                student_logits_l = self.student_model(labeled_img)
                loss_l = self.criterion(student_logits_l, labels)

                with torch.no_grad():
                    if self.teacher_ema is not None:
                        teacher_logits_u = self.teacher_ema(unlabeled_weak)
                    else:
                        teacher_logits_u = self.teacher_model(unlabeled_weak)

                    teacher_probs_u = F.softmax(teacher_logits_u, dim=1)

                    max_probs, pseudo_labels = torch.max(teacher_probs_u, dim=1)

                    mask = max_probs.ge(self.args.threshold).float()

                student_logits_u = self.student_model(unlabeled_strong)
                loss_u = (self.criterion(student_logits_u, pseudo_labels) * mask).mean()

                loss_student = loss_l + self.args.lambda_u * loss_u

            self.scaler.scale(loss_student).backward()
            self.scaler.step(self.student_optimizer)
            self.scaler.update()

            if self.args.use_mpl:
                self.teacher_optimizer.zero_grad()

                with amp.autocast(enabled=self.args.amp):

                    teacher_logits_u2 = self.teacher_model(unlabeled_weak)
                    teacher_probs_u2 = F.softmax(teacher_logits_u2, dim=1)
                    max_probs2, pseudo_labels2 = torch.max(teacher_probs_u2, dim=1)
                    mask2 = max_probs2.ge(self.args.threshold).float()

                    with torch.no_grad():
                        student_logits_u2 = self.student_model(unlabeled_strong)

                    loss_u_mpl = (self.criterion(student_logits_u2, pseudo_labels2) * mask2).mean()

                    student_logits_l2 = self.student_model(labeled_img)
                    loss_l_mpl = self.criterion(student_logits_l2, labels)

                    loss_mpl = loss_l_mpl

                self.scaler.scale(loss_mpl).backward()
                self.scaler.step(self.teacher_optimizer)
                self.scaler.update()

                losses_mpl.update(loss_mpl.item(), batch_size)

            if self.teacher_ema is not None:
                self.teacher_ema.update(self.teacher_model, self.global_step)

            losses_l.update(loss_l.item(), batch_size)
            losses_u.update(loss_u.item(), batch_size)

            acc1, acc5 = accuracy(student_logits_l, labels, topk=(1, 5))
            top1.update(acc1[0].item(), batch_size)
            top5.update(acc5[0].item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            self.global_step += 1

            pbar.set_postfix({
                'L_loss': f'{losses_l.avg:.4f}',
                'U_loss': f'{losses_u.avg:.4f}',
                'MPL_loss': f'{losses_mpl.avg:.4f}',
                'Acc': f'{top1.avg:.2f}',
            })

            if hasattr(self.args, 'writer') and self.args.writer is not None:
                if self.global_step % self.args.log_interval == 0:
                    self.args.writer.add_scalar('train/loss_labeled', losses_l.avg, self.global_step)
                    self.args.writer.add_scalar('train/loss_unlabeled', losses_u.avg, self.global_step)
                    self.args.writer.add_scalar('train/loss_mpl', losses_mpl.avg, self.global_step)
                    self.args.writer.add_scalar('train/acc1', top1.avg, self.global_step)

        pbar.close()

        self.logger.info(
            f'Epoch: [{epoch}] '
            f'Time: {batch_time.sum:.2f}s '
            f'L_Loss: {losses_l.avg:.4f} '
            f'U_Loss: {losses_u.avg:.4f} '
            f'MPL_Loss: {losses_mpl.avg:.4f} '
            f'Acc@1: {top1.avg:.2f} '
            f'Acc@5: {top5.avg:.2f}'
        )

        return losses_l.avg, top1.avg

    def validate(self, epoch):

        self.student_model.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                batch_size = images.size(0)

                with amp.autocast(enabled=self.args.amp):
                    outputs = self.student_model(images)
                    loss = self.criterion(outputs, labels)

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                losses.update(loss.item(), batch_size)
                top1.update(acc1[0].item(), batch_size)
                top5.update(acc5[0].item(), batch_size)

                batch_time.update(time.time() - end)
                end = time.time()

        self.logger.info(
            f'Validation: '
            f'Time: {batch_time.sum:.2f}s '
            f'Loss: {losses.avg:.4f} '
            f'Acc@1: {top1.avg:.2f} '
            f'Acc@5: {top5.avg:.2f}'
        )

        if hasattr(self.args, 'writer') and self.args.writer is not None:
            self.args.writer.add_scalar('val/loss', losses.avg, epoch)
            self.args.writer.add_scalar('val/acc1', top1.avg, epoch)
            self.args.writer.add_scalar('val/acc5', top5.avg, epoch)

        return losses.avg, top1.avg

    def save_checkpoint(self, epoch, is_best=False):

        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'student_state_dict': self.student_model.state_dict(),
            'teacher_state_dict': self.teacher_model.state_dict(),
            'student_optimizer': self.student_optimizer.state_dict(),
            'teacher_optimizer': self.teacher_optimizer.state_dict(),
            'best_acc': self.best_acc,
        }

        if self.teacher_ema is not None:
            state['teacher_ema'] = self.teacher_ema.state_dict()

        checkpoint_path = f'{self.args.checkpoint_dir}/checkpoint_epoch_{epoch}.pth'
        torch.save(state, checkpoint_path)
        self.logger.info(f'Checkpoint saved: {checkpoint_path}')

        if is_best:
            best_path = f'{self.args.checkpoint_dir}/best_model.pth'
            torch.save(state, best_path)
            self.logger.info(f'Best model saved: {best_path}')

    def fit(self):

        self.logger.info('='*60)
        self.logger.info('Starting Training')
        self.logger.info('='*60)

        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss, train_acc = self.train_epoch(epoch)

            val_loss, val_acc = self.validate(epoch)

            self.teacher_scheduler.step()
            self.student_scheduler.step()

            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc

            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(epoch, is_best)

            self.logger.info(f'Best Acc@1: {self.best_acc:.2f}')
            self.logger.info('-'*60)

        self.logger.info('='*60)
        self.logger.info('Training Completed!')
        self.logger.info(f'Best Validation Accuracy: {self.best_acc:.2f}')
        self.logger.info('='*60)
