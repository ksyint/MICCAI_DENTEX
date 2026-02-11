import argparse

def get_default_config():

    parser = argparse.ArgumentParser(description='Semi-Supervised Tooth Classification with MPL')

    parser.add_argument('--data_root', type=str, default='./data/images',
                        help='root directory of images')
    parser.add_argument('--labeled_file', type=str, default='./data/labeled.txt',
                        help='file with labeled data')
    parser.add_argument('--unlabeled_file', type=str, default='./data/unlabeled.txt',
                        help='file with unlabeled data')
    parser.add_argument('--test_file', type=str, default='./data/test.txt',
                        help='file with test data')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size')

    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                'efficientnet_b3', 'efficientnet_b4',
                                'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'],
                        help='model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--use_cbam', action='store_true', default=True,
                        help='use CBAM attention modules')
    parser.add_argument('--guidance_weight', type=float, default=0.3,
                        help='weight for bottom-region guidance (0-1)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate')

    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for labeled data')
    parser.add_argument('--mu', type=int, default=7,
                        help='unlabeled batch size = batch_size * mu')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')

    parser.add_argument('--use_mpl', action='store_true', default=True,
                        help='use Meta Pseudo Label')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='confidence threshold for pseudo-labels')
    parser.add_argument('--lambda_u', type=float, default=1.0,
                        help='weight for unlabeled loss')
    parser.add_argument('--teacher_lr', type=float, default=0.03,
                        help='teacher learning rate')
    parser.add_argument('--teacher_ema', type=float, default=0.999,
                        help='EMA decay rate for teacher (0 to disable)')
    parser.add_argument('--ema_warmup_steps', type=int, default=0,
                        help='warmup steps for EMA')

    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'step', 'multistep'],
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--milestones', type=int, nargs='+', default=[150, 225],
                        help='milestones for MultiStepLR')

    parser.add_argument('--weak_aug', action='store_true', default=True,
                        help='use weak augmentation')
    parser.add_argument('--strong_aug', action='store_true', default=True,
                        help='use strong augmentation (RandAugment)')

    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU IDs to use (comma separated)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='use automatic mixed precision')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory to save logs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='log training stats every N steps')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')

    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint to resume from')

    parser.add_argument('--eval_only', action='store_true',
                        help='only evaluate without training')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')

    return parser

def update_config(args):

    import os
    import torch

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.device = torch.device('cuda')
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device('cpu')
        args.n_gpu = 0

    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]

    args.unlabeled_batch_size = args.batch_size * args.mu

    return args

if __name__ == '__main__':
    parser = get_default_config()
    args = parser.parse_args()
    args = update_config(args)

    print("Configuration:")
    print("-" * 60)
    for arg in vars(args):
        print(f"{arg:30s}: {getattr(args, arg)}")
    print("-" * 60)
