import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from configs.default_config import get_default_config, update_config
from models import create_model
from data.dataset import LabeledDataset, load_data_list
from data.transforms import get_val_transform
from utils import Logger, ConfusionMatrix, MetricCalculator, accuracy

def test_model(model, test_loader, args, logger):

    model.eval()

    metric_calculator = MetricCalculator(
        num_classes=args.num_classes,
        class_names=[f'Class_{i}' for i in range(args.num_classes)]
    )

    all_predictions = []
    all_targets = []
    all_probs = []

    logger.info('Running evaluation...')

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='Testing'):
            images = images.to(args.device)
            targets = targets.to(args.device)

            outputs = model(images)

            metric_calculator.update(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = metric_calculator.compute()

    metric_calculator.summary()

    if hasattr(args, 'output_dir') and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        metric_calculator.confusion_matrix.plot(cm_path, normalize=True)
        logger.info(f'Confusion matrix saved to {cm_path}')

    return metrics, all_predictions, all_targets, all_probs

def generate_predictions(model, test_loader, args, logger):

    model.eval()

    predictions = []
    image_paths = []

    logger.info('Generating predictions...')

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc='Predicting')):
            images = images.to(args.device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            predictions.extend(preds.cpu().numpy())

    if hasattr(args, 'output_file') and args.output_file:
        with open(args.output_file, 'w') as f:
            f.write('image_id,prediction\n')
            for idx, pred in enumerate(predictions):
                f.write(f'{idx},{pred}\n')
        logger.info(f'Predictions saved to {args.output_file}')

    return predictions

def main():

    parser = get_default_config()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='directory to save test results')
    parser.add_argument('--output_file', type=str, default='./predictions.csv',
                        help='file to save predictions')
    parser.add_argument('--save_predictions', action='store_true',
                        help='save predictions to file')

    args = parser.parse_args()
    args = update_config(args)

    logger = Logger(args.log_dir, log_name='test.log')
    logger.info('='*60)
    logger.info('Model Testing')
    logger.info('='*60)

    logger.info('Configuration:')
    logger.info(f'  Checkpoint: {args.checkpoint}')
    logger.info(f'  Model: {args.model}')
    logger.info(f'  Device: {args.device}')
    logger.info('='*60)

    logger.info('Loading test data...')
    test_list = load_data_list(args.test_file)
    test_transform = get_val_transform(
        img_size=args.img_size,
        mean=args.mean,
        std=args.std
    )

    test_dataset = LabeledDataset(
        args.data_root, test_list, transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f'Test samples: {len(test_dataset)}')

    logger.info('Creating model...')
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        pretrained=False,
        use_cbam=args.use_cbam,
        guidance_weight=args.guidance_weight,
        dropout=args.dropout
    )

    logger.info(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    if 'student_state_dict' in checkpoint:
        state_dict = checkpoint['student_state_dict']
    elif 'teacher_state_dict' in checkpoint:
        state_dict = checkpoint['teacher_state_dict']
    else:
        state_dict = checkpoint['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(args.device)

    if args.n_gpu > 1:
        logger.info(f'Using {args.n_gpu} GPUs')
        model = nn.DataParallel(model)

    metrics, predictions, targets, probs = test_model(
        model, test_loader, args, logger
    )

    if args.save_predictions:
        generate_predictions(model, test_loader, args, logger)

    logger.info('='*60)
    logger.info('Testing completed!')
    logger.info(f'Overall Accuracy: {metrics["overall_accuracy"]*100:.2f}%')
    logger.info(f'Mean F1 Score: {metrics["mean_f1"]:.4f}')
    logger.info('='*60)

if __name__ == '__main__':
    main()
