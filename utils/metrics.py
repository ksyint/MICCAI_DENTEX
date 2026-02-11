import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ConfusionMatrix:

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        cm = confusion_matrix(targets, preds, labels=np.arange(self.num_classes))
        self.matrix += cm

    def compute(self):

        per_class_acc = np.diag(self.matrix) / (self.matrix.sum(axis=1) + 1e-10)

        overall_acc = np.diag(self.matrix).sum() / (self.matrix.sum() + 1e-10)

        precision = np.diag(self.matrix) / (self.matrix.sum(axis=0) + 1e-10)
        recall = np.diag(self.matrix) / (self.matrix.sum(axis=1) + 1e-10)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return {
            'overall_accuracy': overall_acc,
            'per_class_accuracy': per_class_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_f1': np.mean(f1),
        }

    def plot(self, save_path=None, normalize=False):

        matrix = self.matrix.astype('float')
        if normalize:
            matrix = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-10)

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.2f' if normalize else 'd',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cmap='Blues', cbar=True)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def summary(self):

        metrics = self.compute()

        print("\n" + "="*60)
        print("Classification Metrics Summary")
        print("="*60)
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Mean F1 Score: {metrics['mean_f1']:.4f}")
        print("\nPer-Class Metrics:")
        print("-"*60)
        print(f"{'Class':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-"*60)

        for i, name in enumerate(self.class_names):
            print(f"{name:<15} {metrics['per_class_accuracy'][i]:<12.4f} "
                  f"{metrics['precision'][i]:<12.4f} {metrics['recall'][i]:<12.4f} "
                  f"{metrics['f1_score'][i]:<12.4f}")
        print("="*60 + "\n")

def calibration_error(probs, labels, num_bins=15):

    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bins[:-1])

    ece = 0.0
    for i in range(1, num_bins + 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            ece += mask.sum() / len(confidences) * abs(bin_accuracy - bin_confidence)

    return ece

class MetricCalculator:

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names
        self.confusion_matrix = ConfusionMatrix(num_classes, class_names)
        self.reset()

    def reset(self):
        self.confusion_matrix.reset()
        self.all_probs = []
        self.all_labels = []
        self.all_preds = []

    def update(self, outputs, labels):

        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.confusion_matrix.update(preds, labels)

        self.all_probs.append(probs.detach().cpu())
        self.all_labels.append(labels.detach().cpu())
        self.all_preds.append(preds.detach().cpu())

    def compute(self):

        cm_metrics = self.confusion_matrix.compute()

        all_probs = torch.cat(self.all_probs, dim=0)
        all_labels = torch.cat(self.all_labels, dim=0)
        ece = calibration_error(all_probs, all_labels)

        return {
            **cm_metrics,
            'calibration_error': ece,
        }

    def summary(self):

        self.confusion_matrix.summary()
        metrics = self.compute()
        print(f"Expected Calibration Error (ECE): {metrics['calibration_error']:.4f}\n")
