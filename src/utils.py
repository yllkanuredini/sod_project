import torch
import torch.nn as nn


def calculate_iou(predictions, targets, threshold=0.5, smooth=1e-6):
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()

    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


def calculate_precision_recall_f1(predictions, targets, threshold=0.5, smooth=1e-6):
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()

    true_positive = (predictions * targets).sum()
    false_positive = (predictions * (1 - targets)).sum()
    false_negative = ((1 - predictions) * targets).sum()

    precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
    recall = (true_positive + smooth) / (true_positive + false_negative + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return precision, recall, f1


class SODLoss(nn.Module):
    def __init__(self):
        super(SODLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        iou = calculate_iou(predictions, targets)
        iou_loss = 1 - iou

        total_loss = bce_loss + 0.5 * iou_loss
        return total_loss