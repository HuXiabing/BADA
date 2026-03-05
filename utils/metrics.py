import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class MapeLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')
        self.epsilon = epsilon

    def forward(self, output, target):
        loss = self.loss(output, target) / (torch.abs(target) + self.epsilon)
        return loss

class BatchResult:

    def __init__(self):
        self.batch_len = 0
        self.measured = []
        self.prediction = []
        self.index = []
        self.losses = []
        self.loss_sum = 0

    @property
    def loss(self):  # average loss of each sample
        if self.batch_len == 0:
            return float('nan')
        return self.loss_sum / self.batch_len

    def __iadd__(self, other):

        self.batch_len += other.batch_len
        self.measured.extend(other.measured)
        self.prediction.extend(other.prediction)
        self.index.extend(other.index)
        self.losses.extend(other.lossses)
        self.loss_sum += other.loss_sum

        return self

    def add_sample(self, prediction, measured, loss, index):

        self.batch_len += 1
        self.prediction.append(prediction)
        self.measured.append(measured)
        self.losses.append(loss)
        self.loss_sum += loss
        self.index.append(index)

    def get_sample_loss(self):
        df = pd.DataFrame({"idx": self.index, "loss": self.losses, "measured": self.measured, "prediction": self.prediction})
        df = df.sort_values('loss', ascending=False)
        return df

    def compute_metrics(self, tolerance=25):
        y_true = np.array(self.measured)
        y_pred = np.array(self.prediction)

        metrics = {}
        metrics["accuracy25"] = compute_accuracy(y_true, y_pred, 25)
        metrics["accuracy10"] = compute_accuracy(y_true, y_pred, 10)
        metrics["accuracy5"] = compute_accuracy(y_true, y_pred, 5)
        metrics["loss"] = self.loss

        return {
            "loss": metrics["loss"],
            "accuracy25": metrics["accuracy25"],
            "accuracy10": metrics["accuracy10"],
            "accuracy5": metrics["accuracy5"]
        }

def correct_regression(pred, answer, tolerance=25):
    """
    Calculate the correctness rate of regression predictions

    Args:
        pred: Predicted values --> tensor
        answer: True values --> tensor
        tolerance: Tolerance percentage, default 10%

    Returns:
        Number of correct predictions
    """
    if isinstance(pred, list):
        pred = torch.tensor(pred)
    if isinstance(answer, list):
        answer = torch.tensor(answer)

    percentage = torch.abs(pred - answer) * 100.0 / (torch.abs(answer) + 1e-3)
    return torch.sum(percentage < tolerance).item()

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tolerance=25) -> float:
    """
    Calculate the accuracy of predictions

    Args:
        y_true:
        y_pred:
        tolerance: Tolerance percentage, default 10%

    Returns:
        Accuracy (float between 0 and 1)
    """

    correct_count = correct_regression(torch.tensor(y_pred), torch.tensor(y_true), tolerance)
    total_count = len(y_true)

    return correct_count / total_count if total_count > 0 else 0.0


