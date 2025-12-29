"""Metrics for LFA-Net evaluation using torchmetrics."""

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryRecall,
    BinarySpecificity,
)


def get_metrics() -> MetricCollection:
    """
    Get a collection of metrics for binary segmentation evaluation.

    Returns:
        MetricCollection with dice, iou, sensitivity, specificity
    """
    return MetricCollection({
        "dice": BinaryF1Score(),
        "iou": BinaryJaccardIndex(),
        "sensitivity": BinaryRecall(),
        "specificity": BinarySpecificity(),
    })
