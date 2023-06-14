from modeling.metrics.base_metrics import Metric


class AccuracyMetric(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, labels, predictions):
        accuracy = labels.eq(predictions).float().mean()
        return (accuracy,)
