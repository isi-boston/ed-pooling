from typing import Any, Dict, List, Optional
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)


class TokenClassificationMetrics:
    def __init__(
            self,
            acc_score: float,
            precision: float,
            recall: float,
            f1: float,
            class_report: Any,
            loss: Optional[float] = None
    ):
        self.accuracy_score = acc_score
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.classification_report = class_report
        self.loss = loss

    def is_better_than(self, other) -> bool:
        return self.f1 > other.f1

    def __repr__(self) -> str:
        r = f'accuracy = {self.accuracy_score}\n'
        r += f'precision = {self.precision}\n'
        r += f'recall = {self.recall}\n'
        r += f'f1 = {self.f1}\n'
        r += f'loss = {self.loss}'
        r += f'{self.classification_report}\n'
        return r


class TokenClassificationMetricsCalculator:

    def __init__(self, label_map: Dict[int, str]):
        self.label_map = label_map

    @staticmethod
    def _make_preds_same_length_as_gold_labels(
            prediction_labels: List[List[str]],
            gold_labels: List[List[str]]
    ) -> List[List[str]]:

        padded_pred_labels = []
        for pred, gold in zip(prediction_labels, gold_labels):
            assert len(pred) <= len(gold)
            pred = pred + ['O'] * (len(gold) - len(pred))
            padded_pred_labels.append(pred)
        return padded_pred_labels

    @staticmethod
    def _remove_root_hyphens(labels: List[str]) -> List[str]:
        """B-Violence-Attack -> B-ViolenceAttack
        Need this because seqeval splits on '-' and takes only the last split.
        If we do not remove the hyphens, in the above example, seqeval will see B-Attack
        """
        labels_no_hyphens = []
        for label in labels:
            if '-' in label:
                bi, label_root = label.split('-', maxsplit=1)
                label_root = label_root.replace('-', '')
                label = f'{bi}-{label_root}'
            labels_no_hyphens.append(label)
        return labels_no_hyphens

    def __call__(
            self,
            prediction_ids: List[List[int]],
            examples
    ) -> TokenClassificationMetrics:

        # note that with this scheme, sub-tokens cannot be labeled
        gold_labels = [ex.labels for ex in examples]
        gold_labels = [
            TokenClassificationMetricsCalculator._remove_root_hyphens(
                l
            ) for l in gold_labels
        ]

        prediction_labels = [[self.label_map[idx] for idx in l] for l in prediction_ids]
        prediction_labels = [
            TokenClassificationMetricsCalculator._remove_root_hyphens(
                l
            ) for l in prediction_labels
        ]
        prediction_labels = \
            TokenClassificationMetricsCalculator._make_preds_same_length_as_gold_labels(
                prediction_labels,
                gold_labels
            )

        return TokenClassificationMetrics(
            acc_score=accuracy_score(gold_labels, prediction_labels),
            precision=precision_score(gold_labels, prediction_labels),
            recall=recall_score(gold_labels, prediction_labels),
            f1=f1_score(gold_labels, prediction_labels),
            class_report=classification_report(gold_labels, prediction_labels)
        )
