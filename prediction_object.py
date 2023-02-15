from typing import Any, List, Optional
from dataclasses import dataclass


@dataclass
class SeqPredictionObject:

    preds: List[int]
    pred_prob: Optional[List[float]]
    example: Optional[Any]

    def __post_init__(self):
        if self.pred_prob is not None:
            assert len(self.pred_prob) == len(self.preds)
