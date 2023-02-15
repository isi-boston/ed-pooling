from typing import Dict, List
import json
from tasks.task_labels import get_bio_labels, convert_ids_to_labels, MinionLabels
from tasks.task_base import TokenClassificationTask
from tasks.task_base import InputExample, Guid
from prediction_object import SeqPredictionObject


class TriggerClassificationMinionTask(TokenClassificationTask):

    def __init__(self):
        self.labels = get_bio_labels(MinionLabels)
        self.label_map = {i: label for i, label in enumerate(self.labels)}

    def get_labels(self) -> List[str]:
        return self.labels

    def get_label_map(self) -> Dict[int, str]:
        return self.label_map

    def read_examples_from_file(
            self,
            file_path: str,
            augment: bool
    ) -> List[InputExample]:

        examples = []
        with open(file_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                entry = json.loads(line)
                words = entry["tokens"]
                labels = []
                for lab in entry["labels"]:
                    if lab == 'O':
                        labels.append(lab)
                    elif lab[:2] == 'B_':
                        labels.append(f"B-{lab[2:]}")
                    elif lab[:2] == 'I_':
                        labels.append(f"I-{lab[2:]}")
                    else:
                        raise ValueError

                guid = Guid(sent_id=str(line_idx), doc_id=str(line_idx))
                examples.append(
                    InputExample(
                        words=words,
                        labels=labels,
                        guid=guid,
                        skip_transformer_tokenization=False
                    )
                )
        return examples

    def write_predictions_to_file(
            self,
            predictions: List[SeqPredictionObject],
            orig_filename: str,
            out_filename: str
    ) -> None:

        examples = [p.example for p in predictions]
        prediction_ids = [p.preds for p in predictions]
        prediction_labels = convert_ids_to_labels(
            label_map=self.get_label_map(),
            ids=prediction_ids
        )

        assert len(examples) == len(prediction_labels)
        with open(out_filename, "w", encoding="utf-8") as f:
            for example, pred_labels in zip(examples, prediction_labels):

                labels = []
                for lab in pred_labels:
                    if lab == 'O':
                        labels.append(lab)
                    elif lab[:2] == 'B-':
                        labels.append(f"B_{lab[2:]}")
                    elif lab[:2] == 'I-':
                        labels.append(f"I_{lab[2:]}")
                    else:
                        raise ValueError

                assert len(labels) <= len(example.words)
                labels += ['O'] * (len(example.words) - len(labels))
                entry = {"tokens": example.words, "labels": labels}
                print(json.dumps(entry), file=f)
