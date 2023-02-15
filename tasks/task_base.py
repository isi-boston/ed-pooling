import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dataclasses import dataclass
import string

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from prediction_object import SeqPredictionObject

import logging


@dataclass
class Guid:
    """
    A unique identifier for the example
    """
    sent_id: str
    doc_id: str


class InputExample:
    """
    A single training/test example for token classification.
    Args:
    guid: Unique id for the example.
    words: list. The words of the sequence.
    labels: list. The labels for each word of the sequence. This should be
    specified for train and dev examples, **and** for test examples.
    For test examples, all labels should be O.

    guid: Guid(some_sent_id, some_doc_id)
    | tokens     | labels             |
    | ---------- | ------------------ |
    | Dozens     |  O                 |
    | of         |  O                 |
    | Filipino   |  O                 |
    | protestors |  B-Conduct-Protest |
    | rallied    |  B-Conduct-Protest |
    | on         |  O                 |
    | Tuesday    |  O                 |

    1) For labels, note that the BIO is attached
    2) For labels, note that hyphen in root is not collapsed.
       B-Conduct-Protest and not B-ConductProtest.
    3) Note that [CLS]/[SEP] is absent.
    """
    def __init__(
            self,
            guid: Guid,
            words: List[str],
            labels: List[str],
            skip_transformer_tokenization: bool
    ):

        self.guid = guid

        # self.words and self.labels can change if augment is called
        self.words = words
        self.labels = labels

        # Whether to skip PreTrainedTokenizer tokenization or not
        self.skip_transformer_tokenization = skip_transformer_tokenization

        # self.orig_words and self.orig_labels cannot change
        self.orig_words: Tuple[str] = tuple(words)
        self.orig_labels: Tuple[str] = tuple(labels)

        # keep track of edits
        self.edits = {
            'insertions': [],
            'deletions': [],
            'substitutions': []
        }

        assert self.is_valid(), \
            f"len(self.words) = {len(self.words)}, len(self.labels) = {len(self.labels)}"

    def is_valid(self) -> bool:
        return len(self.words) == len(self.labels) \
               and len(self.orig_words) == len(self.orig_labels)

    def augment_add_punctuation(self) -> None:

        if self.words[-1] not in string.punctuation:
            # Add full stop
            self.words.append('.')
            self.labels.append('O')
            self.edits['insertions'].append(len(self.words) - 1)
        assert self.is_valid()

    def deaugment(
            self,
            seq: Optional[List[Any]] = None
    ) -> None:

        # If self is not augmented, the call to this function should be a no-op.

        if seq is not None:

            # Depending upon self.edits,
            # change seq so that any additions/deletions are removed from seq

            # Handle deletions
            if self.edits['deletions']:
                raise NotImplementedError

            # Handle substitutions
            if self.edits['substitutions']:
                raise NotImplementedError

            # Handle insertions
            sorted_idx = sorted(self.edits['insertions'], reverse=True)  # descending order
            assert len(set(sorted_idx)) == len(sorted_idx)  # no duplicates
            for idx in sorted_idx:
                # It is not necessary that len(seq) is same as len(self.words).
                # Hence try/except.
                try:
                    seq.pop(idx)
                except IndexError:
                    continue

        # Revert to the original sequence
        self.words = list(self.orig_words)
        self.labels = list(self.orig_labels)
        assert self.is_valid()


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    | tokens   | input_ids | attention_mask | token_type_ids | label_ids | word_offsets |
    | -------- | --------- | -------------- | -------------- | --------- | ------------ |
    | [CLS]    | 101       | 1              | 0              | -100      |              |
    | Do       | 2091      | 1              | 0              | 144       | (1, 3)       |
    | ##zen    | 10947     | 1              | 0              | -100      |              |
    | ##s      | 1116      | 1              | 0              | -100      |              |
    | of       | 1104      | 1              | 0              | 144       | (4, 4)       |
    | Filipino | 10121     | 1              | 0              | 144       | (5, 5)       |
    | protest  | 5641      | 1              | 0              | 4         | (6, 7)       |
    | ##ors    | 3864      | 1              | 0              | -100      |              |
    | rallied  | 27429     | 1              | 0              | 4         | (8, 8)       |
    | on       | 1113      | 1              | 0              | 144       | (9, 9)       |
    | Tuesday  | 9667      | 1              | 0              | 144       | (10, 10)     |
    | [SEP]    | 102       | 1              | 0              | -100      |              |
    | [PAD]    | 0         | 0              | 0              | -100      |              |
    | [PAD]    | 0         | 0              | 0              | -100      |              |
    | [PAD]    | 0         | 0              | 0              | -100      |              |

    For BetterBasicTriggerTask, 144 = 'O' and 4 = 'B-Conduct-Protest'.
    At test time, label_ids will be either ignore_idx (-100) or 'O' (144).

    word_offsets:
    len(word_offsets) = number of words in the original sentence and not number of bpe tokens.
    Also len(word_offsets) = number of labels which are not -100
    There are no word_offsets for CLS/SEP/PAD.
    """
    guid: Guid
    input_ids: List[int]
    attention_mask: List[int]
    example: InputExample
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    word_offsets: Optional[List[Tuple[int, int]]] = None
    token_weights:  Optional[List[List[float]]] = None


class TokenClassificationTask:

    def read_examples_from_file(
            self,
            filename: str,
            augment: bool
    ) -> List[InputExample]:
        raise NotImplementedError

    def write_predictions_to_file(
            self,
            predictions: List[SeqPredictionObject],
            orig_filename: str,
            out_filename: str
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def get_token_weights_morph(
            token_scores,
            words: List[str],
            tokens: List[str],
            word_offsets: List[Tuple[int, int]],
    ) -> List[List[float]]:
        scores = []
        assert len(words) == len(word_offsets)
        for word, offset in zip(words, word_offsets):
            token_span = tokens[offset[0]: offset[1] + 1]
            if word not in token_scores:
                logging.warning(f"{word} not in token_scores")
                scores.append([1. / len(token_span) for _ in token_span])
            elif not token_scores[word]['tokens']:
                logging.warning(f"{token_span} for word {word} not in token_scores")
                assert len(token_span) == 1
                scores.append([1. / len(token_span) for _ in token_span])
            else:
                assert len(token_span) == len(token_scores[word]['tokens']), \
                    f"{token_span}, {token_scores[word]['tokens']}"
                assert all(
                    [
                        m == t for m, t in
                        zip(token_scores[word]['tokens'], tokens[offset[0]: offset[1] + 1])
                    ]
                )
                scores.append(token_scores[word]['flags'])
        return scores

    @staticmethod
    def get_token_weights(
            token_scores: Dict[str, float],
            tokens: List[str],
            word_offsets: List[Tuple[int, int]],
            default_token_score: float,
            temperature: float = 1.
    ) -> List[List[float]]:

        """
        token_scores: {'police': 5, 'attack': 10, 'ed': 1, 'the': 0.5, 'mob': 7, 'tree': 4}
        tokens: ['police', 'attack', 'ed', 'the', 'mob' ]
        word_offsets: [(0, 0), (1, 2), (3, 3), (4, 4)]
        default_token_score: 10.0

        returns: [[1.], [0.99, 0.01], [1.], [1.]]
        """

        weights = []
        for offset in word_offsets:
            word_scores = []
            for tok in tokens[offset[0]: offset[1] + 1]:
                if tok in token_scores:
                    word_scores.append(token_scores[tok])
                else:
                    logging.debug(f"{tok} not present in token_scores")
                    word_scores.append(default_token_score)
            word_weights = [np.exp(s / temperature) for s in word_scores]
            z = sum(word_weights)
            weights.append([w / z for w in word_weights])

        assert len(weights) == len(word_offsets)
        assert sum([len(ww) for ww in weights]) == len(tokens)
        return weights

    @staticmethod
    def get_tokens_labelids_wordoffsets(
            words: List[str],
            labels: List[str],
            label_list: List[str],
            pad_token_label_id: int,
            tokenizer: PreTrainedTokenizer,
            skip_transformer_tokenization: bool
    ):
        label_map = {label: i for i, label in enumerate(label_list)}
        
        if skip_transformer_tokenization:
            tokens = words
            label_ids = [label_map[label] for label in labels]
            word_offsets = [(i, i) for i in range(len(words))]
            return tokens, label_ids, word_offsets

        tokens = []
        label_ids = []
        word_offsets = []
        for word, label in zip(words, labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([])
            # when calling tokenize with just a space.
            if len(word_tokens) > 0:
                word_offsets.append(
                    (len(tokens), len(tokens) + len(word_tokens) - 1)  # inclusive
                )
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word,
                # and padding ids for the remaining tokens
                label_ids.extend(
                    [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
                )
            else:
                # For wordpiece models, ' ' gets tokenized to ''
                word_offsets.append(
                    (len(tokens), len(tokens))  # inclusive
                )
                tokens.append(tokenizer.unk_token)
                label_ids.append(label_map[label])
                logging.warning(f'word = **{word}** got tokenized to empty string')
        return tokens, label_ids, word_offsets

    def convert_examples_to_features(
        self,
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        pooling_strategy: str,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=torch.nn.CrossEntropyLoss().ignore_index,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        token_scores: Optional[Any] = None,
        default_token_score: Optional[float] = None,
        token_scores_temperature: Optional[float] = None
    ) -> List[InputFeatures]:
        """Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` is the segment id associated to the CLS (0 for BERT, 2 for XLNet)
        """

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logging.info(f"Writing example {ex_index} of {len(examples)}")

            tokens, label_ids, word_offsets = self.get_tokens_labelids_wordoffsets(
                words=example.words,
                labels=example.labels,
                label_list=label_list,
                pad_token_label_id=pad_token_label_id,
                tokenizer=tokenizer,
                skip_transformer_tokenization=example.skip_transformer_tokenization
            )

            if pooling_strategy == 'morph':
                token_weights = self.get_token_weights_morph(
                    token_scores=token_scores,
                    words=example.words,
                    tokens=tokens,
                    word_offsets=word_offsets
                )
            else:
                # If not pooling_strategy != morph, the token_weights will be used only for idf.
                # Populating for all strategies anyways.
                token_weights = self.get_token_weights(
                    token_scores=token_scores if token_scores is not None else {},
                    tokens=tokens,
                    word_offsets=word_offsets,
                    default_token_score=default_token_score if default_token_score is not None else 1.,
                    temperature=token_scores_temperature if token_scores_temperature is not None else 1.
                )

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                trimmed_token_weights = []
                trimmed_word_offsets = []
                assert len(token_weights) == len(word_offsets)
                for token_weight, word_offset in zip(token_weights, word_offsets):
                    if word_offset[0] >= max_seq_length - special_tokens_count:
                        break
                    elif word_offset[1] >= max_seq_length - special_tokens_count:
                        trimmed_word_offsets.append(
                            (word_offset[0], max_seq_length - special_tokens_count - 1)
                        )
                        tokens_in_last_word \
                            = trimmed_word_offsets[-1][1] - trimmed_word_offsets[-1][0] + 1
                        trimmed_token_weights.append(
                            [token_weight[idx] for idx in range(tokens_in_last_word)]
                        )
                        break
                    else:
                        trimmed_word_offsets.append(word_offset)
                        trimmed_token_weights.append(token_weight)
                word_offsets = trimmed_word_offsets
                token_weights = trimmed_token_weights
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                word_offsets = [(s + 1, e + 1) for s, e in word_offsets]
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                word_offsets = [(s + padding_length, e + padding_length) for s, e in word_offsets]
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert sum([lab != pad_token_label_id for lab in label_ids]) == len(word_offsets)
            assert len(token_weights) == len(word_offsets)
            assert sum([wo[1] - wo[0] + 1 for wo in word_offsets]) \
                   == sum([len(tw) for tw in token_weights])

            if ex_index < 5:
                logging.debug("*** Example ***")
                logging.debug("guid: %s", example.guid)
                logging.debug("tokens: %s", " ".join([str(x) for x in tokens]))
                logging.debug("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logging.debug("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logging.debug("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logging.debug("label_ids: %s", " ".join([str(x) for x in label_ids]))
                logging.debug("word_offsets: %s", " ".join([str(x) for x in word_offsets]))
                logging.debug("token_weights: %s", " ".join([str(x) for x in token_weights]))

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            features.append(
                InputFeatures(
                    guid=example.guid,
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    example=example,
                    token_type_ids=segment_ids,
                    label_ids=label_ids,
                    word_offsets=word_offsets,
                    token_weights=token_weights
                )
            )
        return features


class TokenClassificationDataset(Dataset):

    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            data_filenames: List[str],
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            pooling_strategy: str,
            max_seq_length: Optional[int] = None,
            augment: bool = False,
            token_scores_file: Optional[str] = None,
            default_token_score: Optional[float] = None,
            token_scores_temperature: Optional[float] = None
    ):

        self.examples = []
        for data_filename in data_filenames:
            self.examples.extend(
                token_classification_task.read_examples_from_file(data_filename, augment)
            )

        token_scores = None
        if pooling_strategy == 'idf':
            assert default_token_score is not None
            token_scores = {}
            with open(token_scores_file, encoding='utf-8') as f:
                for line in f:
                    fields = line.rstrip().split('\t')
                    assert len(fields) == 2
                    token, score = fields
                    token_scores[token] = float(score)
        elif pooling_strategy == 'morph':
            token_scores = json.load(open(token_scores_file, encoding='utf-8'))

        self.features = token_classification_task.convert_examples_to_features(
            self.examples,
            labels,
            max_seq_length,
            tokenizer,
            pooling_strategy=pooling_strategy,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
            token_scores=token_scores,
            default_token_score=default_token_score,
            token_scores_temperature=token_scores_temperature
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def collate_fn(batch: List[InputFeatures]):
    num_words = [len(feature.word_offsets) for feature in batch]
    max_number_words = max(num_words)
    word_offsets = torch.tensor([
        feature.word_offsets + [(0, 0)] * (max_number_words - nw)
        for nw, feature in zip(num_words, batch)
    ], dtype=torch.long)

    # pad for additional words
    token_weights = [
        feature.token_weights + [[0.]] * (max_number_words - nw)
        for nw, feature in zip(num_words, batch)
    ]
    # pad for additional tokens
    max_tokens = max([max([len(tw) for tw in feature.token_weights]) for feature in batch])
    token_weights = [
        [tw + [0.] * (max_tokens - len(tw)) for tw in seq_token_weights]
        for seq_token_weights in token_weights
    ]
    token_weights = torch.tensor(token_weights, dtype=torch.float32)

    return (
        {
            "input_ids":
                torch.tensor([feature.input_ids for feature in batch], dtype=torch.long),
            "attention_mask":
                torch.tensor([feature.attention_mask for feature in batch], dtype=torch.long),
            "token_type_ids": None if batch[0].token_type_ids is None else
            torch.tensor([feature.token_type_ids for feature in batch], dtype=torch.long),
            "labels":
                torch.tensor([feature.label_ids for feature in batch], dtype=torch.long),
            "word_offsets": word_offsets,
            "token_weights": token_weights,
            "examples":
                [feature.example for feature in batch],
        }
    )
