import os
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
from importlib import import_module

import torch
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
)
from logging_utils import init_logging
from tasks.task_base import TokenClassificationDataset, collate_fn
from trainer import Trainer
from metrics import TokenClassificationMetricsCalculator
from model import TokenClassification


def get_tokenizer(model_args):

    pretrained_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        do_lower_case=model_args.do_lowercase
    )
    return pretrained_tokenizer


def get_model(model_args, training_args):

    model = TokenClassification(model_args=model_args).to(training_args.device)
    if training_args.continue_training:
        model.load_state_dict(
            torch.load(training_args.checkpoint_file, map_location=training_args.device),
            strict=True
        )
    elif training_args.transfer_learning:
        pretrained_dict = torch.load(
            training_args.transfer_model_file,
            map_location=training_args.device
        )
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'encoder' in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=True)
    elif not training_args.do_train and training_args.do_predict:
        model.load_state_dict(
            torch.load(training_args.best_model_file, map_location=training_args.device),
            strict=True
        )
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        metadata={"help": "bert/xlmr/bert-token-classification"}
    )
    task_type: Optional[str] = field(
        default="BetterAbstractTrigger",
        metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    pooling_strategy: str = field(
        default='first_token',
        metadata={
            "help": "Choice of pooling strategy. One of: "
                    " - first_token "
                    " - last_token "
                    " - average "
                    " - idf: provide default_token_score/token_scores_file/token_scores_temperature"
                    " - attention: provide token_scores_temperature"
                    " - morph: provide token_scores_file"
        }
    )
    default_token_score: Optional[float] = field(
        default=None,
        metadata={"help": "Score to be assigned to an token not seen while creating scores."}
    )
    token_scores_file: Optional[str] = field(
        default=None,
        metadata={"help": "Tab separated file (token\tscore) containing scores for tokens."}
    )
    token_scores_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature used in softmax to calculate token weights from idfs."}
    )
    do_lowercase: bool = field(
        default=False,
        metadata={"help": "If True, use lowercase when tokenization"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache dir to load huggingface models"}
    )

    def __post_init__(self):

        pooling_strategies = ['first_token', 'last_token', 'average', 'idf', 'attention', 'morph']
        assert self.pooling_strategy in pooling_strategies

        if self.pooling_strategy == 'idf':
            assert self.token_scores_file is not None
            assert self.default_token_score is not None
            assert self.token_scores_temperature is not None
            assert self.token_scores_temperature > 0

        if self.pooling_strategy == 'attention':
            assert self.token_scores_temperature is not None
            assert self.token_scores_temperature > 0

        if self.pooling_strategy == 'morph':
            assert self.token_scores_file is not None


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_files: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Training data file(s)"}
    )
    valid_files: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Validation data file(s)"}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Test data file"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    augment_data: bool = field(
        default=False,
        metadata={"help": "Indicate whether to alter the example or not"}
    )


@dataclass
class TrainArguments:
    """
    Arguments pertaining to training regiment.
    """
    output_dir: str = field(
        metadata={"help": "Dir where the trained model will be saved/read from"}
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "Indicate whether to train"}
    )
    continue_training: bool = field(
        default=False,
        metadata={"help": "Indicate whether to continue training"}
    )
    transfer_learning: bool = field(
        default=False,
        metadata={"help": "Whether to load pre-trained state-dict except the classifier layer"}
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Indicate whether to predict."}
    )
    num_train_epochs: int = field(
        default=5,
        metadata={"help": "Number of training epochs"}
    )
    train_batch_size: int = field(
        default=32,
        metadata={"help": "Training batch size"}
    )
    valid_batch_size: int = field(
        default=32,
        metadata={"help": "Validation batch size"}
    )
    predict_batch_size: int = field(
        default=32,
        metadata={"help": "Predict batch size"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for the AdamW optimizer"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of loss steps before gradients are updated"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradients with norm more than this value is scaled."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay."}
    )
    warmup_proportion: float = field(
        default=0.0,
        metadata={"help": "Proportion of the total number of training steps."}
    )
    freeze_embeddings: bool = field(
        default=False,
        metadata={"help": "If True, do not vary the embeddings of word/position/token"}
    )
    freeze_layers: str = field(
        default="",
        metadata={"help": "Specify which transformer layers to freeze during training. "
                          "Multiple layer-indexes should be separated by ','. Ex: '1,2,3'"}
    )
    bitfit: bool = field(
        default=False,
        metadata={
            "help": "If True, freeze all non-bias transformer parameters "
                    "See https://arxiv.org/pdf/2106.10199.pdf"
            }
    )

    checkpoint_file: str = field(
        default='checkpoint.pt',
        metadata={"help": "Trained model will be stored in output_dir/checkpoint_file"}
    )
    best_model_file: str = field(
        default='bestmodel.pt',
        metadata={
            "help": "Best model per validation score will be stored in output_dir/best_model_file"
        }
    )
    transfer_model_file: str = field(
        default='transfer.pt',
        metadata={"help": "Pre-trained model from which to start transfer_learning"}
    )
    preds_out_file: str = field(
        default='preds.json',
        metadata={
            "help": "If do_predict, predicted file will be stored in output_dir/preds_out_file"
        }
    )
    device: Optional[str] = field(
        default=None,
        metadata={
            "help": "cpu/cuda"
                    "If None, will be determined using no_cuda flag and availability of cuda"
        }
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed to pin random/numpy/torch"}
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Log level"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.device is None:
        training_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if training_args.do_train and training_args.continue_training:
        raise ValueError('Cannot simultaneously set do_train and continue_training.')

    if training_args.continue_training and training_args.transfer_learning:
        raise ValueError('Cannot simultaneously set continue_training and transfer_learning.')

    if training_args.do_train:
        if os.path.exists(training_args.output_dir):
            raise IOError(f'{training_args.output_dir} already exists')
        os.makedirs(training_args.output_dir)

    training_args.checkpoint_file = os.path.join(
        training_args.output_dir,
        training_args.checkpoint_file
    )
    training_args.best_model_file = os.path.join(
        training_args.output_dir,
        training_args.best_model_file
    )
    training_args.preds_out_file = os.path.join(
        training_args.output_dir,
        training_args.preds_out_file
    )

    # logging
    init_logging(os.path.join(training_args.output_dir, 'train.log'), training_args.log_level)
    logging.info(f'model_args: {model_args}')
    logging.info(f'data_args: {data_args}')
    logging.info(f'training_args: {training_args}')

    # log env info
    logging.info(f'Executable = {sys.executable}')

    # seed
    set_seed(training_args.seed)

    # task
    task_modules = [
        'tasks.task_trigger', 'tasks.task_minion',
    ]
    token_classification_task = None
    for mod in task_modules:
        try:
            module = import_module(mod)
            token_classification_task = getattr(module, model_args.task_type)()
        except AttributeError:
            continue
    labels = token_classification_task.get_labels()
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    model_args.num_labels = num_labels

    # cfg/tokenizer/model
    tokenizer = get_tokenizer(model_args)
    model = get_model(model_args, training_args)
    logging.info("Loaded model.")

    # metrics/trainer
    compute_metrics = TokenClassificationMetricsCalculator(label_map=label_map)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    if training_args.do_train or training_args.continue_training or training_args.transfer_learning:

        # dataset
        train_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_filenames=data_args.train_files,
            tokenizer=tokenizer,
            labels=labels,
            model_type=model_args.model_type,
            pooling_strategy=model_args.pooling_strategy,
            max_seq_length=data_args.max_seq_length,
            token_scores_file=model_args.token_scores_file,
            default_token_score=model_args.default_token_score,
            token_scores_temperature=model_args.token_scores_temperature
        )
        valid_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_filenames=data_args.valid_files,
            tokenizer=tokenizer,
            labels=labels,
            model_type=model_args.model_type,
            pooling_strategy=model_args.pooling_strategy,
            max_seq_length=data_args.max_seq_length,
            token_scores_file=model_args.token_scores_file,
            default_token_score=model_args.default_token_score,
            token_scores_temperature=model_args.token_scores_temperature
        )

        # train
        trainer.train(train_dataset=train_dataset, valid_dataset=valid_dataset)

    if training_args.do_predict:
        trainer.model.load_state_dict(
            torch.load(training_args.best_model_file, map_location=training_args.device)
        )

        test_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_filenames=[data_args.test_file],
            tokenizer=tokenizer,
            labels=labels,
            model_type=model_args.model_type,
            pooling_strategy=model_args.pooling_strategy,
            max_seq_length=data_args.max_seq_length,
            augment=data_args.augment_data,
            token_scores_file=model_args.token_scores_file,
            default_token_score=model_args.default_token_score,
            token_scores_temperature=model_args.token_scores_temperature
        )
        preds, _ = trainer.evaluate(dataset=test_dataset)
        assert not os.path.exists(training_args.preds_out_file)
        token_classification_task.write_predictions_to_file(
            predictions=preds,
            orig_filename=data_args.test_file,
            out_filename=training_args.preds_out_file
        )


if __name__ == '__main__':
    main()
