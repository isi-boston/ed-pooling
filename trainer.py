import os
import logging
from typing import Any, Dict, List, Tuple, Union
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)
from transformers import AdamW, get_linear_schedule_with_warmup
from tasks.task_base import InputExample
from prediction_object import SeqPredictionObject


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            args,
            data_collator=None,
            compute_metrics=None):

        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.optimizer = None
        self.lr_scheduler = None

    def _create_optimizer_and_scheduler(
            self,
            num_training_steps: int
    ):

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.args.freeze_embeddings:
            for param in list(self.model.encoder.embeddings.parameters()):
                param.requires_grad = False
            logging.info(
                f"Froze Embedding Layer : "
                f"{[n for n, _ in self.model.encoder.embeddings.named_parameters()]}"
            )

        # freeze_layers is a string "1,2,3" representing layer number
        if self.args.freeze_layers is not "":
            layer_indexes = [int(x) for x in self.args.freeze_layers.split(",")]
            for layer_idx in layer_indexes:
                for param in list(self.model.encoder.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
                logging.info(f"Froze Layer: {layer_idx}")

        if self.args.bitfit:
            # freeze all non-bias transformer parameters
            for name, param in self.model.encoder.named_parameters():
                if "bias" not in name:
                    param.requires_grad = False

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
        )

        warmup_steps = int(num_training_steps * self.args.warmup_proportion)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

    def _get_train_data_loader(self, train_dataset: Dataset):
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=RandomSampler(train_dataset),
            collate_fn=self.data_collator,
        )

    def _get_evaluation_data_loader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.args.valid_batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=self.data_collator,
        )

    def _prepare_inputs(
            self,
            inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[List[InputExample], Dict[str, Union[torch.Tensor, Any]]]:

        tensor_dict = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                tensor_dict[k] = v.to(self.args.device)
        examples = inputs['examples']
        return examples, tensor_dict

    def _training_step(
            self,
            inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:

        self.model.train()
        _, inputs = self._prepare_inputs(inputs)
        outputs = self.model(**inputs)
        loss = outputs['loss']
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    def _prediction_step(
            self,
            inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> List[SeqPredictionObject]:

        examples, inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = self.model.predict(**inputs)
            seq_preds = outputs['predictions']

        assert len(seq_preds) == len(examples)
        for ex, sp in zip(examples, seq_preds):
            sp.example = ex
        return seq_preds

    def _prediction_loop(
            self,
            dataloader: DataLoader,
            description: str
    ):

        logging.info(f"***** Running {description} *****")
        logging.info(f"  Num batches = {len(dataloader)}")
        logging.info(f"  Batch size = {dataloader.batch_size}")

        preds: List[SeqPredictionObject] = []
        self.model.eval()
        for batch_idx, batch in enumerate(dataloader):
            seq_preds = self._prediction_step(batch)
            preds.extend(seq_preds)

        try:
            metrics = self.compute_metrics([p.preds for p in preds], [p.example for p in preds])
        except:
            logging.warning('At test time, the gold data can be empty.')
            metrics = None
        return preds, metrics

    def dump_best_metrics(self, metrics) -> None:
        metrics_file = os.path.join(self.args.output_dir, 'best_metrics.csv')
        f1 = metrics.f1  # only dumping f1
        with open(metrics_file, 'w', encoding="utf-8") as f:
            print(f"f1,{f1}", file=f)

    def train(
            self,
            train_dataset: Dataset,
            valid_dataset: Dataset,
    ) -> None:

        train_dataloader = self._get_train_data_loader(train_dataset)

        # TODO: Check gradient_accumulation step logic
        if self.args.gradient_accumulation_steps != 1:
            raise NotImplementedError(f'Must have {self.args.gradient_accumulation_steps} == 1')
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
        self._create_optimizer_and_scheduler(num_training_steps=t_total)

        logging.info("***** Running training *****")
        logging.info(f" Num batches = {len(train_dataloader)}")
        logging.info(f" Num Epochs = {self.args.num_train_epochs}")
        logging.info(f" Batch size per device = {self.args.train_batch_size}")
        logging.info(f" Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logging.info(f" Total optimization steps = {t_total}")

        self.model.zero_grad()
        best_metrics = None
        for epoch_idx in range(self.args.num_train_epochs):

            tr_loss = torch.tensor(0.0).to(self.args.device)
            for batch_idx, batch in enumerate(train_dataloader):
                tr_loss += self._training_step(batch)

                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0 \
                        or batch_idx == len(train_dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad()

            average_loss = tr_loss.item() / len(train_dataloader)
            logging.info(f'epoch {epoch_idx} : average loss = {average_loss}')

            try:
                metrics = self.evaluate(valid_dataset)[-1]
            except:
                logging.warning('Can happen that some training file has no data.')
                metrics = None

            if best_metrics is None or (
                    metrics is not None and metrics.is_better_than(best_metrics)
            ):
                best_metrics = metrics
                torch.save(self.model.state_dict(), self.args.best_model_file)
                self.dump_best_metrics(best_metrics)

    # save model/optimizer/scheduler for continued training

    def evaluate(self, dataset: Dataset):
        data_loader = self._get_evaluation_data_loader(dataset)
        preds, metrics = self._prediction_loop(
            data_loader,
            description="Evaluation"
        )
        logging.info(metrics)
        return preds, metrics
