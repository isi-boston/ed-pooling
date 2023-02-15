from typing import Dict, List
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    BertModel,
    XLMRobertaModel,
    BertForTokenClassification
)

from span_extractor import (
    average_span_extractor,
    weighted_span_extractor,
    attention_pooled_span_extractor,
    last_token_span_extractor,
    first_token_span_extractor
)
from prediction_object import SeqPredictionObject
MODEL_CLASS = {
    'bert': BertModel,
    'xlmr': XLMRobertaModel,
    'bert-token-classification': BertForTokenClassification
}


def get_encoder(model_args):

    encoder = MODEL_CLASS[model_args.model_type].from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path
    )
    if model_args.model_type == 'bert-token-classification':
        encoder = encoder.bert
    return encoder


class TokenClassification(torch.nn.Module):
    def __init__(self, model_args):
        super(TokenClassification, self).__init__()

        self.encoder = get_encoder(model_args)
        self.num_labels = model_args.num_labels
        self.dropout = torch.nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()  # weights?

        self.pooling_strategy = model_args.pooling_strategy

        if self.pooling_strategy == 'attention':
            self.att_pool_query = torch.nn.Linear(self.encoder.config.hidden_size, 1)
            self.att_pool_temp = model_args.token_scores_temperature

        self._build_classifier()

    def _build_classifier(self):
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            self.num_labels
        )

    def encode(
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
    ):

        """Gets encoded sequence from BERT model and pools the layers accordingly.
        BertModel outputs a tuple whose elements are:
        1- Last encoder layer output. Tensor of shape (B, S, H)
        2- Pooled output of the [CLS] token. Tensor of shape (B, H)
        This method uses just the 1st output.
        """

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs[0]

    def predict_logits(
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
            word_offsets=None,
            token_weights=None
    ):
        """Returns the logits prediction from BERT + classifier."""
        seq_out = self.encode(input_ids, attention_mask, token_type_ids)
        seq_out = self.dropout(seq_out)

        if self.pooling_strategy == 'first_token':
            seq_out = first_token_span_extractor(seq_out.contiguous(), word_offsets)
        elif self.pooling_strategy == 'last_token':
            seq_out = last_token_span_extractor(seq_out.contiguous(), word_offsets)
        elif self.pooling_strategy == 'average':
            seq_out = average_span_extractor(seq_out.contiguous(), word_offsets)
        elif self.pooling_strategy in ['idf', 'morph']:
            seq_out = weighted_span_extractor(
                embeddings=seq_out.contiguous(), offsets=word_offsets, weights=token_weights
            )
        elif self.pooling_strategy == 'attention':
            seq_out = attention_pooled_span_extractor(
                embeddings=seq_out.contiguous(),
                offsets=word_offsets,
                attn=self.att_pool_query,
                temperature=self.att_pool_temp
            )
        else:
            raise NotImplementedError
        logits = self.classifier(seq_out)
        return logits

    def predict(
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
            word_offsets=None,
            token_weights=None,
            labels=None,
    ) -> Dict[str, List[SeqPredictionObject]]:
        """Returns the predictions"""

        outputs = {}
        logits = self.predict_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_offsets=word_offsets,
            token_weights=token_weights
        )
        mask = labels != self.loss_fct.ignore_index

        outputs['predictions'] = []
        for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):

            seq_labels = seq_labels[seq_mask].unsqueeze(0)
            seq_logits = seq_logits[0:seq_labels.shape[1], :].unsqueeze(0)
            assert seq_logits.shape[1] == seq_labels.shape[1]

            y_prob, y_pred = torch.max(torch.softmax(seq_logits, dim=-1), dim=-1)
            y_prob = y_prob.tolist()
            y_pred = y_pred.tolist()
            y_prob, y_pred = y_prob[0], y_pred[0]  # first and only in batch
            outputs['predictions'].append(
                SeqPredictionObject(example=None, preds=y_pred, pred_prob=y_prob)
            )
        return outputs

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
            word_offsets=None,
            token_weights=None,
            labels=None,
    ) -> Dict[str, torch.Tensor]:

        """Performs the forward pass of the network.
        Will calculate and return the loss.

        Args:
            input_ids: tensor of input token ids.
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, self.num_labels - 1] or CrossEntropyLoss().ignore_index.
            word_offsets: tensor Shape: [batch_size, num_orig_tokens, 2].
                Maps token indices to a span in input_ids.
                `input_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
                corresponds to the original j-th word from the i-th batch.
            token_weights: tensor Shape: [batch_size, num_orig_tokens, num_of_max_tokens_for_a_word]

        Returns a dict with calculated tensors:
            - "loss" (if `labels` is not `None`)
        """

        outputs = {}
        logits = self.predict_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_offsets=word_offsets,
            token_weights=token_weights
        )

        batch_size = labels.shape[0]
        mask = labels != self.loss_fct.ignore_index
        loss = 0
        for seq_logits, seq_labels, seq_mask in zip(logits, labels, mask):

            seq_labels = seq_labels[seq_mask].unsqueeze(0)
            seq_logits = seq_logits[0:seq_labels.shape[1], :].unsqueeze(0)
            assert seq_logits.shape[1] == seq_labels.shape[1]

            loss += self.loss_fct(seq_logits.view(-1, self.num_labels), seq_labels.view(-1))

        outputs['loss'] = loss / batch_size
        return outputs
