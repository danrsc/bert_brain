from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel


__all__ = ['BertForTokenRegression']


class BertForTokenRegression(PreTrainedBertModel):

    def __init__(self, config, prediction_key_to_shape, additional_input_keys_to_shapes=None):
        super(BertForTokenRegression, self).__init__(config)
        self.bert = BertModel(config)

        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.prediction_key_to_shape = OrderedDict(prediction_key_to_shape)
        self.addtional_input_keys_to_shapes = None
        if additional_input_keys_to_shapes is not None:
            self.addtional_input_keys_to_shapes = OrderedDict(additional_input_keys_to_shapes)
            additional_input_count = sum(
                int(np.prod(self.addtional_input_keys_to_shapes[k])) for k in
                self.addtional_input_keys_to_shapes)
        else:
            additional_input_count = 0
        self.splits = [max(1, int(np.prod(self.prediction_key_to_shape[k]))) for k in self.prediction_key_to_shape]
        self.regression = nn.Linear(config.hidden_size + additional_input_count, sum(self.splits))
        self.apply(self.init_bert_weights)

    def forward(self, batch):
        sequence_output, _ = self.bert(
            batch['input_ids'],
            token_type_ids=batch['input_type_ids'] if 'input_type_ids' in batch else None,
            attention_mask=batch['input_mask'] if 'input_mask' in batch else None,
            output_all_encoded_layers=False)
        if self.addtional_input_keys_to_shapes is not None:
            all_values = [sequence_output]
            for key in self.addtional_input_keys_to_shapes:
                values = batch[key]
                values = values.view(
                    values.size()[:2] + (int(np.prod(self.addtional_input_keys_to_shapes[key])),))
                all_values.append(values.type(sequence_output.dtype))
            sequence_output = torch.cat(all_values, dim=2)
        predictions = self.regression(sequence_output)
        predictions = torch.split(predictions, self.splits, dim=-1)
        predictions = OrderedDict(
            (k, p.view(p.size()[:2] + self.prediction_key_to_shape[k]))
            for k, p in zip(self.prediction_key_to_shape, predictions))
        return predictions
