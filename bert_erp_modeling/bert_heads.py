import torch
from torch import nn
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel


__all__ = ['BertForTokenRegression']


class BertForTokenRegression(PreTrainedBertModel):

    def __init__(self, config, num_predictions):
        super(BertForTokenRegression, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regression = nn.Linear(config.hidden_size, num_predictions)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        predictions = self.regression(sequence_output)
        return predictions
