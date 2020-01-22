import os
import json
import pickle
from transformers import BertConfig


__all__ = ['BertMultiPredictionHeadConfig']


HEAD_GRAPH_PARTS_PICKLE = 'head_graph_parts.pkl'


class BertMultiPredictionHeadConfig(BertConfig):

    model_type = 'bert_multi_prediction_head'

    def __init__(
            self,
            head_graph_parts,
            token_supplemental_key_to_shape=None,
            token_supplemental_skip_dropout_keys=None,
            pooled_supplemental_key_to_shape=None,
            pooled_supplemental_skip_dropout_keys=None,
            **kwargs):
        super().__init__(**kwargs)
        self.head_graph_parts = head_graph_parts
        self.token_supplemental_key_to_shape = token_supplemental_key_to_shape
        self.token_supplemental_skip_dropout_keys = token_supplemental_skip_dropout_keys
        self.pooled_supplemental_key_to_shape = pooled_supplemental_key_to_shape
        self.pooled_supplemental_skip_dropout_keys = pooled_supplemental_skip_dropout_keys

    @classmethod
    def from_json_file(cls, json_file: str):
        dict_obj = cls._dict_from_json_file(json_file)
        return cls(**dict_obj)

    @classmethod  # upcoming version of transformers uses this call
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        dict_obj = json.loads(text)
        head_graph_parts_file = os.path.join(os.path.split(json_file)[0], HEAD_GRAPH_PARTS_PICKLE)
        head_graph_parts = None
        if os.path.exists(head_graph_parts_file):
            with open(head_graph_parts_file, 'rb') as f:
                head_graph_parts = pickle.load(f)
        dict_obj['head_graph_parts'] = head_graph_parts
        return dict_obj

    def to_json_string(self):
        d = self.to_dict()
        if 'head_graph_parts' in d:
            del d['head_graph_parts']
        return json.dumps(d, indent=2, sort_keys=True) + "\n"

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)
        with open(os.path.join(save_directory, HEAD_GRAPH_PARTS_PICKLE), 'wb') as f:
            pickle.dump(self.head_graph_parts, f, protocol=pickle.HIGHEST_PROTOCOL)
