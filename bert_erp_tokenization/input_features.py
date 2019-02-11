from dataclasses import dataclass
from typing import Sequence, Optional, Mapping, Any

import numpy as np
import torch


__all__ = ['InputFeatures', 'RawData', 'FieldSpec']


@dataclass
class FieldSpec:
    fill_value: Any = None
    tensor_dtype: Any = torch.float
    is_sequence: bool = True

    def __post_init__(self):
        if self.fill_value is None:
            if self.tensor_dtype.is_floating_point:
                self.fill_value = np.nan
            else:
                self.fill_value = 0

    def __eq__(self, other):
        return np.isclose(self.fill_value, other.fill_value, rtol=0, atol=0, equal_nan=True) \
               and self.tensor_dtype == other.tensor_dtype \
               and self.is_sequence == other.is_sequence


@dataclass
class InputFeatures:
    unique_id: int
    tokens: Sequence[str]
    input_ids: Sequence[int]
    input_mask: Sequence[int]
    input_is_stop: Sequence[int]
    input_is_begin_word_pieces: Sequence[int]
    input_type_ids: Sequence[int]
    data_ids: Sequence[int]


@dataclass
class RawData:
    input_examples: Sequence[InputFeatures]
    response_data: Mapping[str, np.array]
    test_input_examples: Optional[Sequence[InputFeatures]] = None
    validation_input_examples: Optional[Sequence[InputFeatures]] = None
    is_pre_split: bool = False
    test_proportion: float = 0.0
    validation_proportion_of_train: float = 0.1
    field_specs: Optional[Mapping[str, FieldSpec]] = None
