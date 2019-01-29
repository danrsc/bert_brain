from dataclasses import dataclass
from typing import Sequence, Optional, Mapping
import numpy as np


__all__ = ['InputFeatures', 'RawData']


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
