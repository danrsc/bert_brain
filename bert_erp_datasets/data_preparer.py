from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Mapping, Callable, Tuple

import numpy as np

from .input_features import InputFeatures, KindData, FieldSpec, RawData, split_data


__all__ = ['PreparedData', 'PreparedDataView', 'DataPreparer']


@dataclass(frozen=True)
class PreparedData:
    train: Optional[Sequence[InputFeatures]] = None
    validation: Optional[Sequence[InputFeatures]] = None
    test: Optional[Sequence[InputFeatures]] = None
    data: Optional[Mapping[str, KindData]] = None
    field_specs: Optional[Mapping[str, FieldSpec]] = None


@dataclass(frozen=True)
class PreparedDataView:
    train: Optional[Sequence[InputFeatures]] = None
    validation: Optional[Sequence[InputFeatures]] = None
    test: Optional[Sequence[InputFeatures]] = None
    data: Optional[np.array] = None


def _make_examples_view(
        examples: Optional[Sequence[InputFeatures]], response_key: str) -> Optional[Sequence[InputFeatures]]:
    return [replace(ex, data_ids=ex.data_ids[response_key]) for ex in examples] if examples is not None else None


def _make_prepared_data_view(prepared_data: PreparedData, response_key: str) -> PreparedDataView:
    return PreparedDataView(
        train=_make_examples_view(prepared_data.train, response_key),
        validation=_make_examples_view(prepared_data.validation, response_key),
        test=_make_examples_view(prepared_data.test, response_key),
        data=prepared_data.data[response_key].data)


def _reconcile_view_examples(
        prepared_data_examples: Sequence[InputFeatures],
        view_examples: Sequence[InputFeatures],
        response_key: str):
    if prepared_data_examples is None:
        return
    view_examples = dict((ex.unique_id, ex.data_ids) for ex in view_examples) if view_examples is not None else {}
    for ex in prepared_data_examples:
        if ex.unique_id not in view_examples:
            ex.data_ids[response_key] = -1 * np.ones_like(ex.data_ids[response_key])
        else:
            ex.data_ids[response_key] = view_examples[ex.unique_id]


def _reconcile_view(prepared_data: PreparedData, view: PreparedDataView, response_key: str):
    _reconcile_view_examples(prepared_data.train, view.train, response_key)
    _reconcile_view_examples(prepared_data.validation, view.validation, response_key)
    _reconcile_view_examples(prepared_data.test, view.test, response_key)
    prepared_data.data[response_key] = replace(prepared_data.data[response_key], data=view.data)


def _copy_examples(examples):
    if examples is None:
        return examples
    # data_ids may be modified, so we copy them
    return [replace(
        ex, data_ids=type(ex.data_ids)((k, np.copy(ex.data_ids[k])) for k in ex.data_ids)) for ex in examples]


class DataPreparer(object):

    def __init__(
            self,
            seed: int,
            preprocess_dict: Mapping[str, Callable[[PreparedData, Optional[Mapping[str, np.array]]], PreparedData]],
            split_function_dict: Mapping[
                str, Callable[
                    [RawData, np.random.RandomState],
                    Tuple[
                        Optional[Sequence[InputFeatures]],
                        Optional[Sequence[InputFeatures]],
                        Optional[Sequence[InputFeatures]]]]]):
        self._seed = seed
        self._random_state = dict()
        self._prepared_cache = dict()
        self._preprocess_dict = dict(preprocess_dict) if preprocess_dict is not None else None
        self._split_function_dict = dict(split_function_dict) if split_function_dict is not None else None

    def prepare(self, raw_data_dict: Mapping[str, RawData]) -> Mapping[str, PreparedData]:
        result = OrderedDict()
        metadata = OrderedDict()

        for k in raw_data_dict:
            metadata[k] = raw_data_dict[k].metadata
            if raw_data_dict[k].is_pre_split:
                result[k] = PreparedData(
                    _copy_examples(raw_data_dict[k].input_examples),
                    _copy_examples(raw_data_dict[k].validation_input_examples),
                    _copy_examples(raw_data_dict[k].test_input_examples),
                    OrderedDict(raw_data_dict[k].response_data),
                    field_specs=raw_data_dict[k].field_specs)
            elif (self._split_function_dict is not None
                    and k in self._split_function_dict and self._split_function_dict[k] is not None):
                if k not in self._random_state:
                    self._random_state[k] = np.random.RandomState(self._seed)

                train_input_examples, validation_input_examples, test_input_examples = self._split_function_dict[k](
                    raw_data=raw_data_dict[k], random_state=self._random_state[k])

                result[k] = PreparedData(
                    _copy_examples(train_input_examples),
                    _copy_examples(validation_input_examples),
                    _copy_examples(test_input_examples),
                    OrderedDict(raw_data_dict[k].response_data),
                    field_specs=raw_data_dict[k].field_specs)
            else:
                if k not in self._random_state:
                    self._random_state[k] = np.random.RandomState(self._seed)

                train_input_examples, validation_input_examples, test_input_examples = split_data(
                    raw_data_dict[k].input_examples,
                    raw_data_dict[k].test_proportion,
                    raw_data_dict[k].validation_proportion_of_train,
                    random_state=self._random_state[k])

                result[k] = PreparedData(
                    _copy_examples(train_input_examples),
                    _copy_examples(validation_input_examples),
                    _copy_examples(test_input_examples),
                    OrderedDict(raw_data_dict[k].response_data),
                    field_specs=raw_data_dict[k].field_specs)

        def _get_preprocessor(preprocess_dict, corpus_key, response_key, kind):
            if preprocess_dict is None:
                return None
            if response_key in preprocess_dict:
                return preprocess_dict[response_key]
            if kind in preprocess_dict:
                return preprocess_dict[kind]
            if corpus_key in preprocess_dict:
                return preprocess_dict[corpus_key]
            return None

        for k in result:
            needs_preprocess = not raw_data_dict[k].is_pre_split or k not in self._prepared_cache
            if needs_preprocess:
                for response_k in result[k].data:
                    preprocessor = _get_preprocessor(
                        self._preprocess_dict, k, response_k, result[k].data[response_k].kind)
                    if preprocessor is not None:
                        processed = preprocessor(_make_prepared_data_view(result[k], response_k), metadata[k])
                        _reconcile_view(result[k], processed, response_k)
                if raw_data_dict[k].is_pre_split:
                    self._prepared_cache[k] = result[k]
            elif raw_data_dict[k].is_pre_split:
                result[k] = self._prepared_cache[k]

        return result
