from inspect import signature
from collections import OrderedDict
from itertools import chain
from dataclasses import dataclass, replace, field
from typing import Optional, Sequence, Mapping, Callable, Tuple, Union, Iterable

import numpy as np

from .input_features import InputFeatures, KindData, FieldSpec, RawData, SplitData


__all__ = [
    'PreparedData',
    'PreparedDataView',
    'ResponseKeyKind',
    'DataPreparer',
    'PreprocessorSequenceT',
    'PhasePreprocessorMappingT',
    'PreprocessForkFnT',
    'SplitFunctionT']


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
    word_ids: Optional[np.array] = None


def _make_examples_view(
        examples: Optional[Sequence[InputFeatures]], response_key: str) -> Optional[Sequence[InputFeatures]]:
    return [replace(ex, data_ids=ex.data_ids[response_key]) for ex in examples] if examples is not None else None


def _make_prepared_data_view(prepared_data: PreparedData, response_key: str) -> PreparedDataView:
    return PreparedDataView(
        train=_make_examples_view(prepared_data.train, response_key),
        validation=_make_examples_view(prepared_data.validation, response_key),
        test=_make_examples_view(prepared_data.test, response_key),
        data=prepared_data.data[response_key].data,
        word_ids=prepared_data.data[response_key].word_ids)


def _reconcile_view_examples(
        prepared_data_examples: Sequence[InputFeatures],
        view_examples: Sequence[InputFeatures],
        response_key: str):
    if prepared_data_examples is None:
        return False
    view_examples = dict((ex.unique_id, ex.data_ids) for ex in view_examples) if view_examples is not None else {}
    is_modified = False
    for ex in prepared_data_examples:
        if ex.unique_id not in view_examples:
            # noinspection PyUnresolvedReferences
            ex.data_ids[response_key] = -1 * np.ones_like(ex.data_ids[response_key])
        else:
            is_modified = is_modified or not np.array_equal(ex.data_ids[response_key], view_examples[ex.unique_id])
            # noinspection PyUnresolvedReferences
            ex.data_ids[response_key] = view_examples[ex.unique_id]
    return is_modified


def _reconcile_view(prepared_data: PreparedData, view: PreparedDataView, response_key: str):
    if view.data is None:
        for ex in chain(prepared_data.train, prepared_data.validation, prepared_data.test):
            # noinspection PyUnresolvedReferences
            del ex.data_ids[response_key]
        del prepared_data.data[response_key]
    else:
        is_modified = _reconcile_view_examples(prepared_data.train, view.train, response_key)
        is_modified = is_modified or _reconcile_view_examples(prepared_data.validation, view.validation, response_key)
        is_modified = is_modified or _reconcile_view_examples(prepared_data.test, view.test, response_key)
        if is_modified and prepared_data.data[response_key].word_ids is not None:
            is_word_ids_updated = len(prepared_data.data[response_key].word_ids) != len(view.word_ids)
            if not is_word_ids_updated:
                for word_ids in prepared_data.data[response_key].word_ids:
                    if not np.array_equal(word_ids, view.word_ids):
                        is_word_ids_updated = True
                        break
            if not is_word_ids_updated:
                raise ValueError('If data_ids are modified, word_ids must also be modified')
        prepared_data.data[response_key] = replace(
            prepared_data.data[response_key], data=view.data, word_ids=view.word_ids)


def _copy_examples(examples):
    if examples is None:
        return []
    # data_ids may be modified, so we copy them
    return [replace(
        ex, data_ids=type(ex.data_ids)((k, np.copy(ex.data_ids[k])) for k in ex.data_ids)) for ex in examples]


MetadataT = Optional[Mapping[str, np.array]]
PreprocessorT = Union[
    Callable[[PreparedDataView, MetadataT, np.random.RandomState], PreparedDataView],
    Callable[[PreparedDataView, MetadataT, np.random.RandomState, str, str], PreparedDataView]]
PhasePreprocessorT = Union[PreprocessorT, str, Tuple[str, PreprocessorT]]
PreprocessorSequenceT = Union[PhasePreprocessorT, Sequence[PhasePreprocessorT]]
PhasePreprocessorMappingT = Union[Mapping[str, PreprocessorSequenceT], Iterable[Tuple[str, PreprocessorSequenceT]]]
PreprocessForkFnT = Callable[
    [str, str, Optional[PhasePreprocessorMappingT]],
    Tuple[str, Optional[PreprocessorSequenceT]]]
SplitFunctionT = Callable[
    [RawData, np.random.RandomState],
    Tuple[
        Optional[Sequence[InputFeatures]],
        Optional[Sequence[InputFeatures]],
        Optional[Sequence[InputFeatures]]]]


@dataclass(frozen=True)
class ResponseKeyKind:
    response_key: str
    kind: str


@dataclass(frozen=True)
class DataPreparer:
    seed: int
    corpus_key: str
    response_key_kinds: Sequence[ResponseKeyKind]
    preprocess_dict: Optional[PhasePreprocessorMappingT]
    split_function: Optional[SplitFunctionT] = None
    preprocess_fork_fn: Optional[PreprocessForkFnT] = None
    forked_response_keys: Optional[Sequence[Tuple[str, str]]] = field(init=False)

    def __post_init__(self):
        if self.response_key_kinds is not None:
            if np.ndim(self.response_key_kinds) == 0:
                object.__setattr__(self, 'response_key_kinds', (self.response_key_kinds,))
            else:
                object.__setattr__(self, 'response_key_kinds', tuple(self.response_key_kinds))
        # filter preprocess_dict to the relevant entries and make it immutable
        preprocess_list = list()
        forked_list = list()
        for response_key_kind in self.response_key_kinds:
            preprocessors = DataPreparer._get_preprocessors(
                self.corpus_key, self.preprocess_dict, response_key_kind.response_key, response_key_kind.kind)
            if preprocessors is not None:
                if DataPreparer._is_single_processor(preprocessors):
                    preprocessors = (preprocessors,)
                else:
                    preprocessors = tuple(preprocessors)
            preprocess_list.append((response_key_kind.response_key, preprocessors))
            if self.preprocess_fork_fn is not None:
                forked_name, forked_preprocessors = self.preprocess_fork_fn(
                    response_key_kind.response_key, response_key_kind.kind, preprocessors)
                if forked_name is not None:
                    if forked_preprocessors is not None:
                        if DataPreparer._is_single_processor(forked_preprocessors):
                            forked_preprocessors = (forked_preprocessors,)
                        else:
                            forked_preprocessors = tuple(forked_preprocessors)
                    preprocess_list.append((forked_name, forked_preprocessors))
                    forked_list.append((forked_name, response_key_kind.response_key))
        object.__setattr__(self, 'preprocess_dict', tuple(preprocess_list))
        object.__setattr__(self, 'forked_response_keys', tuple(forked_list))
        if self.split_function is None:
            object.__setattr__(self, 'split_function', SplitData())

    @staticmethod
    def _is_single_processor(x):
        return np.ndim(x) == 0 or (len(x) == 2 and isinstance(x[0], str))

    @staticmethod
    def _get_preprocessors(corpus_key, preprocess_dict, response_key, kind):
        if preprocess_dict is None:
            return None
        if response_key in preprocess_dict:
            return preprocess_dict[response_key]
        if kind in preprocess_dict:
            return preprocess_dict[kind]
        if corpus_key in preprocess_dict:
            return preprocess_dict[corpus_key]
        return None

    def _run_step(self, step, result, metadata, random_state, dataset_path, response_k=None):
        sig = signature(step.__call__)
        parameters = OrderedDict(sig.parameters)
        loaded_data_tuple = _make_prepared_data_view(result, response_k) if response_k is not None else result
        arguments = OrderedDict(
            loaded_data_tuple=loaded_data_tuple,
            metadata=metadata,
            random_state=random_state,
            dataset_path=dataset_path)
        if response_k is not None:
            arguments['data_key'] = response_k
        step_kwargs = OrderedDict()
        for k in parameters:
            if k in arguments:
                step_kwargs[k] = arguments[k]
                del arguments[k]
        for k in step_kwargs:
            del parameters[k]
        step_args = [arguments[k] for i, k in enumerate(arguments) if i < len(parameters)]
        return step(*step_args, **step_kwargs)

    def prepare(self, raw_data: RawData, dataset_path: str) -> Tuple[PreparedData, Optional[Mapping[str, np.ndarray]]]:

        random_state = np.random.RandomState(self.seed)
        metadata = raw_data.metadata
        preprocess_dict = dict(self.preprocess_dict)

        if raw_data.is_pre_split:
            result = PreparedData(
                _copy_examples(raw_data.input_examples),
                _copy_examples(raw_data.validation_input_examples),
                _copy_examples(raw_data.test_input_examples),
                OrderedDict(raw_data.response_data),
                field_specs=raw_data.field_specs)
        else:
            train_input_examples, validation_input_examples, test_input_examples = self.split_function(
                raw_data=raw_data, random_state=random_state)

            result = PreparedData(
                _copy_examples(train_input_examples),
                _copy_examples(validation_input_examples),
                _copy_examples(test_input_examples),
                OrderedDict(raw_data.response_data),
                field_specs=raw_data.field_specs)

        phases = None
        phase_change_steps = dict()
        phase_steps = OrderedDict()

        for forked_name, response_k in self.forked_response_keys:
            if forked_name in result.data:
                raise ValueError('Duplicate name: {}'.format(forked_name))
            result.data[forked_name] = result.data[response_k].copy()
            for ex in chain(result.train, result.validation, result.test):
                # noinspection PyUnresolvedReferences
                ex.data_ids[forked_name] = np.copy(ex.data_ids[response_k])

        if len(result.data) != len(preprocess_dict):
            raise ValueError('Inconsistency between response_keys: {} and actual data keys: {}'.format(
                preprocess_dict, result.data))
        for k in result.data:
            if k not in preprocess_dict:
                raise ValueError('data key {} not found in preprocess_dict'.format(k))

        for response_k in result.data:
            response_phases = None
            phase_steps[response_k] = None
            preprocessors = preprocess_dict[response_k]
            if preprocessors is not None:
                phase_steps[response_k] = [list()]
                for step in preprocessors:
                    name = None
                    if isinstance(step, str):
                        name = step
                        step = None
                    elif isinstance(step, tuple):
                        name, step = step
                    if name is not None:
                        if response_phases is None:
                            response_phases = [name]
                        else:
                            response_phases.append(name)
                        if step is not None:
                            if name in phase_change_steps:
                                if id(phase_change_steps[name]) != id(step):
                                    raise ValueError('Phase change steps must be specified exactly once')
                            else:
                                phase_change_steps[name] = step
                        phase_steps[response_k].append(list())
                    else:
                        phase_steps[response_k][-1].append(step)
                if phases is None:
                    phases = response_phases
                else:
                    if len(phases) != len(response_phases):
                        raise ValueError(
                            'Unequal phases across response types: {}, {}'.format(phases, response_phases))
                    for p, r in zip(phases, response_phases):
                        if p != r:
                            raise ValueError(
                                'Unequal phases across response types: {}, {}'.format(phases, response_phases))

        if phases is None:
            phases = []

        for phase in phases:
            if phase_change_steps[phase] is None:
                raise ValueError('Phase change step is not specified: {}'.format(phase))

        for index_phase in range(len(phases) + 1):
            current_response_keys = list(result.data)
            for response_k in current_response_keys:
                if phase_steps[response_k] is not None:
                    for step in phase_steps[response_k][index_phase]:
                        processed = self._run_step(step, result, metadata, random_state, dataset_path, response_k)
                        _reconcile_view(result, processed, response_k)
            if index_phase < len(phases):
                phase_change_step = phase_change_steps[phases[index_phase]]
                result, metadata = self._run_step(phase_change_step, result, metadata, dataset_path, random_state)

        return result, metadata
