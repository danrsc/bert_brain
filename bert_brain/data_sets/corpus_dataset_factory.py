import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple
import hashlib
import pickle
import logging

import numpy as np

from .spacy_token_meta import BertSpacyTokenAligner
from .corpus_base import CorpusBase
from .data_preparer import DataPreparer, PhasePreprocessorMappingT, SplitFunctionT, PreprocessForkFnT, ResponseKeyKind
from .data_id_dataset import DataIdDataset

__all__ = ['CorpusDatasetFactory']


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusLoadInfo:
    response_key_kinds: Tuple[ResponseKeyKind, ...]
    true_max_sequence_length: int


@dataclass(frozen=True)
class CorpusDatasetFactory:
    model_tokenizer_name: str = 'bert-base-uncased'
    spacy_language_name: str = 'en_core_web_md'
    cache_path: str = None
    bert_spacy_token_aligner: BertSpacyTokenAligner = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            'bert_spacy_token_aligner',
            BertSpacyTokenAligner(self.model_tokenizer_name, self.spacy_language_name, self.cache_path))

    @classmethod
    def _hash_arguments(cls, kwargs):
        hash_ = hashlib.sha256()
        for key in kwargs:
            s = '{}={}'.format(key, kwargs[key])
            hash_.update(s.encode())
        return hash_.hexdigest()

    def maybe_make_data_set_files(
            self,
            index_run: int,
            corpus: CorpusBase,
            preprocess_dict: Optional[PhasePreprocessorMappingT],
            split_function: Optional[SplitFunctionT] = None,
            preprocess_fork_fn: Optional[PreprocessForkFnT] = None,
            force_cache_miss: bool = False,
            paths_obj=None,
            max_sequence_length: Optional[int] = None,
            use_meta_train: bool = False,
            paths_only: bool = False) -> str:

        corpus = CorpusBase.replace_paths(corpus, paths_obj, index_run=index_run)
        corpus_load_hash = type(self)._hash_arguments(OrderedDict((k, v) for k, v in [
            ('factory', self),
            ('corpus', corpus)]))

        # allow arbitrary type for run_info, as long as it produces a reasonable string
        # and use it to create a dataset seed for things that need randomization like
        # splits and preprocessors
        hash_ = hashlib.sha256('{}'.format(corpus.run_info).encode())
        seed = np.frombuffer(hash_.digest(), dtype='uint32')
        random_state = np.random.RandomState(seed)
        seed = random_state.randint(low=0, high=np.iinfo('uint32').max)

        corpus_info_path = os.path.join(corpus.cache_base_path, corpus_load_hash, 'corpus_info.pkl')
        if os.path.exists(corpus_info_path) and not force_cache_miss:
            with open(corpus_info_path, 'rb') as corpus_info_file:
                corpus_info: CorpusLoadInfo = pickle.load(corpus_info_file)

            data_preparer = DataPreparer(
                seed,
                corpus.corpus_key,
                corpus_info.response_key_kinds,
                preprocess_dict,
                split_function,
                preprocess_fork_fn,
                use_meta_train)

            effective_max_sequence_length = corpus_info.true_max_sequence_length
            if max_sequence_length is not None and max_sequence_length < effective_max_sequence_length:
                effective_max_sequence_length = max_sequence_length

            data_set_hash = type(self)._hash_arguments(
                OrderedDict((k, v) for k, v in [
                    ('factory', self),
                    ('corpus', corpus),
                    ('data_preparer', data_preparer),
                    ('max_sequence_length', effective_max_sequence_length),
                    ('use_meta_train', use_meta_train)]))

            data_set_path = os.path.join(corpus.cache_base_path, data_set_hash)
            init_file_path = os.path.join(data_set_path, DataIdDataset.dataset_init_file)

            if os.path.exists(init_file_path):
                logger.info('Using cached {}'.format(corpus.corpus_key))
                return data_set_path

        if paths_only:
            return ''
        print('Loading {}...'.format(corpus.corpus_key), end='', flush=True)
        if not os.path.exists(os.path.split(corpus_info_path)[0]):
            os.makedirs(os.path.split(corpus_info_path)[0])
        data, true_max_sequence_length = corpus.load(
            self.bert_spacy_token_aligner, max_sequence_length=max_sequence_length, use_meta_train=use_meta_train)
        corpus_info = CorpusLoadInfo(
            response_key_kinds=tuple(ResponseKeyKind(k, data.response_data[k].kind) for k in data.response_data),
            true_max_sequence_length=true_max_sequence_length)
        with open(corpus_info_path, 'wb') as corpus_info_file:
            pickle.dump(corpus_info, corpus_info_file, protocol=pickle.HIGHEST_PROTOCOL)

        data_preparer = DataPreparer(
            seed,
            corpus.corpus_key,
            corpus_info.response_key_kinds,
            preprocess_dict,
            split_function,
            preprocess_fork_fn,
            use_meta_train)

        effective_max_sequence_length = corpus_info.true_max_sequence_length
        if max_sequence_length is not None and max_sequence_length < effective_max_sequence_length:
            effective_max_sequence_length = max_sequence_length

        data_set_hash = type(self)._hash_arguments(
            OrderedDict((k, v) for k, v in [
                ('factory', self),
                ('corpus', corpus),
                ('data_preparer', data_preparer),
                ('max_sequence_length', effective_max_sequence_length),
                ('use_meta_train', use_meta_train)]))

        data_set_path = os.path.join(corpus.cache_base_path, data_set_hash)

        data, metadata = data_preparer.prepare(data, data_set_path)
        DataIdDataset.make_dataset_files(data_set_path, corpus.corpus_key, data, metadata)

        print('done')
        return data_set_path
