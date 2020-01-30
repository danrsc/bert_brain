import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple
import hashlib
import pickle
import logging

from transformers import BertTokenizer
from spacy.language import Language

from .spacy_token_meta import make_tokenizer_model as make_spacy_language
from .corpus_base import CorpusBase
from .data_preparer import DataPreparer, PhasePreprocessorMappingT, SplitFunctionT, PreprocessForkFnT, ResponseKeyKind
from .data_id_dataset import DataIdDataset

__all__ = ['CorpusDatasetFactory']


logger = logging.getLogger(__name__)
logger.setLevel('INFO')


@dataclass(frozen=True)
class CorpusLoadInfo:
    response_key_kinds: Tuple[ResponseKeyKind, ...]
    true_max_sequence_length: int


@dataclass(frozen=True)
class CorpusDatasetFactory:
    model_tokenizer_name: str = 'bert-base-uncased'
    spacy_language_name: str = 'en_core_web_md'
    cache_path: str = None
    model_tokenizer: BertTokenizer = field(init=False, repr=False, compare=False)
    spacy_language: Language = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            'model_tokenizer',
            BertTokenizer.from_pretrained(self.model_tokenizer_name, cache_dir=self.cache_path, do_lower_case=True))
        object.__setattr__(
            self,
            'spacy_language',
            make_spacy_language(self.spacy_language_name))

    @classmethod
    def _hash_arguments(cls, kwargs):
        hash_ = hashlib.sha256()
        for key in kwargs:
            s = '{}={}'.format(key, kwargs[key])
            hash_.update(s.encode())
        return hash_.hexdigest()

    def maybe_make_data_set_files(
            self,
            seed: int,
            index_run: int,
            corpus: CorpusBase,
            preprocess_dict: Optional[PhasePreprocessorMappingT],
            split_function: Optional[SplitFunctionT] = None,
            preprocess_fork_fn: Optional[PreprocessForkFnT] = None,
            force_cache_miss: bool = False,
            paths_obj=None,
            max_sequence_length: Optional[int] = None) -> str:

        corpus = CorpusBase.replace_paths(corpus, paths_obj, index_run=index_run)
        corpus_load_hash = type(self)._hash_arguments(OrderedDict((k, v) for k, v in [
            ('factory', self),
            ('corpus', corpus)]))

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
                preprocess_fork_fn)

            effective_max_sequence_length = corpus_info.true_max_sequence_length
            if max_sequence_length is not None and max_sequence_length < effective_max_sequence_length:
                effective_max_sequence_length = max_sequence_length

            data_set_hash = type(self)._hash_arguments(
                OrderedDict((k, v) for k, v in [
                    ('factory', self),
                    ('corpus', corpus),
                    ('data_preparer', data_preparer),
                    ('max_sequence_length', effective_max_sequence_length)]))

            data_set_path = os.path.join(corpus.cache_base_path, data_set_hash)
            init_file_path = os.path.join(data_set_path, DataIdDataset.dataset_init_file)

            if os.path.exists(init_file_path):
                logger.info('Using cached {}'.format(corpus.corpus_key))
                return data_set_path

        print('Loading {}...'.format(corpus.corpus_key), end='', flush=True)
        if not os.path.exists(os.path.split(corpus_info_path)[0]):
            os.makedirs(os.path.split(corpus_info_path)[0])
        data, true_max_sequence_length = corpus.load(
            self.spacy_language, self.model_tokenizer, max_sequence_length=max_sequence_length)
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
            preprocess_fork_fn)

        effective_max_sequence_length = corpus_info.true_max_sequence_length
        if max_sequence_length is not None and max_sequence_length < effective_max_sequence_length:
            effective_max_sequence_length = max_sequence_length

        data_set_hash = type(self)._hash_arguments(
            OrderedDict((k, v) for k, v in [
                ('factory', self),
                ('corpus', corpus),
                ('data_preparer', data_preparer),
                ('max_sequence_length', effective_max_sequence_length)]))

        data_set_path = os.path.join(corpus.cache_base_path, data_set_hash)

        data, metadata = data_preparer.prepare(data, data_set_path)
        DataIdDataset.make_dataset_files(data_set_path, corpus.corpus_key, data, metadata)

        print('done')
        return data_set_path
