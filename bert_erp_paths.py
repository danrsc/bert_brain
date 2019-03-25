import os
from dataclasses import dataclass
from typing import Optional
from bert_erp_datasets import CorpusLoader


@dataclass
class Paths:

    default_relative_pre_trained_path = 'uncased_L-12_H-768_A-12'
    default_relative_cache_path = 'bert_cache'
    default_relative_geco_path = os.path.join('geco', 'MonolingualReadingData.csv')
    default_relative_bnc_root = os.path.join('british_national_corpus', '2553', 'download', 'Texts')
    default_relative_harry_potter_path = 'harry_potter'
    default_relative_frank_2015_erp_path = os.path.join('frank_2015_erp', 'stimuli_erp.mat')
    default_relative_frank_2013_eye_path = 'frank_2013_eye'
    default_relative_dundee_path = 'dundee'
    default_relative_english_web_universal_dependencies_v_1_2_path = os.path.join(
        'universal-dependencies-1.2', 'UD_English')
    default_relative_english_web_universal_dependencies_v_2_3_path = os.path.join(
        'universal-dependencies-2.3', 'ud-treebanks-v2.3', 'UD_English-EWT')
    default_relative_proto_roles_english_web_path = os.path.join(
        'protoroles_eng_udewt', 'protoroles_eng_ud1.2_11082016.tsv')
    default_relative_proto_roles_prop_bank_path = os.path.join(
        'protoroles_eng_pb', 'protoroles_eng_pb_08302015.tsv')
    default_relative_ontonotes_path = os.path.join('conll-formatted-ontonotes-5.0', 'data')
    default_relative_natural_stories_path = 'natural_stories'
    default_relative_linzen_agreement_path = 'linzen_agreement'

    pre_trained_base_path: str = '/share/volume0/drschwar/BERT'
    result_path: str = '/share/volume0/drschwar/bert_erp/results/'
    data_set_base_path: str = '/share/volume0/drschwar/data_sets/'
    model_path: str = '/share/volume0/drschwar/bert_erp/models/'

    pre_trained_path: Optional[str] = None
    cache_path: Optional[str] = None
    geco_path: Optional[str] = None
    bnc_root: Optional[str] = None
    harry_potter_path: Optional[str] = None
    frank_2015_erp_path: Optional[str] = None
    frank_2013_eye_path: Optional[str] = None
    dundee_path: Optional[str] = None
    english_web_universal_dependencies_v_1_2_path: Optional[str] = None
    english_web_universal_dependencies_v_2_3_path: Optional[str] = None
    proto_roles_english_web_path: Optional[str] = None
    proto_roles_prop_bank_path: Optional[str] = None
    ontonotes_path: Optional[str] = None
    natural_stories_path: Optional[str] = None
    linzen_agreement_path: Optional[str] = None

    def __post_init__(self):
        if self.pre_trained_path is None:
            self.pre_trained_path = os.path.join(self.pre_trained_base_path, Paths.default_relative_pre_trained_path)
        if self.cache_path is None:
            self.cache_path = os.path.join(self.data_set_base_path, Paths.default_relative_cache_path)
        if self.geco_path is None:
            self.geco_path = os.path.join(self.data_set_base_path, Paths.default_relative_geco_path)
        if self.bnc_root is None:
            self.bnc_root = os.path.join(self.data_set_base_path, Paths.default_relative_bnc_root)
        if self.harry_potter_path is None:
            self.harry_potter_path = os.path.join(self.data_set_base_path, Paths.default_relative_harry_potter_path)
        if self.frank_2015_erp_path is None:
            self.frank_2015_erp_path = os.path.join(self.data_set_base_path, Paths.default_relative_frank_2015_erp_path)
        if self.frank_2013_eye_path is None:
            self.frank_2013_eye_path = os.path.join(self.data_set_base_path, Paths.default_relative_frank_2013_eye_path)
        if self.dundee_path is None:
            self.dundee_path = os.path.join(self.data_set_base_path, Paths.default_relative_dundee_path)
        if self.english_web_universal_dependencies_v_1_2_path is None:
            self.english_web_universal_dependencies_v_1_2_path = os.path.join(
                self.data_set_base_path, Paths.default_relative_english_web_universal_dependencies_v_1_2_path)
        if self.english_web_universal_dependencies_v_2_3_path is None:
            self.english_web_universal_dependencies_v_2_3_path = os.path.join(
                self.data_set_base_path, Paths.default_relative_english_web_universal_dependencies_v_2_3_path)
        if self.proto_roles_english_web_path is None:
            self.proto_roles_english_web_path = os.path.join(
                self.data_set_base_path, Paths.default_relative_proto_roles_english_web_path)
        if self.proto_roles_prop_bank_path is None:
            self.proto_roles_prop_bank_path = os.path.join(
                self.data_set_base_path, Paths.default_relative_proto_roles_prop_bank_path)
        if self.ontonotes_path is None:
            self.ontonotes_path = os.path.join(self.data_set_base_path, Paths.default_relative_ontonotes_path)
        if self.natural_stories_path is None:
            self.natural_stories_path = os.path.join(
                self.data_set_base_path, Paths.default_relative_natural_stories_path)
        if self.linzen_agreement_path is None:
            self.linzen_agreement_path = os.path.join(
                self.data_set_base_path, Paths.default_relative_linzen_agreement_path)

    def make_corpus_loader(
            self, bert_pre_trained_model_name='bert-base-uncased', corpus_key_kwarg_dict=None):
        return CorpusLoader(
            bert_pre_trained_model_name,
            self.cache_path, self.geco_path, self.bnc_root, self.harry_potter_path, self.frank_2013_eye_path,
            self.frank_2015_erp_path, self.dundee_path, self.english_web_universal_dependencies_v_1_2_path,
            self.english_web_universal_dependencies_v_2_3_path, self.proto_roles_english_web_path, self.ontonotes_path,
            self.proto_roles_prop_bank_path, self.natural_stories_path, self.linzen_agreement_path,
            corpus_key_kwarg_dict=corpus_key_kwarg_dict)
