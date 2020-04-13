import os
from dataclasses import dataclass, field, fields
from typing import Optional


def _path_field(default_relative):
    return field(default=None, metadata=dict(default_relative=default_relative))


@dataclass
class Paths:

    pre_trained_base_path: str = '/share/volume0/drschwar/BERT'
    result_path: str = '/share/volume0/drschwar/bert_erp/results/'
    data_set_base_path: str = '/share/volume0/drschwar/data_sets/'
    model_path: str = '/share/volume0/drschwar/bert_erp/models/'

    pre_trained_path: Optional[str] = _path_field('uncased_L-12_H-768_A-12')
    cache_path: Optional[str] = _path_field('bert_cache')
    geco_path: Optional[str] = _path_field(os.path.join('geco', 'MonolingualReadingData.csv'))
    bnc_root: Optional[str] = _path_field(os.path.join('british_national_corpus', '2553', 'download', 'Texts'))
    harry_potter_path: Optional[str] = _path_field('harry_potter')
    frank_2015_erp_path: Optional[str] = _path_field(os.path.join('frank_2015_erp', 'stimuli_erp.mat'))
    frank_2013_eye_path: Optional[str] = _path_field('frank_2013_eye')
    dundee_path: Optional[str] = _path_field('dundee')
    english_web_universal_dependencies_v_1_2_path: Optional[str] = _path_field(
        os.path.join('universal-dependencies-1.2', 'UD_English'))
    english_web_universal_dependencies_v_2_3_path: Optional[str] = _path_field(
        os.path.join('universal-dependencies-2.3', 'ud-treebanks-v2.3', 'UD_English-EWT'))
    proto_roles_english_web_path: Optional[str] = _path_field(
        os.path.join('protoroles_eng_udewt', 'protoroles_eng_ud1.2_11082016.tsv'))
    proto_roles_prop_bank_path: Optional[str] = _path_field(
        os.path.join('protoroles_eng_pb', 'protoroles_eng_pb_08302015.tsv'))
    ontonotes_path: Optional[str] = _path_field(os.path.join('conll-formatted-ontonotes-5.0', 'data'))
    natural_stories_path: Optional[str] = _path_field('natural_stories')
    linzen_agreement_path: Optional[str] = _path_field('linzen_agreement')
    glue_path: Optional[str] = _path_field(os.path.join('GLUE', 'glue_data'))
    stanford_sentiment_treebank_path: Optional[str] = _path_field(os.path.join('GLUE', 'glue_data', 'SST-2'))

    # SuperGLUE
    boolq_path: Optional[str] = _path_field(os.path.join('SuperGLUE', 'BoolQ'))
    commitment_bank_path: Optional[str] = _path_field(os.path.join('SuperGLUE', 'CB'))
    choice_of_plausible_alternatives_path: Optional[str] = _path_field(os.path.join('SuperGLUE', 'COPA'))
    multi_sentence_reading_comprehension_path: Optional[str] = _path_field(os.path.join('SuperGLUE', 'MultiRC'))
    reading_comprehension_with_common_sense_reasoning_path: Optional[str] = _path_field(
        os.path.join('SuperGLUE', 'ReCoRD'))
    recognizing_textual_entailment_path: Optional[str] = _path_field(os.path.join('SuperGLUE', 'RTE'))
    word_in_context_path: Optional[str] = _path_field(os.path.join('SuperGLUE', 'WiC'))
    winograd_schema_challenge_path: Optional[str] = _path_field(os.path.join('SuperGLUE', 'WSC'))

    # What you can cram into a single vector
    bigram_shift_path: Optional[str] = _path_field(os.path.join('what_you_can_cram', 'bigram_shift.txt'))
    coordination_inversion_path: Optional[str] = _path_field(
        os.path.join('what_you_can_cram', 'coordination_inversion.txt'))
    object_number_path: Optional[str] = _path_field(os.path.join('what_you_can_cram', 'obj_number.txt'))
    semantic_odd_man_out_path: Optional[str] = _path_field(os.path.join('what_you_can_cram', 'odd_man_out.txt'))
    sentence_length_path: Optional[str] = _path_field(os.path.join('what_you_can_cram', 'sentence_length.txt'))
    subject_number_path: Optional[str] = _path_field(os.path.join('what_you_can_cram', 'subj_number.txt'))
    top_constituents_path: Optional[str] = _path_field(os.path.join('what_you_can_cram', 'top_constituents.txt'))
    tree_depth_path: Optional[str] = _path_field(os.path.join('what_you_can_cram', 'tree_depth.txt'))
    verb_tense_path: Optional[str] = _path_field(os.path.join('what_you_can_cram', 'past_present.txt'))
    word_content_path: Optional[str] = _path_field(os.path.join('what_you_can_cram', 'word_content.txt'))

    # Edge probing
    edge_probing_base = os.path.join('jiant_edge_probing', 'edges')
    part_of_speech_conll_2012_path: Optional[str] = _path_field(
        os.path.join(edge_probing_base, 'ontonotes', 'const', 'pos'))
    constituents_conll_2012_path: Optional[str] = _path_field(
        os.path.join(edge_probing_base, 'ontonotes', 'const', 'nonterminal'))
    semantic_role_label_conll_2012_path: Optional[str] = _path_field(
        os.path.join(edge_probing_base, 'ontonotes', 'srl'))
    named_entity_recognition_conll_2012_path: Optional[str] = _path_field(
        os.path.join(edge_probing_base, 'ontonotes', 'ner'))
    coreference_conll_2012_path: Optional[str] = _path_field(
        os.path.join(edge_probing_base, 'ontonotes', 'coref'))
    dependencies_english_web_path: Optional[str] = _path_field(os.path.join(edge_probing_base, 'dep_ewt'))
    definite_pronoun_resolution_path: Optional[str] = _path_field(os.path.join(edge_probing_base, 'dpr'))
    sem_eval_path: Optional[str] = _path_field(os.path.join(edge_probing_base, 'semeval'))
    semantic_proto_roles_1_path: Optional[str] = _path_field(os.path.join(edge_probing_base, 'spr1'))
    semantic_proto_roles_2_path: Optional[str] = _path_field(os.path.join(edge_probing_base, 'spr2'))

    def __post_init__(self):
        for f in fields(self):
            if f.name == 'pre_trained_path' and getattr(self, f.name) is None:
                setattr(self, f.name, os.path.join(self.pre_trained_base_path, f.metadata['default_relative']))
            elif getattr(self, f.name) is None and f.metadata is not None and 'default_relative' in f.metadata:
                setattr(self, f.name, os.path.join(self.data_set_base_path, f.metadata['default_relative']))
