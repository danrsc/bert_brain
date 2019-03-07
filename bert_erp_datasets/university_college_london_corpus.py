import csv
from dataclasses import dataclass
from typing import Optional
import os
from collections import OrderedDict
import numpy as np

from bert_erp_tokenization import bert_tokenize_with_spacy_meta, RawData

__all__ = ['read_frank_2015_erp', 'read_frank_2013', 'frank_2015_erp_data', 'ucl_data']


def read_frank_2015_erp(spacy_tokenize_model, bert_tokenizer, path, return_baseline=False):
    from scipy.io import loadmat
    data = loadmat(path)
    sentences = data['sentences'].squeeze(axis=1)
    erp = data['ERP'].squeeze(axis=1)
    erp_base = data['ERPbase'].squeeze(axis=1) if return_baseline else None

    examples = list()
    rows = list()
    base_rows = list()

    iterable = zip(sentences, erp, erp_base) if erp_base is not None else zip(sentences, erp)

    for i, item in enumerate(iterable):
        if erp_base is not None:
            s, e, e_base = item
        else:
            s, e = item
            e_base = None
        s = s.squeeze(axis=0)

        input_features = bert_tokenize_with_spacy_meta(
            spacy_tokenize_model, bert_tokenizer, i, [str(w[0]) for w in s], len(rows))

        examples.append(input_features)

        for w_e in e:
            rows.append(np.expand_dims(np.asarray(w_e, dtype=np.float32), 0))

        if e_base is not None:
            for w_base in e_base:
                base_rows.append(np.expand_dims(np.asarray(w_base, dtype=np.float32), 0))

    erp = np.concatenate(rows, axis=0)
    if return_baseline:
        erp_base = np.concatenate(base_rows, axis=0)
    erp_names = ['elan', 'lan', 'n400', 'epnp', 'p600', 'pnp']
    erp = dict([(n, np.squeeze(v, axis=2)) for n, v in zip(erp_names, np.split(erp, 6, axis=2))])
    if erp_base is not None:
        erp_base = dict([(n, np.squeeze(v, axis=2)) for n, v in zip(erp_names, np.split(erp_base, 6, axis=2))])

    if return_baseline:
        return examples, erp, erp_base
    return examples, erp


@dataclass(frozen=True)
class _SubjectData:
    subject_id: int
    age: int
    age_en: int
    is_monolingual: Optional[bool]
    gender: str
    hand: str
    correct: float


@dataclass(frozen=True)
class _EyeData:
    subject_id: int
    sentence_id: int
    sentence_position: int
    correct: str
    answer_time: float
    word_pos: int
    word: str
    first_fixation: float
    first_pass: float
    right_bounded: float
    go_past: float


@dataclass(frozen=True)
class _SelfPacedData:
    subject_id: int
    sentence_id: int
    sentence_pos: int
    correct: str
    answer_time: float
    word_pos: int
    word: str
    reading_time: float


def _read_eye_subject_file(path):
    with open(path, 'rt', newline='') as subject_file:
        for record in csv.DictReader(subject_file, delimiter='\t'):
            if record['age'] == 'NA':
                continue
            yield _SubjectData(
                int(record['subj_nr']),
                int(record['age']),
                int(record['age_en']),
                int(record['monoling']) == 1,
                record['sex'],
                record['hand'],
                float(record['correct']))


def _read_self_paced_subject_file(path):
    with open(path, 'rt', newline='') as subject_file:
        for record in csv.DictReader(subject_file, delimiter='\t'):
            yield _SubjectData(
                int(record['subj_nr']),
                int(record['age']),
                int(record['age_en']),
                None,
                record['sex'],
                record['hand'],
                float(record['correct']))


def _read_eye_data(path):
    with open(path, 'rt', newline='') as eye_file:
        for record in csv.DictReader(eye_file, delimiter='\t'):
            yield _EyeData(
                int(record['subj_nr']),
                int(record['sent_nr']),
                int(record['sent_pos']),
                record['correct'],
                float(record['answer_time']),
                int(record['word_pos']),
                record['word'],
                float(record['RTfirstfix']),
                float(record['RTfirstpass']),
                float(record['RTrightbound']),
                float(record['RTgopast']))


def _read_self_paced_data(path):
    with open(path, 'rt', newline='') as self_paced:
        for record in csv.DictReader(self_paced, delimiter='\t'):
            yield _SelfPacedData(
                int(record['subj_nr']),
                int(record['sent_nr']),
                int(record['sent_pos']),
                record['correct'],
                float(record['answer_time']),
                int(record['word_pos']),
                record['word'],
                float(record['RT']))


def read_frank_2013(
        spacy_tokenize_model, bert_tokenizer, path, include_eye=True, self_paced_inclusion='all'):

    eye_subject_ids = set()
    for subject_data in _read_eye_subject_file(os.path.join(path, 'eyetracking.subj.txt')):
        if not subject_data.is_monolingual:
            continue
        if subject_data.age_en > 0:
            continue
        eye_subject_ids.add(subject_data.subject_id)

    eye_keys = ['first_fixation', 'first_pass', 'right_bounded', 'go_past']

    sentence_words_eye = dict()
    data = dict()
    for k in eye_keys:
        data[k] = dict()
    data['reading_time'] = dict()
    seen_eye_subject_ids = set()
    for eye_data in _read_eye_data(os.path.join(path, 'eyetracking.RT.txt')):
        if eye_data.subject_id not in eye_subject_ids:
            continue
        seen_eye_subject_ids.add(eye_data.subject_id)
        if eye_data.sentence_id not in sentence_words_eye:
            sentence_words_eye[eye_data.sentence_id] = dict()
            for k in eye_keys:
                data[k][eye_data.sentence_id] = dict()
        if eye_data.word_pos not in sentence_words_eye[eye_data.sentence_id]:
            sentence_words_eye[eye_data.sentence_id][eye_data.word_pos] = eye_data.word
            for k in eye_keys:
                data[k][eye_data.sentence_id][eye_data.word_pos] = dict()
        data['first_fixation'][eye_data.sentence_id][eye_data.word_pos][eye_data.subject_id] = eye_data.first_fixation
        data['first_pass'][eye_data.sentence_id][eye_data.word_pos][eye_data.subject_id] = eye_data.first_pass
        data['right_bounded'][eye_data.sentence_id][eye_data.word_pos][eye_data.subject_id] = eye_data.right_bounded
        data['go_past'][eye_data.sentence_id][eye_data.word_pos][eye_data.subject_id] = eye_data.go_past

    self_paced_subject_ids = set()
    for subject_data in _read_self_paced_subject_file(os.path.join(path, 'selfpacedreading.subj.txt')):
        if subject_data.age_en > 0:
            continue
        self_paced_subject_ids.add(subject_data.subject_id)

    sentence_words_self_paced = dict()
    seen_self_paced_subject_ids = set()
    for self_paced_data in _read_self_paced_data(os.path.join(path, 'selfpacedreading.RT.txt')):
        if self_paced_data.subject_id not in self_paced_subject_ids:
            continue
        seen_self_paced_subject_ids.add(self_paced_data.subject_id)
        if self_paced_data.sentence_id not in sentence_words_self_paced:
            sentence_words_self_paced[self_paced_data.sentence_id] = dict()
            data['reading_time'][self_paced_data.sentence_id] = dict()
        if self_paced_data.word_pos not in sentence_words_self_paced[self_paced_data.sentence_id]:
            sentence_words_self_paced[self_paced_data.sentence_id][self_paced_data.word_pos] = self_paced_data.word
            data['reading_time'][self_paced_data.sentence_id][self_paced_data.word_pos] = dict()
        data['reading_time'][self_paced_data.sentence_id][self_paced_data.word_pos][self_paced_data.subject_id] = \
            self_paced_data.reading_time

    seen_eye_subject_ids = sorted(seen_eye_subject_ids)
    seen_self_paced_subject_ids = sorted(seen_self_paced_subject_ids)

    all_sentences = dict()
    merged_data = dict()
    for sentence_id in sorted(sentence_words_self_paced):
        sorted_pos = sorted(sentence_words_self_paced[sentence_id])
        words = tuple(sentence_words_self_paced[sentence_id][p] for p in sorted_pos)
        sentence_reading_times = np.full((len(words), len(seen_self_paced_subject_ids)), np.nan)
        for i, p in enumerate(sorted_pos):
            reading_times = data['reading_time'][sentence_id][p]
            for j, subject_id in enumerate(seen_self_paced_subject_ids):
                if subject_id in reading_times:
                    sentence_reading_times[i, j] = reading_times[subject_id]
        if 'reading_time' not in merged_data:
            merged_data['reading_time'] = dict()
        merged_data['reading_time'][len(all_sentences)] = sentence_reading_times
        all_sentences[words] = len(all_sentences)

    for sentence_id in sorted(sentence_words_eye):
        sorted_pos = sorted(sentence_words_eye[sentence_id])
        words = tuple(sentence_words_eye[sentence_id][p] for p in sorted_pos)
        if words not in all_sentences:
            merged_id = len(all_sentences)
            all_sentences[words] = merged_id
        else:
            merged_id = all_sentences[words]
        for k in eye_keys:
            sentence_data = np.full((len(words), len(seen_eye_subject_ids)), np.nan)
            for i, p in enumerate(sorted_pos):
                eye_data = data[k][sentence_id][p]
                for j, subject_id in enumerate(seen_eye_subject_ids):
                    if subject_id in eye_data:
                        sentence_data[i, j] = eye_data[subject_id]
            if k not in merged_data:
                merged_data[k] = dict()
            merged_data[k][merged_id] = sentence_data

    all_sentences = sorted(all_sentences, key=lambda s: all_sentences[s])

    def _iterate_sentence_info():
        for sentence_id_, sentence_ in enumerate(all_sentences):

            should_add_eye_ = any(sentence_id_ in merged_data[k_] for k_ in eye_keys)
            should_add_self_paced_ = sentence_id_ in merged_data['reading_time']

            if self_paced_inclusion == 'eye':
                should_add_self_paced_ = should_add_eye_
            elif self_paced_inclusion is False or self_paced_inclusion == 'none':
                should_add_self_paced_ = False

            if not include_eye:
                should_add_eye_ = False

            yield sentence_id_, should_add_eye_, should_add_self_paced_, sentence_

    filtered_sentences = set()
    for sentence_id, should_add_eye, should_add_self_paced, sentence in _iterate_sentence_info():
        if not should_add_self_paced and not should_add_eye:
            filtered_sentences.add(sentence_id)

    count_words = sum(len(s) for i, s in enumerate(all_sentences) if i not in filtered_sentences)

    merged_data_flat = dict()
    offset = 0
    for sentence_id, should_add_eye, should_add_self_paced, sentence in _iterate_sentence_info():
        if not should_add_self_paced and not should_add_eye:
            continue

        for k in merged_data:
            if k == 'reading_time':
                if not should_add_self_paced:
                    continue
            elif not should_add_eye:
                continue
            if sentence_id in merged_data[k]:
                if k not in merged_data_flat:
                    shape = merged_data[k][sentence_id][0].shape
                    merged_data_flat[k] = np.full((count_words,) + shape, np.nan)
                merged_data_flat[k][np.arange(len(sentence)) + offset] = merged_data[k][sentence_id]

        offset += len(sentence)

    offset = 0
    examples = list()
    for sentence_id, sentence in enumerate(all_sentences):
        if sentence_id in filtered_sentences:
            continue

        input_features = bert_tokenize_with_spacy_meta(
            spacy_tokenize_model, bert_tokenizer, sentence_id, sentence, offset)

        examples.append(input_features)

        offset += len(sentence)

    return examples, merged_data_flat


def ucl_data(spacy_tokenize_model, bert_tokenizer, frank_2013_eye_path, frank_2015_erp_path,
             subtract_erp_baseline=False, include_erp=True, include_eye=True, self_paced_inclusion='all'):

    examples_erp = None
    erp = None
    if include_erp:
        erp_result = read_frank_2015_erp(
            spacy_tokenize_model, bert_tokenizer, frank_2015_erp_path, return_baseline=subtract_erp_baseline)
        if subtract_erp_baseline:
            examples_erp, erp, baseline = erp_result
            for k in erp:
                erp[k] = erp[k] - baseline[k]
        else:
            examples_erp, erp = erp_result

    examples_eye = None
    eye = None
    if include_eye or (self_paced_inclusion is not False and self_paced_inclusion != 'none'):
        examples_eye, eye = read_frank_2013(
            spacy_tokenize_model, bert_tokenizer, frank_2013_eye_path, include_eye, self_paced_inclusion)

    if examples_eye is not None and examples_erp is not None:
        merged_examples = list()
        merge_dict = OrderedDict()
        for i, ex in enumerate(examples_eye):
            key = tuple(ex.tokens)
            merge_dict[key] = (i, None)
        for i, ex in enumerate(examples_erp):
            key = tuple(ex.tokens)
            if key in merge_dict:
                merge_dict[key] = (merge_dict[key][0], i)
            else:
                merge_dict[key] = (None, i)

        merged_data = dict()

        def _num_samples(data_dict):
            if data_dict is None:
                return 0
            for k_ in data_dict:
                return len(data_dict[k_])

        axis_0_size = max(_num_samples(eye), _num_samples(erp))

        offset = 0
        for idx_new, key in enumerate(merge_dict):
            idx_eye, idx_erp = merge_dict[key]
            eye_indices, erp_indices = None, None
            example = examples_eye[idx_eye] if idx_eye is not None else examples_erp[idx_erp]
            example.unique_id = idx_new
            if idx_eye is not None:
                eye_indices = examples_eye[idx_eye].data_ids
                assert(np.array_equal(eye_indices >= 0, example.data_ids >= 0))
                eye_indices = eye_indices[eye_indices >= 0]
                for k in eye:
                    if k not in merged_data:
                        merged_data[k] = np.full((axis_0_size,) + eye[k].shape[1:], np.nan, eye[k].dtype)
                    merged_data[k][np.arange(len(eye_indices)) + offset] = eye[k][eye_indices]
                example.data_ids[example.data_ids >= 0] = np.arange(len(eye_indices)) + offset
            if idx_erp is not None:
                erp_indices = examples_erp[idx_erp].data_ids
                assert (np.array_equal(erp_indices >= 0, example.data_ids >= 0))
                erp_indices = erp_indices[erp_indices >= 0]
                for k in erp:
                    if k not in merged_data:
                        merged_data[k] = np.full((axis_0_size,) + erp[k].shape[1:], np.nan, erp[k].dtype)
                    merged_data[k][np.arange(len(erp_indices)) + offset] = erp[k][erp_indices]
                    example.data_ids[example.data_ids >= 0] = np.arange(len(erp_indices)) + offset

            offset += len(eye_indices) if eye_indices is not None else len(erp_indices)
            merged_examples.append(example)

        examples = merged_examples
        data = merged_data
    elif examples_eye is not None:
        examples = examples_eye
        data = eye
    elif examples_erp is not None:
        examples = examples_erp
        data = erp
    else:
        raise RuntimeError('this should never happen')

    return RawData(examples, data, test_proportion=0., validation_proportion_of_train=0.1)


def frank_2015_erp_data(spacy_tokenize_model, bert_tokenizer, path, subtract_baseline=False):
    result = read_frank_2015_erp(spacy_tokenize_model, bert_tokenizer, path, return_baseline=subtract_baseline)
    if subtract_baseline:
        frank, erp, baseline = result
        for k in erp:
            erp[k] = erp[k] - baseline[k]
    else:
        frank, erp = result

    return RawData(frank, erp, test_proportion=0., validation_proportion_of_train=0.1)
